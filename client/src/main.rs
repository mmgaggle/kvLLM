// kllm-chat — a client for the kLLM character device (Granite-backed).
//
// The device (/dev/kllm) speaks token IDs; the BPE tokenizer and chat template
// live here in userspace. A request is a single descriptor transaction:
//
//     write:  "<n_new> <temp> <id> <id> ..."
//     read:   "<id> <id> ...\n# prefill_skipped ... | ... | store_total N\n"
//
// One-shot:    kllm-chat "a question"
// Interactive: kllm-chat                 (a multi-turn chat; /reset, /quit)
// Raw mode:    kllm-chat --raw "text"    (plain completion, no chat template)

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;
use std::process::exit;

use tokenizers::Tokenizer;

const EOS: u32 = 100257; // Granite <|end_of_text|>

struct Opts {
    prompt: Option<String>,
    device: String,
    tokenizer: Option<String>,
    n_new: usize,
    temp: f32,
    verbose: bool,
    raw: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: kllm-chat [options] [prompt]\n\
         \n\
         options:\n\
         \x20 -n, --tokens N     max tokens to generate (default 64)\n\
         \x20 -t, --temp F       sampling temperature, 0 = greedy (default 0)\n\
         \x20 -d, --device PATH  device node (default /dev/kllm)\n\
         \x20     --tokenizer P  path to tokenizer.json\n\
         \x20     --raw          plain completion, no instruct chat template\n\
         \x20 -v, --verbose      print KV-cache stats to stderr\n\
         \n\
         With a prompt argument, answers once. Without one, starts a chat REPL."
    );
    exit(2);
}

fn parse_args() -> Opts {
    let mut o = Opts {
        prompt: None,
        device: "/dev/kllm".into(),
        tokenizer: None,
        n_new: 64,
        temp: 0.0,
        verbose: false,
        raw: false,
    };
    let mut args = env::args().skip(1);
    let mut positional: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => usage(),
            "-v" | "--verbose" => o.verbose = true,
            "--raw" => o.raw = true,
            "-n" | "--tokens" => o.n_new = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| usage()),
            "-t" | "--temp" => o.temp = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| usage()),
            "-d" | "--device" => o.device = args.next().unwrap_or_else(|| usage()),
            "--tokenizer" => o.tokenizer = Some(args.next().unwrap_or_else(|| usage())),
            s if s.starts_with('-') && s.len() > 1 => usage(),
            _ => positional.push(a),
        }
    }
    if !positional.is_empty() {
        o.prompt = Some(positional.join(" "));
    }
    o
}

// Find the Granite tokenizer.json (repo-relative or next to the binary).
fn find_tokenizer() -> Option<PathBuf> {
    if let Ok(p) = env::var("KLLM_TOKENIZER") {
        return Some(PathBuf::from(p));
    }
    let mut candidates = vec![PathBuf::from("models/granite/tokenizer.json")];
    if let Ok(exe) = env::current_exe() {
        // client/target/release/kllm-chat -> repo root is three up
        if let Some(root) = exe.ancestors().nth(4) {
            candidates.push(root.join("models/granite/tokenizer.json"));
        }
    }
    candidates.into_iter().find(|p| p.exists())
}

// Granite instruct chat template for a single user turn (assistant left open).
fn user_turn(msg: &str) -> String {
    format!(
        "<|start_of_role|>user<|end_of_role|>{msg}<|end_of_text|>\n\
         <|start_of_role|>assistant<|end_of_role|>"
    )
}

// One descriptor transaction with the device. Returns (generated ids, stats).
fn generate(device: &str, ids: &[u32], n_new: usize, temp: f32) -> io::Result<(Vec<u32>, String)> {
    let mut req = format!("{} {}", n_new, temp);
    for id in ids {
        req.push(' ');
        req.push_str(&id.to_string());
    }
    let mut f = OpenOptions::new().read(true).write(true).open(device)?;
    f.write_all(req.as_bytes())?;
    let mut resp = String::new();
    f.read_to_string(&mut resp)?;

    let mut gen = Vec::new();
    let mut stats = String::new();
    for line in resp.lines() {
        if let Some(rest) = line.strip_prefix('#') {
            stats = rest.trim().to_string();
        } else if gen.is_empty() && !line.trim().is_empty() {
            gen = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        }
    }
    Ok((gen, stats))
}

// Build the device input ids for a prompt (templated or raw).
fn run(tok: &Tokenizer, o: &Opts, text: &str) -> (Vec<u32>, String, String) {
    let enc = tok.encode(text, false).expect("tokenize");
    let (mut gen, stats) = generate(&o.device, enc.get_ids(), o.n_new, o.temp).unwrap_or_else(|e| {
        eprintln!("kllm-chat: {} ({})", e, o.device);
        if e.kind() == io::ErrorKind::PermissionDenied {
            eprintln!("hint: `sudo chmod 666 {}` or run as root", o.device);
        }
        exit(1);
    });
    if let Some(i) = gen.iter().position(|&t| t == EOS) {
        gen.truncate(i); // stop at end-of-text
    }
    let reply = tok.decode(&gen, true).expect("detokenize");
    (gen, reply, stats)
}

fn repl(tok: &Tokenizer, o: &Opts) {
    eprintln!(
        "kllm-chat -> {} | granite | n={} temp={} | /reset /quit",
        o.device, o.n_new, o.temp
    );
    let stdin = io::stdin();
    let mut convo = String::new();
    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            eprintln!();
            break;
        }
        let line = line.trim_end();
        match line {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                convo.clear();
                eprintln!("(context cleared)");
                continue;
            }
            _ => {}
        }
        let full = format!("{convo}{}", user_turn(line));
        let (_gen, reply, stats) = run(tok, o, &full);
        println!("{}", reply.trim());
        if o.verbose {
            eprintln!("[{stats}]");
        }
        convo = format!("{full}{reply}<|end_of_text|>\n");
    }
}

fn main() {
    let o = parse_args();
    let tok_path = o
        .tokenizer
        .clone()
        .map(PathBuf::from)
        .or_else(find_tokenizer)
        .unwrap_or_else(|| {
            eprintln!("kllm-chat: tokenizer.json not found; pass --tokenizer PATH (or set KLLM_TOKENIZER)");
            exit(1);
        });
    let tok = Tokenizer::from_file(&tok_path).unwrap_or_else(|e| {
        eprintln!("kllm-chat: failed to load tokenizer {}: {}", tok_path.display(), e);
        exit(1);
    });
    let _ = fs::metadata(&tok_path); // touch (clarity)

    match &o.prompt {
        Some(p) => {
            let text = if o.raw { p.clone() } else { user_turn(p) };
            let (_gen, reply, stats) = run(&tok, &o, &text);
            if o.raw {
                println!("{p}{reply}");
            } else {
                println!("{}", reply.trim());
            }
            if o.verbose {
                eprintln!("[{stats}]");
            }
        }
        None => repl(&tok, &o),
    }
}
