// kllm-chat — a client for the kLLM character device.
//
// The device (/dev/kllm) speaks token IDs; the BPE tokenizer lives here in
// userspace. A request is a single descriptor transaction:
//
//     write:  "<n_new> <temp> <id> <id> ..."
//     read:   "<id> <id> ...\n# prefill_skipped ... | ... | store_total N\n"
//
// One-shot:    kllm-chat "a prompt"
// Interactive: kllm-chat            (reads lines; /reset, /quit; --chat keeps context)

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;
use std::process::exit;

use tokenizers::Tokenizer;

struct Opts {
    prompt: Option<String>,
    device: String,
    tokenizer: Option<String>,
    n_new: usize,
    temp: f32,
    verbose: bool,
    chat: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: kllm-chat [options] [prompt]\n\
         \n\
         options:\n\
         \x20 -n, --tokens N     tokens to generate (default 30)\n\
         \x20 -t, --temp F       sampling temperature, 0 = greedy (default 0)\n\
         \x20 -d, --device PATH  device node (default /dev/kllm)\n\
         \x20     --tokenizer P  path to tokenizer.json (default: GPT-2 in HF cache)\n\
         \x20     --chat         interactive mode: keep conversation context\n\
         \x20 -v, --verbose      print KV-cache stats to stderr\n\
         \n\
         With a prompt argument, completes it once. Without one, starts a REPL."
    );
    exit(2);
}

fn parse_args() -> Opts {
    let mut o = Opts {
        prompt: None,
        device: "/dev/kllm".into(),
        tokenizer: None,
        n_new: 30,
        temp: 0.0,
        verbose: false,
        chat: false,
    };
    let mut args = env::args().skip(1).peekable();
    let mut positional: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => usage(),
            "-v" | "--verbose" => o.verbose = true,
            "--chat" => o.chat = true,
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

// Locate the GPT-2 tokenizer.json in the Hugging Face cache.
fn find_tokenizer() -> Option<PathBuf> {
    let home = env::var("HOME").ok()?;
    let base = PathBuf::from(home).join(".cache/huggingface/hub/models--gpt2/snapshots");
    for entry in fs::read_dir(base).ok()? {
        let p = entry.ok()?.path().join("tokenizer.json");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

// One descriptor transaction with the device. Returns (generated ids, stats line).
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

fn complete(tok: &Tokenizer, o: &Opts, prompt: &str) -> (String, String) {
    let enc = tok.encode(prompt, false).expect("tokenize");
    let (gen, stats) = generate(&o.device, enc.get_ids(), o.n_new, o.temp).unwrap_or_else(|e| {
        eprintln!("kllm-chat: {} ({})", e, o.device);
        if e.kind() == io::ErrorKind::PermissionDenied {
            eprintln!("hint: the device is root-owned; `sudo chmod 666 {}` or run as root", o.device);
        }
        exit(1);
    });
    let text = tok.decode(&gen, false).expect("detokenize");
    (text, stats)
}

fn repl(tok: &Tokenizer, o: &Opts) {
    eprintln!(
        "kllm-chat -> {} | n={} temp={}{} | /reset /quit",
        o.device, o.n_new, o.temp, if o.chat { " | chat" } else { "" }
    );
    let stdin = io::stdin();
    let mut context = String::new();
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
                context.clear();
                eprintln!("(context cleared)");
                continue;
            }
            _ => {}
        }
        let prompt = if o.chat { format!("{context}{line}") } else { line.to_string() };
        let (text, stats) = complete(tok, o, &prompt);
        println!("{}", text.trim_start());
        if o.verbose {
            eprintln!("[{stats}]");
        }
        if o.chat {
            context = format!("{context}{line}{text}");
        }
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
            eprintln!("kllm-chat: GPT-2 tokenizer.json not found in the HF cache; pass --tokenizer PATH");
            exit(1);
        });
    let tok = Tokenizer::from_file(&tok_path).unwrap_or_else(|e| {
        eprintln!("kllm-chat: failed to load tokenizer {}: {}", tok_path.display(), e);
        exit(1);
    });

    match &o.prompt {
        Some(p) => {
            let (text, stats) = complete(&tok, &o, p);
            println!("{p}{text}");
            if o.verbose {
                eprintln!("[{stats}]");
            }
        }
        None => repl(&tok, &o),
    }
}
