//! Minimal native LSP client: JSON-RPC 2.0 over stdio with Content-Length
//! framing, on one background reader thread. Only the API surface the editor
//! actually uses: initialize / didOpen / didChange / completion /
//! publishDiagnostics.
//!
//! The protocol layer is proven against a mock server in tests; a real server
//! (rust-analyzer, pylsp, …) is auto-detected on PATH and optional.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

pub struct Lsp {
    child: Child,
    stdin: std::sync::Mutex<std::process::ChildStdin>,
    tx: std::sync::Mutex<Sender<serde_json::Value>>,
    rx: Receiver<serde_json::Value>,
    next_id: std::sync::atomic::AtomicI64,
    pub server: String,
}

const CANDIDATES: &[(&str, &[&str])] = &[
    ("rust-analyzer", &[]),
    ("pylsp", &[]),
    ("typescript-language-server", &["--stdio"]),
    ("clangd", &["--stdio"]),
];

impl Lsp {
    /// First candidate that spawns wins; `None` = no server (feature idles).
    pub fn detect(root: &Path) -> Option<Self> {
        CANDIDATES
            .iter()
            .find_map(|(cmd, args)| Self::spawn(cmd, args, root))
    }

    pub fn spawn(cmd: &str, args: &[&str], root: &Path) -> Option<Self> {
        let mut child = Command::new(cmd)
            .args(args)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW on Windows
            .spawn()
            .ok()?;
        let stdout = child.stdout.take()?;
        let stdin = child.stdin.take()?;
        let (tx, rx) = channel();
        let tx2 = tx.clone();
        thread::Builder::new()
            .name("vx-lsp-read".into())
            .spawn(move || read_loop(stdout, tx2))
            .ok()?;
        Some(Self {
            child,
            stdin: std::sync::Mutex::new(stdin),
            tx: std::sync::Mutex::new(tx),
            rx,
            next_id: std::sync::atomic::AtomicI64::new(1),
            server: cmd.to_string(),
        })
    }

    fn send(&self, msg: serde_json::Value) {
        let body = serde_json::to_string(&msg).unwrap_or_default();
        let framed = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        if let Ok(mut stdin) = self.stdin.lock() {
            let _ = stdin.write_all(framed.as_bytes());
            let _ = stdin.flush();
        }
    }

    pub fn request(&self, method: &str, params: serde_json::Value) -> i64 {
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.send(serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params
        }));
        id
    }

    pub fn notify(&self, method: &str, params: serde_json::Value) {
        self.send(serde_json::json!({
            "jsonrpc": "2.0", "method": method, "params": params
        }));
    }

    pub fn initialize(&self, root: &Path) {
        self.request(
            "initialize",
            serde_json::json!({
                "processId": std::process::id(),
                "rootUri": uri(root),
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {},
                        "completion": {
                            "completionItem": { "snippetSupport": false }
                        },
                        "synchronization": { "didSave": true }
                    },
                    "workspace": { "workspaceFolders": true }
                }
            }),
        );
        self.notify("initialized", serde_json::json!({}));
    }

    pub fn did_open(&self, path: &Path, text: &str) {
        self.notify(
            "textDocument/didOpen",
            serde_json::json!({
                "textDocument": {
                    "uri": uri(path),
                    "languageId": lang_id(path),
                    "version": 1,
                    "text": text
                }
            }),
        );
    }

    pub fn did_change(&self, path: &Path, text: &str, version: i64) {
        self.notify(
            "textDocument/didChange",
            serde_json::json!({
                "textDocument": { "uri": uri(path), "version": version },
                "contentChanges": [{ "text": text }]
            }),
        );
    }

    pub fn completion(&self, path: &Path, line: usize, col: usize) -> i64 {
        let id = self.request(
            "textDocument/completion",
            serde_json::json!({
                "textDocument": { "uri": uri(path) },
                "position": { "line": line, "character": col }
            }),
        );
        id
    }

    /// Non-blocking drain of everything the reader thread has decoded.
    pub fn drain(&self) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        while let Ok(v) = self.rx.try_recv() {
            out.push(v);
        }
        out
    }
}

impl Drop for Lsp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_loop<R: Read + Send + 'static>(out: R, tx: Sender<serde_json::Value>) {
    let mut reader = BufReader::new(out);
    loop {
        // headers
        let mut len = 0usize;
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
            let line = line.trim();
            if line.is_empty() {
                break;
            }
            if let Some(v) = line
                .strip_prefix("Content-Length:")
                .or_else(|| line.strip_prefix("content-length:"))
            {
                len = v.trim().parse().unwrap_or(0);
            }
        }
        if len == 0 {
            continue;
        }
        let mut buf = vec![0u8; len];
        if reader.read_exact(&mut buf).is_err() {
            return;
        }
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&buf) {
            if tx.send(v).is_err() {
                return;
            }
        }
    }
}

pub fn uri(path: &Path) -> String {
    let p = path.to_string_lossy().replace('\\', "/");
    if p.starts_with('/') {
        format!("file://{p}")
    } else {
        format!("file:///{p}")
    }
}

fn lang_id(path: &Path) -> &'static str {
    match path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
        .as_str()
    {
        "rs" => "rust",
        "py" => "python",
        "ts" => "typescript",
        "tsx" => "typescriptreact",
        "js" | "jsx" => "javascript",
        _ => "plaintext",
    }
}

pub fn workspace_root_of(file: &Path) -> PathBuf {
    let mut dir = file.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    loop {
        for marker in [".git", "Cargo.toml", "pyproject.toml", "package.json"] {
            if dir.join(marker).exists() {
                return dir;
            }
        }
        if !dir.pop() {
            return PathBuf::from(".");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Mock LSP server: answers initialize + sends a diagnostic on didOpen.
    fn mock_server_src() -> String {
        let py = r#"
import sys, json
def send(obj):
    b = json.dumps(obj).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(b) + b)
    sys.stdout.buffer.flush()
def read_msg():
    hdr = b""
    while not hdr.endswith(b"\r\n\r\n"):
        c = sys.stdin.buffer.read(1)
        if not c: return None
        hdr += c
    n = 0
    for line in hdr.decode().splitlines():
        if line.lower().startswith("content-length:"):
            n = int(line.split(":")[1])
    body = sys.stdin.buffer.read(n)
    return json.loads(body)
send({"jsonrpc":"2.0","id":0,"result":{"capabilities":{}}})
while True:
    m = read_msg()
    if m is None: break
    if m.get("method") == "initialize":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"capabilities":{}}})
    elif m.get("method") == "textDocument/didOpen":
        send({"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{
            "uri": m["params"]["textDocument"]["uri"],
            "diagnostics":[{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":4}},
                            "severity":2,"message":"mock warning"}]}})
    elif m.get("method") == "textDocument/completion":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"isIncomplete":False,
              "items":[{"label":"mock_item","kind":6}]}})
"#;
        py.to_string()
    }

    #[test]
    fn protocol_against_mock_server() {
        // stage mock as a temp .py and drive the full client
        let dir = std::env::temp_dir().join("vx_lsp_mock");
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("mock_lsp.py");
        std::fs::write(&script, mock_server_src()).unwrap();

        let root = std::env::temp_dir();
        let mut lsp = Lsp::spawn(
            "python",
            &[script.to_str().unwrap()],
            &root,
        )
        .expect("spawn mock");
        lsp.initialize(&root);
        lsp.did_open(&PathBuf::from("/tmp/x.rs"), "fn main() {}\n");
        lsp.completion(&PathBuf::from("/tmp/x.rs"), 0, 3);

        let mut saw_init = false;
        let mut saw_diag = false;
        let mut saw_completion = false;
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline && !(saw_init && saw_diag && saw_completion) {
            for v in lsp.drain() {
                match v.get("method").and_then(|m| m.as_str()) {
                    Some("textDocument/publishDiagnostics") => saw_diag = true,
                    _ => {
                        if v.get("id").and_then(|i| i.as_i64()) == Some(1) {
                            saw_init = true;
                        }
                        if v
                            .pointer("/result/items/0/label")
                            .and_then(|l| l.as_str())
                            == Some("mock_item")
                        {
                            saw_completion = true;
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        assert!(saw_init, "initialize response");
        assert!(saw_diag, "publishDiagnostics notification");
        assert!(saw_completion, "completion items");
    }
}
