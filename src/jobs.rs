//! Lock-free-ish worker pool: N background threads pull jobs from a
//! crossbeam channel (MPMC). Results flow back via std mpsc; the UI thread
//! drains them per frame and never performs file I/O, regex, walking or
//! parsing itself.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use crossbeam_channel::{unbounded, Receiver as CbReceiver};

pub enum Job {
    /// Build workspace file list for the fuzzy finder.
    Walk { seq: u64, root: PathBuf, cap: usize },
    /// Regex search across files; returns matching line refs.
    Search { seq: u64, pattern: String, root: PathBuf },
    /// Load a file end-to-end: mmap → UTF-8 (validated on this thread) → Rope.
    Open { seq: u64, path: PathBuf },
    /// Incremental tree-sitter parse: worker reuses the cached tree for this
    /// path when `base_revision` still matches, applying `edit` via InputEdit.
    Parse {
        path: PathBuf,
        revision: u64,
        base_revision: u64,
        edit: Option<(usize, usize, usize)>,
        text: String,
    },
}

pub enum Outcome {
    Files { seq: u64, paths: Vec<String> },
    Hits { seq: u64, hits: Vec<Hit> },
    Opened { seq: u64, path: PathBuf, text: String, len: u64 },
    Parsed {
        revision: u64,
        tree: Box<tree_sitter::Tree>,
        incremental: bool,
    },
    Failed { seq: u64, what: String, error: String },
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub path: String,
    pub line: usize,
    pub text: String,
}

type TreeCache = std::sync::Arc<
    std::sync::Mutex<std::collections::HashMap<String, (tree_sitter::Tree, u64)>>,
>;

pub struct Workers {
    jobs_tx: crossbeam_channel::Sender<Job>,
    results: Receiver<Outcome>,
    pub in_flight: std::sync::atomic::AtomicUsize,
    pub trees: TreeCache,
}

impl Workers {
    pub fn spawn(count: usize) -> Self {
        let (jobs_tx, jobs_rx) = unbounded::<Job>();
        let (tx, results) = channel::<Outcome>();
        let jobs_rx: CbReceiver<Job> = jobs_rx;
        let trees: TreeCache = std::sync::Arc::new(std::sync::Mutex::new(Default::default()));
        for i in 0..count.max(2) {
            let rx = jobs_rx.clone();
            let tx = tx.clone();
            let trees = trees.clone();
            thread::Builder::new()
                .name(format!("vx-worker-{i}"))
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        run(job, &tx, &trees);
                    }
                })
                .expect("spawn worker");
        }
        Self {
            jobs_tx,
            results,
            in_flight: std::sync::atomic::AtomicUsize::new(0),
            trees,
        }
    }

    pub fn submit(&self, job: Job) {
        self.in_flight
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _ = self.jobs_tx.send(job);
    }

    /// Non-blocking: pull every completed outcome.
    pub fn drain(&self) -> Vec<Outcome> {
        let mut out = Vec::new();
        while let Ok(o) = self.results.try_recv() {
            self.in_flight
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            out.push(o);
        }
        out
    }

    pub fn busy(&self) -> usize {
        self.in_flight.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn run(job: Job, tx: &Sender<Outcome>, trees: &TreeCache) {
    match job {
        Job::Walk { seq, root, cap } => {
            let paths = crate::fuzzy::walk(&root, cap);
            let _ = tx.send(Outcome::Files { seq, paths });
        }
        Job::Search { seq, pattern, root } => match regex::Regex::new(&pattern) {
            Ok(re) => {
                let hits = search(&re, &root, 400);
                let _ = tx.send(Outcome::Hits { seq, hits });
            }
            Err(e) => {
                let _ = tx.send(Outcome::Failed {
                    seq,
                    what: "regex".into(),
                    error: e.to_string(),
                });
            }
        },
        Job::Parse {
            path,
            revision,
            base_revision,
            edit,
            text,
        } => {
            let key = path.to_string_lossy().to_string();
            let mut cache = trees.lock().unwrap();
            let mut incremental = false;
            let mut syn = match crate::syntax::Syntax::new(&key) {
                Some(s) => s,
                None => return, // not a parsed language — idle silently
            };
            let rope = ropey::Rope::from_str(&text);
            let prev = cache.get(&key).cloned();
            match (prev, edit) {
                (Some((tree, cached_rev)), Some(e)) if cached_rev == base_revision => {
                    // true incremental path: edit old tree, reuse it
                    syn.install(tree);
                    syn.apply_edit(&rope, e);
                    incremental = true;
                    syn.parse_with(&rope);
                }
                _ => {
                    syn.parse_with(&rope);
                }
            }
            if let Some(tree) = syn.take_tree() {
                cache.insert(key, (tree.clone(), revision));
                let _ = tx.send(Outcome::Parsed {
                    revision,
                    tree: Box::new(tree),
                    incremental,
                });
            }
        }
        Job::Open { seq, path } => match open_file(&path) {
            Ok((text, len)) => {
                let _ = tx.send(Outcome::Opened {
                    seq,
                    path,
                    text,
                    len,
                });
            }
            Err(e) => {
                let _ = tx.send(Outcome::Failed {
                    seq,
                    what: path.display().to_string(),
                    error: e,
                });
            }
        },
    }
}

/// mmap open: no read() copy, OS pages in on demand; UTF-8 validated here,
/// off the UI thread.
fn open_file(path: &Path) -> Result<(String, u64), String> {
    use std::fs::File;
    let file = File::open(path).map_err(|e| e.to_string())?;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    if len > (8usize << 30) as u64 {
        return Err("file > 8 GiB refused".into());
    }
    if len == 0 {
        return Ok((String::new(), 0));
    }
    unsafe {
        let mmap = memmap2::Mmap::map(&file).map_err(|e| e.to_string())?;
        // Validated on this worker: invalid UTF-8 → lossy, never panics the UI.
        let text = match std::str::from_utf8(&mmap) {
            Ok(s) => s.to_string(),
            Err(_) => String::from_utf8_lossy(&mmap).into_owned(),
        };
        Ok((text, len))
    }
}

const SKIP: [&str; 5] = [".git", "node_modules", "target", "__pycache__", ".venv"];

fn search(re: &regex::Regex, root: &Path, cap: usize) -> Vec<Hit> {
    let mut hits = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if hits.len() >= cap {
            break;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            if hits.len() >= cap {
                break;
            }
            let p = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                if !SKIP.contains(&name.as_str()) && !name.starts_with('.') {
                    stack.push(p);
                }
                continue;
            }
            // text-ish extension filter (cheap)
            let ext = p
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            const TEXTY: [&str; 14] = [
                "rs", "py", "js", "ts", "tsx", "jsx", "toml", "json", "md", "txt", "html",
                "css", "yml", "yaml",
            ];
            if !TEXTY.contains(&ext.as_str()) && name != "Cargo.toml" {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&p) else {
                continue;
            };
            for (i, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    hits.push(Hit {
                        path: p.to_string_lossy().to_string(),
                        line: i,
                        text: line.chars().take(160).collect(),
                    });
                    if hits.len() >= cap {
                        break;
                    }
                }
            }
        }
    }
    hits
}
