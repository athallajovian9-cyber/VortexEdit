//! VortexEdit — GPU text editor.
//! wgpu + rustybuzz + ropey + tree-sitter + memmap2, zero web runtime.
//!
//! Keys: type to edit · Alt+Click add cursor · Ctrl+D select-next-occurrence
//! Ctrl+P fuzzy file finder · Ctrl+F regex workspace search (worker)
//! Ctrl+Space LSP completion · wheel/keys scroll · Esc collapse cursors/overlays
//! Ctrl+Q quit.

mod buffer;
mod fuzzy;
mod gpu;
mod jobs;
mod lsp;
mod syntax;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes},
};

use buffer::Editor;
use gpu::{OverlayText, Renderer, SEARCH_C};
use jobs::{Job, Outcome, Workers};
use lsp::Lsp;
use syntax::Syntax;

static SEQ: AtomicU64 = AtomicU64::new(1);
fn next_seq() -> u64 {
    SEQ.fetch_add(1, Ordering::Relaxed)
}

const LINE_LIMIT: usize = 50_000; // files larger than this: no syntax parse

enum Mode {
    Normal,
    Finder,
    Search,
    Complete,
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    editor: Editor,
    syntax: Option<Syntax>,
    workers: Workers,
    lsp: Option<Lsp>,
    file: Option<PathBuf>,
    root: PathBuf,
    mode: Mode,
    input: String,
    finder_files: Vec<String>,
    finder_hits: Vec<(i64, usize)>,
    finder_sel: usize,
    search_hits: Vec<jobs::Hit>,
    search_sel: usize,
    completion: Vec<String>,
    completion_sel: usize,
    diag: Vec<(usize, usize)>, // char ranges
    scroll: f64,
    status: String,
    syntax_submitted_rev: u64,
    lsp_version: i64,
    t_start: Instant,
    first_frame: Option<Instant>,
    blink: Instant,
    last_frame: Instant,
    frame_ema_ms: f64,
    pending_query: Option<String>,
    open_requested: Option<(PathBuf, Option<usize>)>,
    modifiers: ModifiersState,
    last_cursor_pos: Option<(f64, f64)>,
}

impl App {
    fn new(root: PathBuf) -> Self {
        let workers = Workers::spawn(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        );
        workers.submit(Job::Walk {
            seq: next_seq(),
            root: root.clone(),
            cap: 50_000,
        });
        Self {
            window: None,
            renderer: None,
            editor: Editor::new(""),
            syntax: None,
            workers,
            lsp: None,
            file: None,
            root,
            mode: Mode::Normal,
            input: String::new(),
            finder_files: Vec::new(),
            finder_hits: Vec::new(),
            finder_sel: 0,
            search_hits: Vec::new(),
            search_sel: 0,
            completion: Vec::new(),
            completion_sel: 0,
            diag: Vec::new(),
            scroll: 0.0,
            status: "ready · Ctrl+P finder · Ctrl+F search · Ctrl+Space completion".into(),
            syntax_submitted_rev: 0,
            lsp_version: 0,
            t_start: Instant::now(),
            first_frame: None,
            blink: Instant::now(),
            last_frame: Instant::now(),
            frame_ema_ms: 0.0,
            pending_query: None,
            open_requested: None,
            modifiers: ModifiersState::default(),
            last_cursor_pos: None,
        }
    }

    fn metrics(&self) -> (f32, f32) {
        let r = self.renderer.as_ref().unwrap();
        let px = (r.metrics.base_px * scale_of(self.window.as_ref()) as f32).round();
        (px * r.metrics.line_h, px)
    }

    fn request_open(&mut self, path: PathBuf, line: Option<usize>) {
        self.open_requested = Some((path.clone(), line));
        let seq = next_seq();
        self.workers.submit(Job::Open {
            seq,
            path: path.clone(),
        });
        self.status = format!("opening {} …", path.display());
    }

    fn maybe_submit_parse(&mut self) {
        let Some(path) = self.file.clone() else { return };
        if self.editor.line_count() > LINE_LIMIT {
            return;
        }
        if self.syntax.is_none() {
            return;
        }
        if self.syntax_submitted_rev == self.editor.revision {
            return;
        }
        let rev = self.editor.revision;
        let base = self.syntax_submitted_rev;
        let edit = self.editor.last_edit;
        let text = self.editor.rope.slice(..).to_string();
        self.syntax_submitted_rev = rev;
        self.workers.submit(Job::Parse {
            path,
            revision: rev,
            base_revision: base,
            edit,
            text,
        });
    }

    fn handle_outcomes(&mut self) {
        for o in self.workers.drain() {
            match o {
                Outcome::Files { paths, .. } => {
                    self.finder_files = paths;
                    self.status = format!("indexed {} files", self.finder_files.len());
                }
                Outcome::Opened { path, text, len, .. } => {
                    self.editor = Editor::new(&text);
                    self.file = Some(path.clone());
                    self.syntax = Syntax::new(&path.to_string_lossy());
                    // force the first parse submission (editor revision starts at 0)
                    self.syntax_submitted_rev = u64::MAX;
                    self.scroll = 0.0;
                    self.diag.clear();
                    self.status = format!(
                        "{} · {:.1} KiB · mmap→rope off-thread",
                        path.display(),
                        len as f64 / 1024.0
                    );
                    // LSP attach
                    if let Some(l) = &self.lsp {
                        l.did_open(&path, &text);
                    }
                    self.maybe_submit_parse();
                    if let Some((_, line)) = self.open_requested.take() {
                        if let Some(l) = line {
                            let idx = self
                                .editor
                                .line_range(l)
                                .map(|r| r.0)
                                .unwrap_or(0);
                            self.editor.set_primary(idx);
                        }
                    }
                }
                Outcome::Hits { hits, .. } => {
                    self.search_hits = hits;
                    self.search_sel = 0;
                    self.status = format!("{} matches", self.search_hits.len());
                }
                Outcome::Parsed {
                    revision,
                    tree,
                    incremental,
                } => {
                    if revision == self.editor.revision {
                        if let Some(s) = &mut self.syntax {
                            s.install(*tree);
                        }
                        if let Some(r) = &mut self.renderer {
                            r.invalidate();
                        }
                        self.status = format!(
                            "parsed ({})",
                            if incremental { "incremental" } else { "full" }
                        );
                    }
                }
                Outcome::Failed { what, error, .. } => {
                    self.status = format!("failed: {what}: {error}");
                }
            }
        }
    }

    fn handle_lsp(&mut self) {
        let msgs: Vec<serde_json::Value> =
            self.lsp.as_ref().map(|l| l.drain()).unwrap_or_default();
        for m in msgs {
            match m.get("method").and_then(|x| x.as_str()) {
                Some("textDocument/publishDiagnostics") => {
                    let mut ranges = Vec::new();
                    if let Some(ds) = m.pointer("/params/diagnostics").and_then(|d| d.as_array()) {
                        for d in ds {
                            let (sl, sc) = (
                                d.pointer("/range/start/line").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                                d.pointer("/range/start/character").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                            );
                            let (el, ec) = (
                                d.pointer("/range/end/line").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                                d.pointer("/range/end/character").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                            );
                            // UTF-16 columns assumed == chars (ASCII files exact)
                            if let (Some(a), Some(b)) = (self.editor.line_range(sl), self.editor.line_range(el)) {
                                let s = (a.0 + sc).min(a.1);
                                let e = (b.0 + ec).min(b.1);
                                if e > s {
                                    ranges.push((s, e));
                                }
                            }
                        }
                    }
                    self.diag = ranges;
                }
                None => {
                    // completion response?
                    let items = m.pointer("/result/items").and_then(|i| i.as_array());
                    if let Some(items) = items {
                        let labels: Vec<String> = items
                            .iter()
                            .take(40)
                            .filter_map(|i| i.get("label").and_then(|l| l.as_str()).map(String::from))
                            .collect();
                        if !labels.is_empty() {
                            self.completion = labels;
                            self.completion_sel = 0;
                            self.mode = Mode::Complete;
                            self.status = "completion — Tab accept · Esc cancel".into();
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn requery_finder(&mut self) {
        let q = self.input.clone();
        self.finder_hits = fuzzy::rank(&q, &self.finder_files);
        self.finder_hits.truncate(60);
        self.finder_sel = 0;
    }

    fn cursor_screen_y(&self) -> f64 {
        let (line_h, _) = self.metrics();
        let line = self
            .editor
            .rope
            .try_char_to_line(self.editor.primary())
            .unwrap_or(0) as f64;
        line * line_h as f64 - self.scroll
    }

    fn ensure_cursor_visible(&mut self) {
        let (line_h, _) = self.metrics();
        let (_, h) = self.renderer.as_ref().map(|r| r.size()).unwrap_or((0, 0));
        let y = self.cursor_screen_y();
        if y < line_h as f64 * 2.0 {
            self.scroll = (y - line_h as f64 * 2.0).max(0.0);
        } else if y > h as f64 - line_h as f64 * 4.0 {
            self.scroll = y - h as f64 + line_h as f64 * 4.0;
        }
    }

    fn draw_frame(&mut self) {
        let scale = scale_of(self.window.as_ref());
        let (line_h, _) = self.metrics();
        let cursor_y = self.cursor_screen_y();
        let Some(renderer) = self.renderer.as_mut() else { return };
        let (w, h) = renderer.size();
        let (w, h) = (w as f32, h as f32);

        let caret_on = self.blink.elapsed().as_millis() % 1000 < 620;
        let mut overlays: Vec<OverlayText> = Vec::new();
        let mut panels: Vec<(f32, f32, f32, f32, [f32; 4])> = Vec::new();
        let mut underlays: Vec<(f32, f32, f32, f32, [f32; 4])> = Vec::new();

        // search-hit underlines as doc-space underlays (line for current file only:
        // hits from other files aren't in this buffer — filter)
        // status bar
        let lh = line_h;
        panels.push((0.0, h - lh, w, lh, gpu::PANEL));
        let (l, c) = self.editor.line_col(self.editor.primary());
        let syn_name = self.syntax.as_ref().map(|s| s.lang).unwrap_or("-");
        let lsp_name = self.lsp.as_ref().map(|l| l.server.as_str()).unwrap_or("off");
        let status = format!(
            " {} │ {}:{} │ {}c │ {} │ LSP:{} │ jobs:{} │ {:.1}ms │ {}",
            self.file
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| "[no file — Ctrl+P]".into()),
            l + 1,
            c + 1,
            self.editor.cursors.len(),
            syn_name,
            lsp_name,
            self.workers.busy(),
            self.frame_ema_ms,
            self.status,
        );
        overlays.push(OverlayText {
            text: status.chars().take(200).collect(),
            x: 8.0,
            y: h - lh + (lh - 15.0 * scale as f32 * 1.4) / 2.0,
            color_override: Some(gpu::DIM),
        });

        match self.mode {
            Mode::Finder => {
                let pw = w * 0.6;
                let px0 = (w - pw) / 2.0;
                panels.push((px0, 40.0, pw, lh * 13.0, gpu::PANEL));
                overlays.push(OverlayText {
                    text: format!("> {}_", self.input),
                    x: px0 + 10.0,
                    y: 50.0,
                    color_override: Some(gpu::GREEN),
                });
                for (i, (_, idx)) in self.finder_hits.iter().enumerate().take(11) {
                    let path = &self.finder_files[*idx];
                    if i == self.finder_sel {
                        panels.push((px0 + 4.0, 50.0 + lh * (i as f32 + 1.6), pw - 8.0, lh, gpu::SEARCH_C));
                    }
                    overlays.push(OverlayText {
                        text: {
                            let rs = self.root.to_string_lossy();
                            path.replace(rs.as_ref(), "")
                        },
                        x: px0 + 12.0,
                        y: 52.0 + lh * (i as f32 + 1.7),
                        color_override: if i == self.finder_sel { Some(gpu::GREEN) } else { None },
                    });
                }
            }
            Mode::Search => {
                panels.push((0.0, h - lh * 2.0, w, lh, gpu::PANEL));
                overlays.push(OverlayText {
                    text: format!("regex> {}_   ({} hits, Enter open, Esc cancel)", self.input, self.search_hits.len()),
                    x: 8.0,
                    y: h - lh * 2.0 + 3.0,
                    color_override: Some(gpu::GREEN),
                });
                for (i, hit) in self.search_hits.iter().enumerate().take(6) {
                    if i == self.search_sel {
                        panels.push((0.0, h - lh * (8.0 - i as f32), w, lh, gpu::SEARCH_C));
                    }
                    overlays.push(OverlayText {
                        text: format!("{}:{}  {}", short(&hit.path, &self.root), hit.line + 1, hit.text.trim()),
                        x: 8.0,
                        y: h - lh * (7.9 - i as f32),
                        color_override: if i == self.search_sel { Some(gpu::GREEN) } else { Some(gpu::DIM) },
                    });
                }
            }
            Mode::Complete => {
                let lh2 = lh;
                let y0 = cursor_y.max(0.0) as f32 + lh2;
                panels.push((120.0, y0, 340.0, lh2 * (self.completion.len().min(12) as f32 + 1.2), gpu::PANEL));
                for (i, item) in self.completion.iter().take(12).enumerate() {
                    if i == self.completion_sel {
                        panels.push((124.0, y0 + 4.0 + lh2 * i as f32, 332.0, lh2, gpu::SEARCH_C));
                    }
                    overlays.push(OverlayText {
                        text: item.clone(),
                        x: 130.0,
                        y: y0 + 5.0 + lh2 * i as f32,
                        color_override: if i == self.completion_sel { Some(gpu::GREEN) } else { None },
                    });
                }
            }
            Mode::Normal => {}
        }

        let hits: Vec<(usize, usize)> = Vec::new(); // per-line search overlay: kept minimal
        let verts = renderer.build(
            &self.editor,
            self.syntax.as_ref(),
            self.scroll,
            scale,
            caret_on,
            &hits,
            &self.diag,
            &overlays,
            &panels,
            &underlays,
        );
        renderer.render(&verts, self.scroll as f32);
        let _ = SEARCH_C;
    }
}

fn short(path: &str, root: &PathBuf) -> String {
    let rs = root.to_string_lossy();
    path.replace(rs.as_ref(), "").trim_start_matches(['/', '\\']).to_string()
}

fn scale_of(window: Option<&Arc<Window>>) -> f64 {
    window.map(|w| w.scale_factor()).unwrap_or(1.0)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            el.create_window(
                WindowAttributes::default()
                    .with_title("VortexEdit")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 820.0)),
            )
            .expect("window"),
        );
        let renderer = Renderer::new(window.clone());
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.status = format!("first frame in {:.1} ms", self.t_start.elapsed().as_secs_f64() * 1000.0);
        // optional LSP auto-detect (silent when no server exists)
        self.lsp = Lsp::detect(&self.root);
        if let Some(l) = &self.lsp {
            l.initialize(&self.root);
        }
        if let (Some(l), Some(f)) = (&self.lsp, &self.file) {
            let text = self.editor.rope.slice(..).to_string();
            l.did_open(f, &text);
        }
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: winit::window::WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(sz) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(sz.width, sz.height);
                }
            }
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = m.state();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if self.modifiers.alt_key() {
                    // add a cursor at the clicked position
                    if let (Some(r), Some(pos)) = (self.renderer.as_ref(), self.last_cursor_pos) {
                        let scale = scale_of(self.window.as_ref());
                        let (line_h, _) = self.metrics();
                        let line = ((pos.1 + self.scroll) / line_h as f64) as usize;
                        let adv = r.metrics.advance * (r.metrics.base_px * scale as f32);
                        let col = (pos.0 as f32 / adv).round() as usize;
                        if let Some((ls, le)) = self.editor.line_range(line) {
                            let idx = (ls + col).min(le);
                            self.editor.add_cursor(idx);
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.last_cursor_pos = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64 * 3.0,
                    MouseScrollDelta::PixelDelta(p) => p.y,
                };
                let (line_h, _) = self.metrics();
                let max = (self.editor.line_count() as f64) * line_h as f64;
                self.scroll = (self.scroll - d).clamp(0.0, (max - 100.0).max(0.0));
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: ElementState::Pressed,
                    logical_key,
                    ..
                },
                ..
            } => {
                self.on_key(el, code, logical_key);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f64() * 1000.0;
                self.last_frame = now;
                self.frame_ema_ms = if self.frame_ema_ms == 0.0 {
                    dt
                } else {
                    self.frame_ema_ms * 0.9 + dt * 0.1
                };
                if self.first_frame.is_none() {
                    self.first_frame = Some(now);
                }
                self.handle_outcomes();
                self.handle_lsp();
                self.maybe_submit_parse();
                if let Some(q) = self.pending_query.take() {
                    self.workers.submit(Job::Search {
                        seq: next_seq(),
                        pattern: q,
                        root: self.root.clone(),
                    });
                }
                self.draw_frame();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl App {
    fn on_key(&mut self, el: &ActiveEventLoop, code: KeyCode, logical: Key) {
        let ctrl = self.modifiers.control_key();
        let alt = self.modifiers.alt_key();

        // ---- mode-typing first
        if !ctrl && !alt {
            if let (Mode::Finder | Mode::Search | Mode::Complete, Key::Character(s)) =
                (&self.mode, &logical)
            {
                match self.mode {
                    Mode::Complete => {} // falls through to completion nav below
                    _ => {
                        self.input.push_str(&s.to_string());
                        if matches!(self.mode, Mode::Finder) {
                            self.requery_finder();
                        }
                        return;
                    }
                }
            }
        }

        match self.mode {
            Mode::Finder => match code {
                KeyCode::Escape => {
                    self.mode = Mode::Normal;
                    self.input.clear();
                }
                KeyCode::Enter => {
                    if let Some((_, idx)) = self.finder_hits.get(self.finder_sel) {
                        let path = PathBuf::from(&self.finder_files[*idx]);
                        self.mode = Mode::Normal;
                        self.request_open(path, None);
                    }
                }
                KeyCode::ArrowDown | KeyCode::KeyN if ctrl => {
                    self.finder_sel = (self.finder_sel + 1).min(self.finder_hits.len().saturating_sub(1));
                }
                KeyCode::ArrowUp | KeyCode::KeyP if ctrl => {
                    self.finder_sel = self.finder_sel.saturating_sub(1);
                }
                KeyCode::Backspace => {
                    self.input.pop();
                    self.requery_finder();
                }
                _ => {}
            },
            Mode::Search => match code {
                KeyCode::Escape => {
                    self.mode = Mode::Normal;
                    self.input.clear();
                }
                KeyCode::Enter => {
                    if let Some(hit) = self.search_hits.get(self.search_sel) {
                        let p = PathBuf::from(&hit.path);
                        let l = hit.line;
                        self.mode = Mode::Normal;
                        self.request_open(p, Some(l));
                    }
                }
                KeyCode::ArrowDown => {
                    self.search_sel = (self.search_sel + 1).min(self.search_hits.len().saturating_sub(1));
                }
                KeyCode::ArrowUp => {
                    self.search_sel = self.search_sel.saturating_sub(1);
                }
                KeyCode::Backspace => {
                    self.input.pop();
                }
                _ => {}
            },
            Mode::Complete => match code {
                KeyCode::Escape => {
                    self.mode = Mode::Normal;
                    self.completion.clear();
                }
                KeyCode::ArrowDown => {
                    self.completion_sel = (self.completion_sel + 1).min(self.completion.len().saturating_sub(1));
                }
                KeyCode::ArrowUp => {
                    self.completion_sel = self.completion_sel.saturating_sub(1);
                }
                KeyCode::Enter | KeyCode::Tab => {
                    if let Some(item) = self.completion.get(self.completion_sel).cloned() {
                        // insert selected label at primary cursor
                        let w = self.editor.word_at(self.editor.primary());
                        if let Some((a, b)) = w {
                            self.editor.set_primary(a);
                            self.editor.cursors = vec![a];
                            for _ in a..b {
                                self.editor.delete_forward();
                            }
                        }
                        self.editor.insert(&item);
                        self.ensure_cursor_visible();
                    }
                    self.mode = Mode::Normal;
                    self.completion.clear();
                }
                _ => {}
            },
            Mode::Normal => {}
        }
        if !matches!(self.mode, Mode::Normal) {
            return;
        }

        // ---- global chords
        if ctrl {
            match code {
                KeyCode::KeyQ => {
                    el.exit();
                    return;
                }
                KeyCode::KeyP => {
                    self.mode = Mode::Finder;
                    self.input.clear();
                    self.requery_finder();
                    return;
                }
                KeyCode::KeyF => {
                    self.mode = Mode::Search;
                    self.input.clear();
                    self.status = "regex search (worker) — type, Enter".into();
                    return;
                }
                KeyCode::Space => {
                    if let (Some(l), Some(f)) = (&self.lsp, &self.file) {
                        let (line, col) = self.editor.line_col(self.editor.primary());
                        l.completion(f, line, col);
                        self.status = "completion requested…".into();
                    } else {
                        self.status = "LSP off (no server on PATH)".into();
                    }
                    return;
                }
                KeyCode::KeyD => {
                    // add next occurrence of word under primary cursor
                    let p = self.editor.primary();
                    if let Some((a, b)) = self.editor.word_at(p) {
                        let word = self.editor.text_slice(a, b);
                        if let Some(next) = self.editor.find_next(&word, p + 1) {
                            self.editor.add_cursor(next);
                            self.ensure_cursor_visible();
                        }
                    }
                    return;
                }
                _ => {}
            }
        }

        // ---- editing & movement
        match &logical {
            Key::Character(s) if !ctrl && !alt => {
                self.editor.insert(s);
                self.after_edit();
            }
            Key::Named(NamedKey::Enter) => {
                self.editor.insert("\n");
                self.after_edit();
            }
            Key::Named(NamedKey::Tab) => {
                self.editor.insert("    ");
                self.after_edit();
            }
            Key::Named(NamedKey::Backspace) => {
                self.editor.backspace();
                self.after_edit();
            }
            Key::Named(NamedKey::Delete) => {
                self.editor.delete_forward();
                self.after_edit();
            }
            Key::Named(NamedKey::ArrowLeft) => {
                let n = self.editor.move_all(-1);
                self.editor.cursors = n;
                self.editor.normalize();
                self.ensure_cursor_visible();
            }
            Key::Named(NamedKey::ArrowRight) => {
                let n = self.editor.move_all(1);
                self.editor.cursors = n;
                self.editor.normalize();
                self.ensure_cursor_visible();
            }
            Key::Named(NamedKey::ArrowUp) | Key::Named(NamedKey::ArrowDown) => {
                let up = matches!(logical, Key::Named(NamedKey::ArrowUp));
                let len = self.editor.rope.len_chars();
                let mut next = Vec::with_capacity(self.editor.cursors.len());
                for &c in &self.editor.cursors {
                    let line = self.editor.rope.try_char_to_line(c).unwrap_or(0);
                    let (ls, le) = match self.editor.line_range(line) {
                        Some(r) => r,
                        None => {
                            next.push(c);
                            continue;
                        }
                    };
                    let col = c - ls;
                    let target_line = if up {
                        line.saturating_sub(1)
                    } else {
                        (line + 1).min(self.editor.line_count().saturating_sub(1))
                    };
                    let idx = if target_line == line {
                        if up { 0 } else { len }
                    } else {
                        let (ts, te) = self.editor.line_range(target_line).unwrap();
                        (ts + col).min(te)
                    };
                    next.push(idx);
                    let _ = le;
                }
                self.editor.cursors = next;
                self.editor.normalize();
                self.ensure_cursor_visible();
            }
            Key::Named(NamedKey::Home) => {
                let p = self.editor.primary();
                let line = self.editor.rope.try_char_to_line(p).unwrap_or(0);
                if let Some((ls, _)) = self.editor.line_range(line) {
                    self.editor.set_primary(ls);
                }
            }
            Key::Named(NamedKey::End) => {
                let p = self.editor.primary();
                let line = self.editor.rope.try_char_to_line(p).unwrap_or(0);
                if let Some((_, le)) = self.editor.line_range(line) {
                    self.editor.set_primary(le);
                }
            }
            Key::Named(NamedKey::Escape) => {
                let p = self.editor.primary();
                self.editor.cursors = vec![p];
                self.status = "esc".into();
            }
            Key::Named(NamedKey::PageUp) | Key::Named(NamedKey::PageDown) => {
                let (_, h) = self.renderer.as_ref().map(|r| r.size()).unwrap_or((0, 800));
                let (line_h, _) = self.metrics();
                let step = h as f64 / line_h as f64;
                if matches!(logical, Key::Named(NamedKey::PageUp)) {
                    self.scroll = (self.scroll - step * line_h as f64).max(0.0);
                } else {
                    self.scroll += step * line_h as f64;
                }
            }
            _ => {}
        }
    }

    fn after_edit(&mut self) {
        self.ensure_cursor_visible();
        if let (Some(l), Some(f)) = (&self.lsp, self.file.clone()) {
            if self.editor.rope.len_bytes() < 1_000_000 {
                let text = self.editor.rope.slice(..).to_string();
                self.lsp_version += 1;
                l.did_change(&f, &text, self.lsp_version);
            }
        }
    }
}

// ------------------------------------------------------------- bench ------
fn bench(args: &[String]) {
    let mut file = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--bench" {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                file = Some(args[i + 1].clone());
            }
        }
        i += 1;
    }
    println!("=== VortexEdit bench ===");
    // 10k cursors
    let text: String = (0..2000).map(|_| "fn value_9() { let x = 0; }\n").collect();
    let mut e = Editor::new(&text);
    e.cursors = (0..10_000).map(|i| (i * e.rope.len_chars()) / 10_000).collect();
    e.normalize();
    let t = Instant::now();
    e.insert("»");
    println!("10k-cursor insert: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    // fuzzy 100k
    let paths: Vec<String> = (0..100_000).map(|i| format!("src/module_{i}/lib_{i}.rs")).collect();
    let t = Instant::now();
    let r = fuzzy::rank("mod42", &paths);
    println!("fuzzy rank 100k: {:.2} ms ({} hits)", t.elapsed().as_secs_f64() * 1000.0, r.len());
    // mmap file open (worker path)
    if let Some(f) = file {
        let w = Workers::spawn(1);
        let t = Instant::now();
        w.submit(Job::Open {
            seq: 1,
            path: PathBuf::from(&f),
        });
        loop {
            for o in w.drain() {
                if let Outcome::Opened { len, text, .. } = o {
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    println!("mmap open {:.1} MiB → rope-ready: {:.1} ms", len as f64 / 1048576.0, ms);
                    let t2 = Instant::now();
                    let ed = Editor::new(&text);
                    println!("rope build: {:.1} ms ({} lines)", t2.elapsed().as_secs_f64() * 1000.0, ed.line_count());
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--bench") {
        bench(&args);
        return;
    }
    let file = args.iter().find(|a| !a.starts_with("--")).cloned();
    let mut root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut app = if let Some(f) = &file {
        let p = PathBuf::from(f);
        if let Some(dir) = p.parent() {
            if dir.exists() {
                root = dir.to_path_buf();
            }
        }
        let mut app = App::new(root);
        app.request_open(p, None);
        app
    } else {
        App::new(root)
    };
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.run_app(&mut app).expect("run");
    let _ = &mut app;
}
