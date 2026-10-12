//! Tree-sitter integration: incremental CST on background-safe calls,
//! pragmatic token classification for colors (no highlight-query crate —
//! node-kind matching is sufficient and one less dependency).

use ropey::Rope;
use tree_sitter::{InputEdit, Parser, Tree};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Kind {
    Text,
    Comment,
    String,
    Number,
    Keyword,
    Function,
    Type,
    Punct,
}

pub struct Syntax {
    parser: Parser,
    tree: Option<Tree>,
    pub lang: &'static str,
}

fn language_for(path: &str) -> Option<(&'static str, tree_sitter::Language)> {
    let p = path.to_ascii_lowercase();
    if p.ends_with(".rs") {
        Some(("rust", tree_sitter_rust::LANGUAGE.into()))
    } else if p.ends_with(".py") {
        Some(("python", tree_sitter_python::LANGUAGE.into()))
    } else if p.ends_with(".js")
        || p.ends_with(".jsx")
        || p.ends_with(".ts")
        || p.ends_with(".tsx")
    {
        Some(("javascript", tree_sitter_javascript::LANGUAGE.into()))
    } else {
        None
    }
}

impl Syntax {
    pub fn new(path: &str) -> Option<Self> {
        let (lang, language) = language_for(path)?;
        let mut parser = Parser::new();
        parser.set_language(&language).ok()?;
        Some(Self {
            parser,
            tree: None,
            lang,
        })
    }

    /// Install a tree produced elsewhere (worker thread → UI swap).
    pub fn install(&mut self, tree: Tree) {
        self.tree = Some(tree);
    }

    pub fn take_tree(&mut self) -> Option<Tree> {
        self.tree.take()
    }

    /// Apply an InputEdit to the cached tree without parsing (cheap,
    /// called on the worker between edit and reparse).
    pub fn apply_edit(&mut self, rope: &Rope, (start_b, old_end_b, new_end_b): (usize, usize, usize)) {
        let Some(tree) = self.tree.take() else { return };
        let point = |byte: usize| {
            let line = rope.try_byte_to_line(byte).unwrap_or(0);
            let ls = rope.try_line_to_byte(line).unwrap_or(0);
            tree_sitter::Point { row: line, column: byte.saturating_sub(ls) }
        };
        let mut t = tree;
        t.edit(&InputEdit {
            start_byte: start_b,
            old_end_byte: old_end_b,
            new_end_byte: new_end_b,
            start_position: point(start_b),
            old_end_position: point(old_end_b),
            new_end_position: point(new_end_b),
        });
        self.tree = Some(t);
    }

    /// Full or incremental (whichever tree state allows) parse of `rope`.
    pub fn parse_with(&mut self, rope: &Rope) {
        let text = rope_to_source(rope);
        self.tree = self.parser.parse(&text, self.tree.as_ref());
    }

    /// Incremental: apply the last edit to the old tree, reparse reusing it.
    pub fn update(&mut self, rope: &Rope, edit: Option<(usize, usize, usize)>) {
        if let (Some(tree), Some((start_b, old_end_b, new_end_b))) =
            (self.tree.take(), edit)
        {
            let start_char = rope.try_byte_to_char(start_b).unwrap_or(0);
            let old_end_char = rope.try_byte_to_char(old_end_b).unwrap_or(0);
            let new_end_char = (start_char + (new_end_b - start_b)).min(rope.len_chars());
            let point = |byte: usize| {
                let line = rope.try_byte_to_line(byte).unwrap_or(0);
                let ls = rope.try_line_to_byte(line).unwrap_or(0);
                tree_sitter::Point {
                    row: line,
                    column: byte.saturating_sub(ls),
                }
            };
            let mut t = tree;
            t.edit(&InputEdit {
                start_byte: start_b,
                old_end_byte: old_end_b,
                new_end_byte: new_end_b,
                start_position: point(start_b),
                old_end_position: point(old_end_b),
                new_end_position: point(new_end_b),
            });
            let text = rope_to_source(rope);
            self.tree = self.parser.parse(&text, Some(&t));
            let _ = (start_char, old_end_char, new_end_char);
        } else {
            let text = rope_to_source(rope);
            self.tree = self.parser.parse(&text, None);
        }
    }

    /// Classify tokens overlapping char range [a,b). Returns (start_char, end_char, kind).
    pub fn highlights(&self, rope: &Rope, a: usize, b: usize) -> Vec<(usize, usize, Kind)> {
        let Some(tree) = &self.tree else {
            return Vec::new();
        };
        let byte_a = rope.char_to_byte(a);
        let byte_b = rope.char_to_byte(b.min(rope.len_chars()));
        let mut out = Vec::new();
        let mut cursor = tree.walk();
        let mut stack = vec![tree.root_node()];
        while let Some(node) = stack.pop() {
            if node.byte_range().end < byte_a || node.byte_range().start > byte_b {
                continue;
            }
            let kind = classify(node.kind(), node.parent().map(|p| p.kind()));
            let is_leaf = node.child_count() == 0;
            if is_leaf {
                if kind != Kind::Text {
                    if let (Ok(s), Ok(e)) = (
                        rope.try_byte_to_char(node.start_byte()),
                        rope.try_byte_to_char(node.end_byte()),
                    ) {
                        out.push((s, e.min(b), kind));
                    }
                }
                continue;
            }
            // interior nodes: never record (a parent span would shadow its
            // children in first-match color lookup) — always descend.
            for i in 0..node.child_count() {
                if let Some(c) = node.child(i) {
                    stack.push(c);
                }
            }
        }
        let _ = &mut cursor;
        out.sort_unstable();
        out
    }
}

impl Syntax {
    /// test-only: flattened leaf dump
    pub fn debug_tree(&self, _rope: &Rope) -> String {
        let Some(tree) = &self.tree else { return "<none>".into() };
        let mut out = String::new();
        let mut cur = tree.walk();
        let mut stack = vec![tree.root_node()];
        while let Some(n) = stack.pop() {
            if n.child_count() == 0 {
                out.push_str(&format!("[{}..{}:{}] ", n.start_byte(), n.end_byte(), n.kind()));
            } else {
                for i in (0..n.child_count()).rev() {
                    if let Some(c) = n.child(i) {
                        stack.push(c);
                    }
                }
            }
        }
        let _ = &mut cur;
        out
    }
}

fn rope_to_source(rope: &Rope) -> String {
    // Full snapshot for parse; incremental trees keep reparse cost small.
    rope.slice(..).to_string()
}

const KEYWORDS: &[&str] = &[
    "let", "mut", "fn", "struct", "enum", "impl", "trait", "use", "pub", "mod", "match", "if",
    "else", "for", "while", "loop", "return", "const", "static", "true", "false", "async", "await",
    "where", "self", "Self", "def", "class", "import", "from", "as", "in", "not", "and", "or",
    "lambda", "yield", "try", "except", "finally", "with", "function", "var", "new", "typeof",
    "export", "default", "delete", "instanceof", "void", "break", "continue", "elif", "pass",
];

fn classify(kind: &str, parent: Option<&str>) -> Kind {
    // tree-sitter-rust names comment tokens "//" / "/*"; others say "comment"
    if kind.contains("comment") || kind.starts_with("//") || kind.starts_with("/*") {
        return Kind::Comment;
    }
    if kind.contains("string") || kind.contains("character") {
        return Kind::String;
    }
    if kind.contains("integer")
        || kind.contains("float")
        || kind.contains("number")
        || kind == "true"
        || kind == "false"
    {
        return Kind::Number;
    }
    if KEYWORDS.contains(&kind) {
        return Kind::Keyword;
    }
    if kind == "identifier" {
        if let Some(p) = parent {
            if p.contains("function") || p.contains("call") || p == "method_declaration" {
                return Kind::Function;
            }
            if p.contains("type") || p.contains("struct") || p.contains("class") {
                return Kind::Type;
            }
        }
        return Kind::Text;
    }
    if kind.contains("function") && kind.contains("item") {
        return Kind::Function;
    }
    if kind.contains("type") || kind.contains("struct_item") || kind.contains("class") {
        return Kind::Type;
    }
    if kind.contains("primitive_type") {
        return Kind::Type;
    }
    if kind == "(" || kind == ")" || kind == "{" || kind == "}" || kind == "[" || kind == "]" {
        return Kind::Punct;
    }
    Kind::Text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_parse_and_classify() {
        let mut s = Syntax::new("main.rs").expect("lang");
        let src = "fn main() { // hi\n let x = 42; }";
        let rope = Rope::from_str(src);
        s.update(&rope, None);
        let hs = s.highlights(&rope, 0, rope.len_chars());
        if !hs.iter().any(|h| h.2 == Kind::Comment) {
            panic!("no comment; leaves: {}", s.debug_tree(&rope));
        }
        assert!(hs.iter().any(|h| h.2 == Kind::Keyword), "{hs:?}");
        assert!(hs.iter().any(|h| h.2 == Kind::Number), "{hs:?}");
    }

    #[test]
    fn incremental_reparse_survives_edit() {
        let mut s = Syntax::new("a.py").expect("lang");
        let mut rope = Rope::from_str("def f():\n    return 1\n");
        s.update(&rope, None);
        // insert a line at the top
        rope.insert(0, "# header\n");
        let start_b = 0;
        s.update(&rope, Some((start_b, start_b, start_b + 9)));
        let hs = s.highlights(&rope, 0, rope.len_chars());
        assert!(hs.iter().any(|h| h.2 == Kind::Comment), "{hs:?}");
    }
}
