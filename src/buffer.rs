//! Text core: rope-backed buffer + multi-cursor engine.
//!
//! Invariants:
//!   - cursors are unique, sorted char indices; every operation restores both
//!   - all cursors receive every edit (10k+ must stay sub-frame)
//!   - revision bumps on every mutation; downstream caches key on it

use ropey::Rope;

#[derive(Default)]
pub struct Editor {
    pub rope: Rope,
    /// sorted, unique char indices
    pub cursors: Vec<usize>,
    pub revision: u64,
    /// byte range of last edit (for incremental tree-sitter input_edit)
    pub last_edit: Option<(usize, usize, usize)>, // (start_b, old_end_b, new_end_b)
}

impl Editor {
    pub fn new(text: &str) -> Self {
        Self {
            rope: Rope::from_str(text),
            cursors: vec![0],
            revision: 0,
            last_edit: None,
        }
    }

    pub fn line_col(&self, idx: usize) -> (usize, usize) {
        let line = self.rope.try_byte_to_char(
            self.rope
                .try_line_to_byte(self.rope.try_char_to_line(idx).unwrap_or(0))
                .unwrap_or(0),
        );
        let line_char_start = line.unwrap_or(0);
        let line = self.rope.try_char_to_line(idx).unwrap_or(0);
        let col = idx.saturating_sub(line_char_start);
        (line, col)
    }

    fn touch(&mut self, start_b: usize, old_end_b: usize, new_end_b: usize) {
        self.revision += 1;
        self.last_edit = Some((start_b, old_end_b, new_end_b));
    }

    /// Insert `text` at every cursor. O(cursors · log n).
    pub fn insert(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let delta = text.chars().count();
        let mut c = self.cursors.clone();
        c.sort_unstable();
        c.dedup();
        let mut first_start_b = 0;
        for (i, &idx) in c.iter().enumerate().rev() {
            let start_b = self.rope.char_to_byte(idx);
            if i == 0 {
                first_start_b = start_b;
            }
            self.rope.insert(idx, text);
        }
        let new_end_b = first_start_b + text.len();
        for cur in &mut self.cursors {
            // cursors at/after an insertion point shift forward
            *cur += delta;
        }
        self.touch(first_start_b, first_start_b, new_end_b);
        self.normalize();
    }

    /// Delete one char before every cursor (merge colliding cursors).
    pub fn backspace(&mut self) {
        let mut c = self.cursors.clone();
        c.sort_unstable();
        c.dedup();
        let mut touched_start_b = None;
        for &idx in c.iter().rev() {
            if idx == 0 {
                continue;
            }
            let start_b = self.rope.char_to_byte(idx - 1);
            let end_b = self.rope.char_to_byte(idx);
            touched_start_b.get_or_insert(start_b);
            self.rope.remove((idx - 1)..idx);
            for cur in &mut self.cursors {
                if *cur >= idx {
                    *cur -= 1;
                }
            }
        }
        if let Some(sb) = touched_start_b {
            self.touch(sb, sb + 1, sb);
        }
        self.normalize();
    }

    /// Delete forward (Delete key).
    pub fn delete_forward(&mut self) {
        let len = self.rope.len_chars();
        let mut c = self.cursors.clone();
        c.sort_unstable();
        c.dedup();
        let mut touched = None;
        for &idx in c.iter().rev() {
            if idx >= len {
                continue;
            }
            let start_b = self.rope.char_to_byte(idx);
            let end_b = self.rope.char_to_byte(idx + 1);
            touched.get_or_insert((start_b, end_b));
            self.rope.remove(idx..(idx + 1));
        }
        if let Some((sb, eb)) = touched {
            self.touch(sb, eb, sb);
        }
        self.normalize();
    }

    pub fn normalize(&mut self) {
        let len = self.rope.len_chars();
        for c in &mut self.cursors {
            *c = (*c).min(len);
        }
        self.cursors.sort_unstable();
        self.cursors.dedup();
        if self.cursors.is_empty() {
            self.cursors.push(0);
        }
    }

    pub fn add_cursor(&mut self, idx: usize) {
        self.cursors.push(idx.min(self.rope.len_chars()));
        self.normalize();
    }

    /// Primary (last-focused) cursor.
    pub fn primary(&self) -> usize {
        *self.cursors.last().unwrap_or(&0)
    }

    pub fn set_primary(&mut self, idx: usize) {
        self.cursors.clear();
        self.cursors.push(idx.min(self.rope.len_chars()));
    }

    /// Word boundaries around `idx` (identifier chars).
    pub fn word_at(&self, idx: usize) -> Option<(usize, usize)> {
        let is_word = |c: char| c.is_alphanumeric() || c == '_';
        let chars: Vec<char> = self.rope.slice(..).chars().collect();
        if chars.is_empty() {
            return None;
        }
        let mut i = idx.min(chars.len());
        if i == chars.len() || !is_word(chars[i]) {
            if i == 0 || !is_word(chars[i.saturating_sub(1)]) {
                return None;
            }
            i -= 1;
        }
        let mut a = i;
        while a > 0 && is_word(chars[a - 1]) {
            a -= 1;
        }
        let mut b = i;
        while b < chars.len() && is_word(chars[b]) {
            b += 1;
        }
        if a >= b {
            None
        } else {
            Some((a, b))
        }
    }

    /// Char range of a line (start..end, excluding newline).
    pub fn line_range(&self, line: usize) -> Option<(usize, usize)> {
        if line >= self.rope.len_lines() {
            return None;
        }
        let s = self.rope.try_line_to_char(line).ok()?;
        let e = self
            .rope
            .try_line_to_char(line + 1)
            .unwrap_or(self.rope.len_chars());
        Some((s, e))
    }

    pub fn line_count(&self) -> usize {
        self.rope.len_lines()
    }

    pub fn text_slice(&self, a: usize, b: usize) -> String {
        self.rope.slice(a..b).to_string()
    }

    /// Move every cursor by `delta` chars, clamped.
    pub fn move_all(&self, delta: isize) -> Vec<usize> {
        let len = self.rope.len_chars();
        self.cursors
            .iter()
            .map(|&c| {
                let n = c as isize + delta;
                n.clamp(0, len as isize) as usize
            })
            .collect()
    }

    /// Next occurrence of `word` after `from`, wrapping.
    pub fn find_next(&self, word: &str, from: usize) -> Option<usize> {
        if word.is_empty() {
            return None;
        }
        let hay: String = self.rope.slice(..).chars().collect();
        let start = from.min(hay.len());
        hay[start..]
            .find(word)
            .map(|p| start + p)
            .or_else(|| hay[..start].find(word))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn insert_at_all_cursors_keeps_sorted_unique() {
        let mut e = Editor::new("aaaa");
        e.cursors = vec![0, 2, 4];
        e.insert("X");
        assert_eq!(e.rope.to_string(), "XaaXaaX");
        assert_eq!(e.cursors, vec![1, 3, 5]);
    }

    #[test]
    fn backspace_at_two_cursors_deletes_both() {
        let mut e = Editor::new("ab");
        e.cursors = vec![1, 2];
        e.backspace();
        assert_eq!(e.rope.to_string(), "");
        assert_eq!(e.cursors, vec![0]);
    }

    #[test]
    fn backspace_two_cursors_delete_distinct_preceding_chars() {
        let mut e = Editor::new("abc");
        e.cursors = vec![2, 3]; // deletes 'b' and 'c'
        e.backspace();
        assert_eq!(e.rope.to_string(), "a");
        assert_eq!(e.cursors, vec![1]);
    }

    #[test]
    fn word_at_identifiers() {
        let e = Editor::new("let foo_bar = 1;");
        assert_eq!(e.word_at(4), Some((4, 11)));
        assert_eq!(e.word_at(0), Some((0, 3)));
    }

    #[test]
    fn ten_thousand_cursors_insert_under_50ms() {
        let text: String = (0..2000).map(|_| "fn value_9() { let x = 0; }\n").collect();
        let mut e = Editor::new(&text);
        e.cursors = (0..10_000).map(|i| (i * e.rope.len_chars()) / 10_000).collect();
        e.normalize();
        assert_eq!(e.cursors.len(), 10_000);
        let t = Instant::now();
        e.insert("»");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("10k-cursor insert: {ms:.2} ms");
        assert!(ms < 50.0, "took {ms:.2} ms");
    }

    #[test]
    fn find_next_wraps() {
        let e = Editor::new("x = foo; y = foo;");
        let a = e.find_next("foo", 0).unwrap();
        let b = e.find_next("foo", a + 1).unwrap();
        assert!(b > a);
        assert_eq!(e.find_next("foo", b + 1), Some(a));
    }
}
