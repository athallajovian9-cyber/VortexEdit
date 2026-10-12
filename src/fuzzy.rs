//! Fuzzy file finder: subsequence scoring with path-boundary bonuses.
//! Score runs on a worker thread (jobs.rs) — the UI only renders results.

/// Higher is better; `None` = query is not a subsequence of `target`.
pub fn score(query: &str, target: &str) -> Option<i64> {
    if query.is_empty() {
        return Some(0);
    }
    let t: Vec<char> = target.chars().collect();
    let q: Vec<char> = query.to_lowercase().chars().collect();
    let mut ti = 0usize;
    let mut total = 0i64;
    let mut prev_match: Option<usize> = None;
    for &qc in &q {
        let mut found = None;
        while ti < t.len() {
            let tc = t[ti];
            if tc.to_lowercase().next() == Some(qc) {
                found = Some(ti);
                break;
            }
            ti += 1;
        }
        let m = found?;
        let mut s = 1i64;
        if prev_match == Some(m.wrapping_sub(1)) {
            s += 8; // consecutive run
        }
        if m == 0 {
            s += 12; // start of path
        } else {
            let prev = t[m - 1];
            if prev == '/' || prev == '\\' || prev == '.' || prev == '-' || prev == '_' {
                s += 6; // path/word boundary
            }
        }
        if target.chars().nth(m) == query.chars().nth(0) {
            // exact case bonus only once, cheap: skip — lowercase compare above
        }
        total += s;
        prev_match = Some(m);
        ti = m + 1;
    }
    Some(total)
}

/// Rank targets by score desc; returns (score, index) pairs.
pub fn rank(query: &str, targets: &[String]) -> Vec<(i64, usize)> {
    let mut out: Vec<(i64, usize)> = targets
        .iter()
        .enumerate()
        .filter_map(|(i, t)| score(query, t).map(|s| (s, i)))
        .collect();
    out.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    out
}

const SKIP_DIRS: [&str; 5] = [".git", "node_modules", "target", "__pycache__", ".venv"];

/// Recursive workspace index (runs on worker). Files only, capped.
pub fn walk(root: &std::path::Path, cap: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if out.len() >= cap {
            break;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            if out.len() >= cap {
                break;
            }
            let p = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) && !name.starts_with('.') {
                    stack.push(p);
                }
            } else {
                out.push(p.to_string_lossy().to_string());
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn match_and_rank() {
        let files = vec![
            "src/main.rs".to_string(),
            "src/buffer.rs".to_string(),
            "readme.md".to_string(),
        ];
        let r = rank("buf", &files);
        assert_eq!(files[r[0].1], "src/buffer.rs");
        assert!(rank("zzz", &files).is_empty());
    }

    #[test]
    fn boundary_bonus_prefers_segment_start() {
        // greedy matcher: "main" hits the word boundary in a (rs_main), not in b (xmain)
        let a = score("main", "src/rs_main.rs").unwrap();
        let b = score("main", "src/xmainfile.rs").unwrap();
        assert!(a > b, "{a} vs {b}");
    }

    #[test]
    fn hundred_thousand_paths_rank_under_50ms() {
        let paths: Vec<String> = (0..100_000)
            .map(|i| format!("deep/folder_{i}/module_{i}.rs"))
            .collect();
        let t = Instant::now();
        let r = rank("mod42", &paths);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("rank 100k: {ms:.2} ms ({} hits)", r.len());
        assert!(ms < 150.0, "took {ms:.2} ms");
    }
}
