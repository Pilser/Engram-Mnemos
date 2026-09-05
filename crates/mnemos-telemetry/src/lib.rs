//! mnemos-telemetry: cross-crate failure tracking and diagnosis.
//!
//! Every fallible operation records an [`Event`] (what ran, where, whether it
//! worked, why not, how long it took). Events live in a bounded in-memory
//! ring and are appended as JSONL to **per-day files** (`YYYY-MM-DD.jsonl`)
//! inside the telemetry folder for dashboard use and offline analysis.
//!
//! Use the [`global`] instance from any crate — no plumbing required:
//!
//! ```rust,no_run
//! # use mnemos_telemetry::global;
//! # async fn f() -> Result<String, String> {
//! let start = std::time::Instant::now();
//! let out = fallible().await;
//! global().record("mnemos-ingestion", "ingest", out.is_ok(), &match &out {
//!     Ok(_) => String::new(),
//!     Err(e) => e.clone(),
//! });
//! let _ = start;
//! # Ok(String::new())
//! # }
//! # async fn fallible() -> Result<String, String> { Ok(String::new()) }
//! ```

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Maximum events kept in memory (oldest evicted first).
pub const MAX_EVENTS: usize = 1024;

/// Maximum weight/system snapshots kept.
pub const MAX_SNAPSHOTS: usize = 256;

/// One recorded operation outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// RFC 3339 timestamp.
    pub ts: String,
    /// Originating crate, e.g. `"mnemos-ingestion"`.
    pub crate_name: String,
    /// Operation, e.g. `"llm.chat"`, `"ingest"`, `"recall"`.
    pub op: String,
    /// Whether the operation succeeded.
    pub ok: bool,
    /// Failure reason or notable detail (raw LLM snippets truncated).
    pub detail: String,
    /// Wall time in milliseconds.
    pub latency_ms: u64,
    /// Optional trace correlation id linking recall→reward→consolidate for one session.
    #[serde(default)]
    pub trace_id: String,
    /// Structured meta for filtering (model, `recall_id`, tokens, `query_len`, etc.).
    #[serde(default)]
    pub meta: HashMap<String, String>,
}

/// Per-`crate::op` counter for live dashboards.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Counter {
    pub total: u64,
    pub ok: u64,
    pub err: u64,
    pub latency_sum_ms: u64,
    pub latency_min_ms: u64,
    pub latency_max_ms: u64,
}

impl Counter {
    #[must_use]
    pub fn avg_latency(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.latency_sum_ms as f64 / self.total as f64
        }
    }
}

/// Weight snapshot for `EdgeWeights` evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSnapshot {
    pub ts: String,
    pub weights: serde_json::Value,
}

/// System stats snapshot for `MemoryStats` evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatsSnapshot {
    pub ts: String,
    pub stats: serde_json::Value,
}

/// Aggregate failure counts keyed by `(crate, op)`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FailureSummary {
    /// Total events recorded.
    pub total: u64,
    /// Failures grouped as `"crate::op" -> count`.
    pub failures: HashMap<String, u64>,
    /// Latest failure detail per `"crate::op"`.
    pub latest_detail: HashMap<String, String>,
}

/// One failing `(crate, op)` pair with its count and latest detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePair {
    /// `"crate::op"` key.
    pub key: String,
    /// How many times this pair failed.
    pub count: u64,
    /// Latest failure detail for this pair.
    pub latest: String,
}

/// Cross-project diagnosis report from [`Telemetry::diagnose`].
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Diagnosis {
    /// Total events recorded.
    pub total_events: u64,
    /// Total failed events.
    pub total_failures: u64,
    /// Top failing `(crate, op)` pairs, sorted by count descending.
    pub top_failures: Vec<FailurePair>,
    /// Most recent N failure events (newest first).
    pub recent_failures: Vec<Event>,
}

/// Metadata for one per-day telemetry file (dashboard file manager).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryFile {
    /// File name, always `YYYY-MM-DD.jsonl`.
    pub name: String,
    /// Date part (`YYYY-MM-DD`).
    pub date: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Number of parseable event lines.
    pub events: usize,
}

/// Bounded recorder with per-day JSONL file sink.
pub struct Telemetry {
    inner: Mutex<Inner>,
    dir: Option<PathBuf>,
}

struct Inner {
    enabled: bool,
    events: VecDeque<Event>,
    total: u64,
    counters: HashMap<String, Counter>,
    weights_history: VecDeque<WeightSnapshot>,
    system_history: VecDeque<SystemStatsSnapshot>,
    /// Cached open file handle: (date `YYYY-MM-DD`, file). Rotated when the
    /// UTC date changes so each day gets its own file.
    file: Option<(String, std::fs::File)>,
}

impl Telemetry {
    /// Build from env: `MNEMOS_TELEMETRY` (`1`/`true`/`yes`, default on),
    /// `MNEMOS_TELEMETRY_DIR` (folder holding per-day `YYYY-MM-DD.jsonl`
    /// files, default `./data/helix/telemetry`).
    #[must_use]
    pub fn from_env() -> Self {
        let enabled = std::env::var("MNEMOS_TELEMETRY").ok().is_none_or(|v| {
            matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
        });
        let dir = std::env::var("MNEMOS_TELEMETRY_DIR")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
            .or_else(|| Some(PathBuf::from("./data/helix/telemetry")));
        Self::new(enabled, dir)
    }

    /// Build explicitly (tests, embedding in other config systems).
    /// `dir` receives per-day `YYYY-MM-DD.jsonl` files; `None` disables files.
    #[must_use]
    pub fn new(enabled: bool, dir: Option<PathBuf>) -> Self {
        Self {
            inner: Mutex::new(Inner {
                enabled,
                events: VecDeque::with_capacity(MAX_EVENTS.min(64)),
                total: 0,
                counters: HashMap::new(),
                weights_history: VecDeque::with_capacity(MAX_SNAPSHOTS.min(32)),
                system_history: VecDeque::with_capacity(MAX_SNAPSHOTS.min(32)),
                file: None,
            }),
            dir,
        }
    }

    /// Record one outcome. No-op when disabled. Never panics, never blocks
    /// on I/O beyond one append write.
    pub fn record(
        &self,
        crate_name: &'static str,
        op: &'static str,
        ok: bool,
        detail: &str,
    ) {
        self.record_with_latency(crate_name, op, ok, detail, 0);
    }

    /// Record with trace correlation and structured meta.
    pub fn record_with_meta(
        &self,
        crate_name: &'static str,
        op: &'static str,
        ok: bool,
        detail: &str,
        latency_ms: u64,
        trace_id: &str,
        meta: HashMap<String, String>,
    ) {
        let event = Event {
            ts: chrono::Utc::now().to_rfc3339(),
            crate_name: crate_name.to_string(),
            op: op.to_string(),
            ok,
            detail: truncate(detail, 500),
            latency_ms,
            trace_id: trace_id.to_string(),
            meta,
        };
        self.push_event(event);
    }

    /// Record with an explicit latency (measure with [`Instant`] at the call site).
    pub fn record_with_latency(
        &self,
        crate_name: &'static str,
        op: &'static str,
        ok: bool,
        detail: &str,
        latency_ms: u64,
    ) {
        let event = Event {
            ts: chrono::Utc::now().to_rfc3339(),
            crate_name: crate_name.to_string(),
            op: op.to_string(),
            ok,
            detail: truncate(detail, 500),
            latency_ms,
            trace_id: String::new(),
            meta: HashMap::new(),
        };
        self.push_event(event);
    }

    fn push_event(&self, event: Event) {
        if let Some(dir) = &self.dir.clone() {
            // Append to today's file; rotate the cached handle on date change.
            let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
            let _ = std::fs::create_dir_all(dir);
            if let Ok(line) = serde_json::to_string(&event) {
                use std::io::Write as _;
                if let Ok(mut inner) = self.inner.lock() {
                    let same_day = inner
                        .file
                        .as_ref()
                        .is_some_and(|(d, _)| d == &today);
                    if !same_day {
                        let path = dir.join(format!("{today}.jsonl"));
                        if let Ok(f) = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(path)
                        {
                            inner.file = Some((today, f));
                        }
                    }
                    if let Some((_, f)) = inner.file.as_mut() {
                        let _ = writeln!(f, "{line}");
                    }
                }
            }
        }
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if !inner.enabled {
            return;
        }
        // Counters for live dashboards.
        let key = format!("{}::{}", event.crate_name, event.op);
        let counter = inner.counters.entry(key).or_default();
        counter.total += 1;
        if event.ok {
            counter.ok += 1;
        } else {
            counter.err += 1;
        }
        counter.latency_sum_ms += event.latency_ms;
        if counter.total == 1 {
            counter.latency_min_ms = event.latency_ms;
            counter.latency_max_ms = event.latency_ms;
        } else {
            counter.latency_min_ms = counter.latency_min_ms.min(event.latency_ms);
            counter.latency_max_ms = counter.latency_max_ms.max(event.latency_ms);
        }
        inner.total += 1;
        if inner.events.len() >= MAX_EVENTS {
            inner.events.pop_front();
        }
        inner.events.push_back(event);
    }

    /// Time an async closure, recording its outcome automatically.
    pub async fn time_async<F, T, E>(
        &self,
        crate_name: &'static str,
        op: &'static str,
        fut: F,
    ) -> Result<T, E>
    where
        F: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let start = Instant::now();
        let out = fut.await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match &out {
            Ok(_) => self.record_with_latency(crate_name, op, true, "", ms),
            Err(e) => {
                self.record_with_latency(crate_name, op, false, &e.to_string(), ms);
            }
        }
        out
    }

    /// Newest-first snapshot of buffered events.
    pub fn snapshot(&self) -> Vec<Event> {
        self.inner
            .lock()
            .map(|i| i.events.iter().rev().cloned().collect())
            .unwrap_or_default()
    }

    /// Failure counts + latest reasons, for diagnosis.
    pub fn failure_summary(&self) -> FailureSummary {
        let mut summary = FailureSummary::default();
        if let Ok(inner) = self.inner.lock() {
            summary.total = inner.total;
            for e in &inner.events {
                if !e.ok {
                    let key = format!("{}::{}", e.crate_name, e.op);
                    *summary.failures.entry(key.clone()).or_insert(0) += 1;
                    summary.latest_detail.insert(key, e.detail.clone());
                }
            }
        }
        summary
    }

    /// Structured diagnosis report for cross-project failure analysis.
    ///
    /// Returns a `Diagnosis` with:
    /// - total event count and failure count
    /// - top failing `(crate, op)` pairs sorted by count descending
    /// - the most recent N failure events (newest first)
    ///
    /// Use this to answer "what failed and why across the project".
    pub fn diagnose(&self, recent_n: usize) -> Diagnosis {
        let mut diagnosis = Diagnosis::default();
        if let Ok(inner) = self.inner.lock() {
            diagnosis.total_events = inner.total;
            let mut pairs: Vec<(String, u64, String)> = Vec::new();
            for e in &inner.events {
                if !e.ok {
                    diagnosis.total_failures += 1;
                    let key = format!("{}::{}", e.crate_name, e.op);
                    if let Some(pos) = pairs.iter().position(|(k, _, _)| k == &key) {
                        pairs[pos].1 += 1;
                        pairs[pos].2 = e.detail.clone();
                    } else {
                        pairs.push((key, 1, e.detail.clone()));
                    }
                }
            }
            // Sort by count descending.
            pairs.sort_by(|a, b| b.1.cmp(&a.1));
            diagnosis.top_failures = pairs
                .into_iter()
                .map(|(key, count, latest)| FailurePair { key, count, latest })
                .collect();
            // Most recent N failures (newest first).
            diagnosis.recent_failures = inner
                .events
                .iter()
                .rev()
                .filter(|e| !e.ok)
                .take(recent_n)
                .cloned()
                .collect();
        }
        diagnosis
    }

    /// Record an `EdgeWeights` snapshot for dashboard evolution charts.
    pub fn record_weights(&self, weights: serde_json::Value) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if !inner.enabled {
            return;
        }
        if inner.weights_history.len() >= MAX_SNAPSHOTS {
            inner.weights_history.pop_front();
        }
        inner.weights_history.push_back(WeightSnapshot {
            ts: chrono::Utc::now().to_rfc3339(),
            weights,
        });
    }

    /// Record a `MemoryStats` snapshot for dashboard evolution charts.
    pub fn record_system_stats(&self, stats: serde_json::Value) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if !inner.enabled {
            return;
        }
        if inner.system_history.len() >= MAX_SNAPSHOTS {
            inner.system_history.pop_front();
        }
        inner.system_history.push_back(SystemStatsSnapshot {
            ts: chrono::Utc::now().to_rfc3339(),
            stats,
        });
    }

    /// Per-`crate::op` counters for live dashboards (avg latency, ok/err).
    pub fn counters_snapshot(&self) -> HashMap<String, Counter> {
        self.inner
            .lock()
            .map(|i| i.counters.clone())
            .unwrap_or_default()
    }

    /// Recent weight snapshots (oldest first).
    pub fn weights_history_snapshot(&self) -> Vec<WeightSnapshot> {
        self.inner
            .lock()
            .map(|i| i.weights_history.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Recent system stats snapshots (oldest first).
    pub fn system_history_snapshot(&self) -> Vec<SystemStatsSnapshot> {
        self.inner
            .lock()
            .map(|i| i.system_history.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Validate a `YYYY-MM-DD` date string (strict — prevents path traversal).
    #[must_use]
    pub fn valid_date(date: &str) -> bool {
        let b = date.as_bytes();
        if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
            return false;
        }
        for (i, c) in b.iter().enumerate() {
            if i == 4 || i == 7 {
                continue;
            }
            if !c.is_ascii_digit() {
                return false;
            }
        }
        true
    }

    /// Today's date as `YYYY-MM-DD` (UTC).
    #[must_use]
    pub fn today() -> String {
        chrono::Utc::now().format("%Y-%m-%d").to_string()
    }

    /// List per-day telemetry files (newest first) for the dashboard file manager.
    /// Returns empty when no dir is configured.
    pub fn telemetry_files(&self) -> Vec<TelemetryFile> {
        let Some(dir) = &self.dir else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut files: Vec<TelemetryFile> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|x| x == "jsonl")
                    && p.file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(Self::valid_date)
            })
            .filter_map(|p| {
                let meta = std::fs::metadata(&p).ok()?;
                let name = p.file_name()?.to_str()?.to_string();
                let date = name.trim_end_matches(".jsonl").to_string();
                // Count parseable event lines (cheap scan, no full load).
                let content = std::fs::read_to_string(&p).unwrap_or_default();
                let events = content
                    .lines()
                    .filter(|l| serde_json::from_str::<Event>(l).is_ok())
                    .count();
                Some(TelemetryFile {
                    name,
                    date,
                    size_bytes: meta.len(),
                    events,
                })
            })
            .collect();
        files.sort_by(|a, b| b.date.cmp(&a.date));
        files
    }

    /// Read events from one per-day file (newest first).
    ///
    /// `limit`/`offset` page through the file (defaults 200/0, capped at 5000).
    /// `ok_only`: `Some(true)` keeps successes, `Some(false)` keeps failures,
    /// `None` keeps all. Returns `None` for an invalid date or missing dir/file.
    pub fn read_file(
        &self,
        date: &str,
        limit: usize,
        offset: usize,
        ok_only: Option<bool>,
    ) -> Option<Vec<Event>> {
        if !Self::valid_date(date) {
            return None;
        }
        let dir = self.dir.as_ref()?;
        let content = std::fs::read_to_string(dir.join(format!("{date}.jsonl"))).ok()?;
        let limit = limit.min(5000);
        let mut events: Vec<Event> = content
            .lines()
            .filter_map(|l| serde_json::from_str::<Event>(l).ok())
            .filter(|e| ok_only.is_none_or(|want| e.ok == want))
            .collect();
        events.reverse();
        Some(events.into_iter().skip(offset).take(limit).collect())
    }

    /// Delete one per-day file. Returns `true` if a file was removed.
    /// Invalid dates always return `false`. Also drops the cached open
    /// handle for that date — otherwise further writes would go to the
    /// unlinked inode and silently vanish.
    pub fn delete_file(&self, date: &str) -> bool {
        if !Self::valid_date(date) {
            return false;
        }
        let Some(dir) = &self.dir else {
            return false;
        };
        let removed = std::fs::remove_file(dir.join(format!("{date}.jsonl"))).is_ok();
        if removed {
            if let Ok(mut inner) = self.inner.lock() {
                if inner.file.as_ref().is_some_and(|(d, _)| d == date) {
                    inner.file = None;
                }
            }
        }
        removed
    }

    /// Full snapshot for `GET /telemetry` — everything a dashboard needs.
    pub fn full_snapshot(&self) -> serde_json::Value {
        let inner = match self.inner.lock() {
            Ok(i) => i,
            Err(_) => return serde_json::json!({}),
        };
        let diagnosis = {
            let mut d = Diagnosis::default();
            d.total_events = inner.total;
            let mut pairs: Vec<(String, u64, String)> = Vec::new();
            for e in &inner.events {
                if !e.ok {
                    d.total_failures += 1;
                    let key = format!("{}::{}", e.crate_name, e.op);
                    if let Some(pos) = pairs.iter().position(|(k, _, _)| k == &key) {
                        pairs[pos].1 += 1;
                        pairs[pos].2 = e.detail.clone();
                    } else {
                        pairs.push((key, 1, e.detail.clone()));
                    }
                }
            }
            pairs.sort_by(|a, b| b.1.cmp(&a.1));
            d.top_failures = pairs
                .into_iter()
                .map(|(key, count, latest)| FailurePair { key, count, latest })
                .collect();
            d.recent_failures = inner.events.iter().rev().filter(|e| !e.ok).take(20).cloned().collect();
            d
        };
        serde_json::json!({
            "total_events": inner.total,
            "events": inner.events.iter().rev().take(100).cloned().collect::<Vec<_>>(),
            "diagnosis": diagnosis,
            "counters": inner.counters,
            "weights_history": inner.weights_history.iter().cloned().collect::<Vec<_>>(),
            "system_history": inner.system_history.iter().cloned().collect::<Vec<_>>(),
        })
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::from_env()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…[{} bytes total]", &s[..max], s.len())
    }
}

static GLOBAL: OnceLock<Telemetry> = OnceLock::new();

/// Process-wide instance, built once from env.
pub fn global() -> &'static Telemetry {
    GLOBAL.get_or_init(Telemetry::from_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local() -> Telemetry {
        Telemetry::new(true, None)
    }

    #[test]
    fn records_ok_and_err() {
        let t = local();
        t.record("c", "op", true, "");
        t.record("c", "op", false, "boom");
        let snap = t.snapshot();
        assert_eq!(snap.len(), 2);
        assert!(!snap[0].ok);
        assert_eq!(snap[0].detail, "boom");
    }

    #[test]
    fn disabled_records_nothing() {
        let t = Telemetry::new(false, None);
        t.record("c", "op", false, "x");
        assert!(t.snapshot().is_empty());
        assert_eq!(t.failure_summary().total, 0);
    }

    #[test]
    fn summary_counts_failures() {
        let t = local();
        t.record("a", "x", false, "first");
        t.record("a", "x", false, "second");
        t.record("a", "y", true, "");
        let s = t.failure_summary();
        assert_eq!(s.total, 3);
        assert_eq!(s.failures["a::x"], 2);
        assert_eq!(s.latest_detail["a::x"], "second");
        assert!(!s.failures.contains_key("a::y"));
    }

    #[test]
    fn ring_evicts_oldest() {
        let t = local();
        for i in 0..MAX_EVENTS + 10 {
            t.record("c", "op", true, &i.to_string());
        }
        assert_eq!(t.snapshot().len(), MAX_EVENTS);
    }

    #[tokio::test]
    async fn time_async_records_outcome() {
        let t = local();
        let ok: Result<u32, String> = t.time_async("c", "op", async { Ok(1) }).await;
        assert_eq!(ok.unwrap(), 1);
        let err: Result<u32, String> =
            t.time_async("c", "op", async { Err("bad".to_string()) }).await;
        assert!(err.is_err());
        assert_eq!(t.failure_summary().failures["c::op"], 1);
    }

    #[test]
    fn diagnose_ranks_failures_and_captures_recent() {
        let t = local();
        // 3 failures of a::x, 1 failure of a::y, 2 ok events.
        t.record("a", "x", false, "first");
        t.record("a", "y", false, "only");
        t.record("a", "x", false, "second");
        t.record("a", "x", false, "third");
        t.record("a", "x", true, "");
        t.record("a", "y", true, "");
        let d = t.diagnose(2);
        assert_eq!(d.total_events, 6);
        assert_eq!(d.total_failures, 4);
        // a::x should be top (3 failures).
        assert_eq!(d.top_failures[0].key, "a::x");
        assert_eq!(d.top_failures[0].count, 3);
        assert_eq!(d.top_failures[0].latest, "third");
        // a::y second (1 failure).
        assert_eq!(d.top_failures[1].key, "a::y");
        assert_eq!(d.top_failures[1].count, 1);
        // Recent 2 failures (newest first: third, second).
        assert_eq!(d.recent_failures.len(), 2);
        assert_eq!(d.recent_failures[0].detail, "third");
        assert_eq!(d.recent_failures[1].detail, "second");
    }

    #[test]
    fn valid_date_accepts_only_ymd() {
        assert!(Telemetry::valid_date("2026-09-05"));
        assert!(!Telemetry::valid_date("../secret"));
        assert!(!Telemetry::valid_date("2026-9-5"));
        assert!(!Telemetry::valid_date("2026-09-05.jsonl"));
        assert!(!Telemetry::valid_date(""));
        assert!(!Telemetry::valid_date("2026/09/05"));
    }

    #[test]
    fn day_files_list_read_delete() {
        let dir = std::env::temp_dir().join(format!(
            "mnemos-tel-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let t = Telemetry::new(true, Some(dir.clone()));
        t.record("a", "x", true, "");
        t.record("a", "y", false, "boom");
        // A planted non-telemetry file must not appear.
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), "not telemetry").unwrap();

        let files = t.telemetry_files();
        assert_eq!(files.len(), 1);
        assert!(files[0].name.ends_with(".jsonl"));
        assert_eq!(files[0].events, 2);

        let read = t.read_file(&files[0].date, 200, 0, None).expect("read day");
        assert_eq!(read.len(), 2);
        assert!(!read[0].ok); // newest first
        let fails = t
            .read_file(&files[0].date, 200, 0, Some(false))
            .expect("read fails");
        assert_eq!(fails.len(), 1);
        let paged = t
            .read_file(&files[0].date, 1, 1, None)
            .expect("read page");
        assert_eq!(paged.len(), 1);

        assert!(!t.delete_file("../evil"));
        assert!(t.delete_file(&files[0].date));
        assert!(t.telemetry_files().is_empty());
        assert!(t.read_file(&files[0].date, 200, 0, None).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
