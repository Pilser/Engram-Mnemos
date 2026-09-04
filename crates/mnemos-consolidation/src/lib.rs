#![recursion_limit = "256"]
//! mnemos-consolidation: background maintenance ("sleep" cycle).
//!
//! [`ConsolidationPipeline`] runs an Ebbinghaus decay pass over raw engrams
//! (prune / compress / promote) and detects mitosis candidates (concepts
//! recalled by many engrams).
//!
//! Two later enhancements are documented, not implemented:
//! - contradiction detection: [`ConsolidationPipeline::consolidate`] always
//!   reports `contradictions_linked = 0` for now;
//! - concept splitting: [`ConsolidationPipeline::run_mitosis`] performs
//!   detection only and returns the candidate count.

use helix_db::dsl::prelude::*;
use mnemos_core::{
    parse_timestamp_rfc3339, ConsolidationConfig, ConsolidationReport, EngramCandidate,
    MnemosError, Result,
};
use mnemos_storage::Storage;
use serde::Deserialize;
use serde_json::Value;

/// Ebbinghaus retention: `exp(-decay_rate * days / sqrt(activations + 1))`.
///
/// Fresh, frequently activated engrams score near `1.0`; ancient,
/// never-activated ones decay toward `0.0`. Frequent activation slows decay
/// via the `sqrt(activation_count + 1)` divisor.
#[must_use]
pub fn retention(decay_rate: f64, days_elapsed: f64, activation_count: i64) -> f64 {
    ((-decay_rate * days_elapsed) / ((activation_count.max(0) as f64) + 1.0).sqrt()).exp()
}

/// Fetch raw engrams (`compression_level == 0`) for the decay check.
#[query]
fn get_engrams_for_decay_check() -> ReadBatch {
    read_batch()
        .var_as(
            "engrams",
            g().n_with_label("Engram")
                .where_(Predicate::eq("compression_level", 0_i64))
                .value_map(Some(vec![
                    "$id",
                    "decay_rate",
                    "activation_count",
                    "timestamp",
                    "importance_score",
                ])),
        )
        .returning(["engrams"])
}

/// Delete an engram (prune path).
#[query]
fn prune_engram(engram_id: i64) -> WriteBatch {
    // Reference the ident so the macro-generated binding is used; the value
    // reaches HelixDB as the named param behind `NodeRef::param`.
    let _ = &engram_id;
    write_batch()
        .var_as("pruned", g().n(NodeRef::param("engram_id")).drop())
        .returning(["pruned"])
}

/// Mark an engram compressed (`compression_level = 1`; raw text already
/// summarized upstream, so only the level flips here).
#[query]
fn compress_engram(engram_id: i64) -> WriteBatch {
    // See `prune_engram`: keep the macro-generated binding used.
    let _ = &engram_id;
    write_batch()
        .var_as(
            "compressed",
            g().n(NodeRef::param("engram_id"))
                .set_property("compression_level", 1_i64),
        )
        .returning(["compressed"])
}

/// Promote an engram to semantic memory (`engram_type = semantic`,
/// `compression_level = 2` — crystallized).
#[query]
fn promote_to_semantic(engram_id: i64) -> WriteBatch {
    // See `prune_engram`: keep the macro-generated binding used.
    let _ = &engram_id;
    write_batch()
        .var_as(
            "promoted",
            g().n(NodeRef::param("engram_id"))
                .set_property("engram_type", "semantic")
                .set_property("compression_level", 2_i64),
        )
        .returning(["promoted"])
}

/// Fetch concepts recalled by at least `min_engrams` engrams.
#[query]
fn get_mitosis_candidates(min_engrams: i64) -> ReadBatch {
    read_batch()
        .var_as(
            "concepts",
            g().n_with_label("Concept")
                .where_(Predicate::gte("source_count", min_engrams))
                .value_map(Some(vec!["$id", "name", "source_count"])),
        )
        .returning(["concepts"])
}

/// Background maintenance pipeline: decay, compress, promote, mitosis.
pub struct ConsolidationPipeline {
    storage: Storage,
    config: ConsolidationConfig,
}

impl ConsolidationPipeline {
    /// Build a pipeline over `storage` with `config` thresholds.
    #[must_use]
    pub fn new(storage: Storage, config: ConsolidationConfig) -> Self {
        Self { storage, config }
    }

    /// Full sleep cycle: decay pass plus mitosis detection, merged into one
    /// report.
    ///
    /// Contradiction detection is a later enhancement, so the merged report
    /// always carries `contradictions_linked = 0` for now.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when any `HelixDB` query fails.
    pub async fn consolidate(&self) -> Result<ConsolidationReport> {
        let mut report = self.run_decay().await?;
        // Detection only: candidates are counted, splitting is a later
        // enhancement, so the count has no report field yet.
        let _mitosis_candidates = self.run_mitosis().await?;
        report.contradictions_linked = 0;
        Ok(report)
    }

    /// Decay pass over raw engrams: prune, compress, or promote each one by
    /// its Ebbinghaus retention and the configured thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when any `HelixDB` query fails.
    pub async fn run_decay(&self) -> Result<ConsolidationReport> {
        let request =
            get_engrams_for_decay_check().map_err(|e| MnemosError::Storage(e.to_string()))?;
        let response: Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        let engrams: Vec<EngramCandidate> = response_rows(&response, "engrams");
        let now = chrono::Utc::now().timestamp() as f64;

        let mut report = ConsolidationReport::default();
        for engram in &engrams {
            let days = (now - parse_timestamp_rfc3339(&engram.timestamp)) / 86400.0;
            let kept = retention(engram.decay_rate, days, engram.activation_count);
            if kept < self.config.prune_retention
                && engram.importance_score < self.config.prune_importance
            {
                let request = prune_engram(node_param(engram.id)?)
                    .map_err(|e| MnemosError::Storage(e.to_string()))?;
                let _: Value = self
                    .storage
                    .client()
                    .query(request)
                    .send()
                    .await
                    .map_err(|e| MnemosError::Storage(e.to_string()))?;
                report.pruned += 1;
            } else if kept < self.config.compress_retention
                && engram.importance_score < self.config.compress_importance
            {
                let request = compress_engram(node_param(engram.id)?)
                    .map_err(|e| MnemosError::Storage(e.to_string()))?;
                let _: Value = self
                    .storage
                    .client()
                    .query(request)
                    .send()
                    .await
                    .map_err(|e| MnemosError::Storage(e.to_string()))?;
                report.compressed += 1;
            } else if kept > self.config.promote_retention
                && engram.activation_count > self.config.promote_min_activations
            {
                let request = promote_to_semantic(node_param(engram.id)?)
                    .map_err(|e| MnemosError::Storage(e.to_string()))?;
                let _: Value = self
                    .storage
                    .client()
                    .query(request)
                    .send()
                    .await
                    .map_err(|e| MnemosError::Storage(e.to_string()))?;
                report.promoted += 1;
            }
        }
        Ok(report)
    }

    /// Detect mitosis candidates: concepts with `source_count >=
    /// config.mitosis_min_engrams`.
    ///
    /// Detection only; splitting a concept is a later enhancement. Returns
    /// the number of candidate concepts.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when the `HelixDB` query fails.
    pub async fn run_mitosis(&self) -> Result<u64> {
        let request = get_mitosis_candidates(self.config.mitosis_min_engrams)
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        let response: Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        Ok(response_count(&response, "concepts"))
    }
}

/// Convert a u64 node id to the i64 query param the `#[query]` macro
/// accepts (the macro rejects `u64`; `HelixDB` ids always fit in `i64`).
fn node_param(id: u64) -> Result<i64> {
    i64::try_from(id).map_err(|_| MnemosError::Storage(format!("node id out of range: {id}")))
}

/// Deserialize the row array under `key` (tolerates a single object, missing,
/// or null bindings by yielding zero or one rows; unparseable rows are
/// skipped).
fn response_rows<T>(response: &Value, key: &str) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    match response.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| serde_json::from_value(item.clone()).ok())
            .collect(),
        Some(item) if item.is_object() => serde_json::from_value(item.clone())
            .ok()
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

/// Count the rows under `key` (array length; a single object counts as one;
/// missing or null counts as zero).
fn response_count(response: &Value, key: &str) -> u64 {
    match response.get(key) {
        Some(Value::Array(items)) => u64::try_from(items.len()).unwrap_or(0),
        Some(item) if item.is_object() => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_active_retention_near_one() {
        let kept = retention(0.05, 0.0, 10);
        assert!((kept - 1.0).abs() < 1e-9, "fresh engram kept={kept}");
    }

    #[test]
    fn ancient_inactive_retention_near_zero() {
        let kept = retention(0.1, 3650.0, 0);
        assert!(kept < 1e-6, "ancient engram kept={kept}");
    }

    #[test]
    fn activation_slows_decay() {
        let idle = retention(0.1, 30.0, 0);
        let active = retention(0.1, 30.0, 100);
        assert!(
            active > idle,
            "active kept={active} should exceed idle kept={idle}"
        );
    }

    fn live_pipeline() -> ConsolidationPipeline {
        let storage =
            Storage::new("http://localhost:6969", "mnemos").expect("storage builds without I/O");
        ConsolidationPipeline::new(storage, ConsolidationConfig::default())
    }

    #[tokio::test]
    #[ignore = "requires live HelixDB at http://localhost:6969"]
    async fn live_run_decay_reports_counts() {
        let report = live_pipeline()
            .run_decay()
            .await
            .expect("decay against live DB");
        let _ = report;
    }

    #[tokio::test]
    #[ignore = "requires live HelixDB at http://localhost:6969"]
    async fn live_run_mitosis_returns_count() {
        let count = live_pipeline()
            .run_mitosis()
            .await
            .expect("mitosis against live DB");
        let _ = count;
    }

    #[tokio::test]
    #[ignore = "requires live HelixDB at http://localhost:6969"]
    async fn live_consolidate_merges_report() {
        let report = live_pipeline()
            .consolidate()
            .await
            .expect("consolidate against live DB");
        assert_eq!(report.contradictions_linked, 0);
    }
}
