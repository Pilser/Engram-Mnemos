//! mnemos-stimulation: spreading-activation engine.
//!
//! Pure-math core (seed activation, edge transfer, decay, surfacing,
//! recency) plus a thin HelixDB neighbor fetch. See
//! `__reference/AGI-Memory-Research/Explanation-docs/07-Stimulation-Layer.md`.

use std::collections::HashMap;

use helix_db::dsl::prelude::*;
use mnemos_core::{MnemosError, StimulationConfig};
use mnemos_edge_weights::EdgeWeights;
use mnemos_storage::Storage;

/// ONE generic neighbor-fetch read: start node → outgoing edges filtered
/// by a runtime edge label → target node ids.
///
/// DSL methods used (all grep-verified in
/// `helix-db-3.0.0/src/dsl.rs` → `helix-ast`): `read_batch`, `g`, `n`,
/// `out_e`, `where_`, `out_n`, `id`, `var_as`, `returning`, plus
/// `NodeRef::param`, `Predicate::eq`, and `#[query]` from
/// `helix_db::dsl::prelude`.
///
/// The node id binds via `NodeRef::param("node_id")` (runtime `i64`
/// parameter). The edge label cannot go through `out()` — its argument is
/// `Option<impl Into<String>>`, a concrete string baked into the AST — so
/// the label stays a runtime `String` parameter by filtering traversed
/// edges with `Predicate::eq("$label", edge_label)` (`$label` is the
/// canonical label predicate; `e_with_label` is defined the same way).
#[query]
fn neighbors_query(node_id: i64, edge_label: String) {
    let _ = &node_id;
    read_batch()
        .var_as(
            "neighbors",
            g().n(NodeRef::param("node_id"))
                .out_e(None::<String>)
                .where_(Predicate::eq("$label", edge_label))
                .out_n()
                .id(),
        )
        .returning(["neighbors"])
}

/// Spreading-activation engine: config + learnable edge weights.
pub struct StimulationEngine {
    config: StimulationConfig,
    weights: EdgeWeights,
}

impl StimulationEngine {
    /// Build from stimulation config and edge weights.
    #[must_use]
    pub fn new(config: StimulationConfig, weights: EdgeWeights) -> Self {
        Self { config, weights }
    }

    /// Borrow the stimulation config.
    #[must_use]
    pub fn config(&self) -> &StimulationConfig {
        &self.config
    }

    /// Borrow the learnable edge weights.
    #[must_use]
    pub fn weights(&self) -> &EdgeWeights {
        &self.weights
    }

    /// Mutably borrow the learnable edge weights (e.g. for Adam updates).
    pub fn weights_mut(&mut self) -> &mut EdgeWeights {
        &mut self.weights
    }

    /// Seed activation: `semantic_sim * recency * (1 + |emotional_charge|)`.
    ///
    /// Direct seed — no `HasVector` bridge (vectors live on the node).
    #[must_use]
    pub fn initial_activation(
        &self,
        semantic_sim: f64,
        recency: f64,
        emotional_charge: f64,
    ) -> f64 {
        semantic_sim * recency * (1.0 + emotional_charge.abs())
    }

    /// Activation transferred across one edge: `activation * weight(idx)`.
    ///
    /// Negative weights (e.g. `contradicts`) suppress the neighbor.
    #[must_use]
    pub fn transfer(&self, edge_idx: usize, activation: f64) -> f64 {
        activation * self.weights.weight(edge_idx)
    }

    /// Per-timestep energy decay: `activation * gamma`.
    #[must_use]
    pub fn apply_decay(activation: f64, gamma: f64) -> f64 {
        activation * gamma
    }

    /// Ids whose activation is strictly above `tau`, sorted ascending.
    #[must_use]
    pub fn surfaced(activations: &HashMap<u64, f64>, tau: f64) -> Vec<u64> {
        let mut out: Vec<u64> = activations
            .iter()
            .filter(|(_, a)| **a > tau)
            .map(|(id, _)| *id)
            .collect();
        out.sort_unstable();
        out
    }

    /// Recency weight: `exp(-0.01 * days_since).max(0.01)`.
    ///
    /// `days_since = (now_unix_secs - parsed_timestamp) / 86400`.
    #[must_use]
    pub fn compute_recency_weight(timestamp_rfc3339: &str, now_unix_secs: f64) -> f64 {
        let ts = mnemos_core::parse_timestamp_rfc3339(timestamp_rfc3339);
        let days = (now_unix_secs - ts) / 86_400.0;
        (-0.01 * days).exp().max(0.01)
    }

    /// Fetch outgoing neighbor ids of `node_id` across `edge_label`.
    ///
    /// Thin wrapper over the single generic `neighbors_query` read; the
    /// edge label travels as a runtime `String` parameter. Accepts raw
    /// numeric ids as well as `$id`/`id` objects per element.
    pub async fn neighbors(
        &self,
        storage: &Storage,
        node_id: u64,
        edge_label: &str,
    ) -> mnemos_core::Result<Vec<u64>> {
        // `Client::query` signature (helix-db 3.0.0 `lib.rs`):
        // `pub fn query<R: Deserialize>(&self, request: QueryRequest)
        //  -> QueryExecutionRequest<'_, 'static, R>`, then `.send().await`.
        let request = neighbors_query(node_id as i64, edge_label.to_string())
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        let response: serde_json::Value = storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        Ok(parse_neighbor_ids(&response))
    }
}

/// Extract neighbor ids from a `neighbors_query` response envelope.
fn parse_neighbor_ids(response: &serde_json::Value) -> Vec<u64> {
    let items = match response.get("neighbors") {
        Some(v) => v,
        None => return Vec::new(),
    };
    let list: Vec<&serde_json::Value> = match items {
        serde_json::Value::Array(arr) => arr.iter().collect(),
        single => vec![single],
    };
    list.into_iter().filter_map(parse_id_value).collect()
}

/// One id from a raw number, numeric string, or `$id`/`id` object.
fn parse_id_value(v: &serde_json::Value) -> Option<u64> {
    match v {
        serde_json::Value::Number(n) => n
            .as_u64()
            .or_else(|| {
                n.as_i64()
                    .and_then(|i| u64::try_from(i).ok())
            })
            .or_else(|| {
                n.as_f64()
                    .filter(|f| *f >= 0.0)
                    .map(|f| f as u64)
            }),
        serde_json::Value::String(s) => s.parse().ok(),
        serde_json::Value::Object(m) => m
            .get("$id")
            .or_else(|| m.get("id"))
            .and_then(parse_id_value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_edge_weights::{
        IDX_ABSTRACTS_TO, IDX_CONTRADICTS, IDX_RECALLS, IDX_RECURRENT,
    };

    fn engine() -> StimulationEngine {
        StimulationEngine::new(
            StimulationConfig::default(),
            EdgeWeights::defaults(),
        )
    }

    #[test]
    fn seed_math_matches_spec() {
        let e = engine();
        let got = e.initial_activation(0.8, 0.9, 0.5);
        assert!((got - 0.8 * 0.9 * 1.5).abs() < 1e-12);
    }

    #[test]
    fn seed_emotional_sign_is_symmetric() {
        let e = engine();
        assert_eq!(
            e.initial_activation(0.8, 0.9, -0.5),
            e.initial_activation(0.8, 0.9, 0.5)
        );
    }

    #[test]
    fn transfer_applies_edge_weight() {
        let e = engine();
        assert!((e.transfer(IDX_RECALLS, 1.0) - 0.70).abs() < 1e-12);
        assert!((e.transfer(IDX_ABSTRACTS_TO, 2.0) - 1.0).abs() < 1e-12);
        assert!((e.transfer(IDX_RECURRENT, 1.0) - 0.35).abs() < 1e-12);
    }

    #[test]
    fn transfer_contradicts_is_negative() {
        let e = engine();
        let got = e.transfer(IDX_CONTRADICTS, 1.0);
        assert!(got < 0.0, "contradicts must suppress: {got}");
        assert!((got - -0.40).abs() < 1e-12);
    }

    #[test]
    fn decay_multiplies_by_gamma() {
        assert!((StimulationEngine::apply_decay(1.0, 0.75) - 0.75).abs() < 1e-12);
        assert_eq!(StimulationEngine::apply_decay(0.0, 0.75), 0.0);
    }

    #[test]
    fn threshold_filter_is_strict() {
        let activations: HashMap<u64, f64> =
            [(1, 0.5), (2, 0.15), (3, 0.149), (4, 0.0)].into_iter().collect();
        // 0.15 is NOT above tau=0.15; only id 1 surfaces.
        assert_eq!(StimulationEngine::surfaced(&activations, 0.15), vec![1]);
        assert_eq!(
            StimulationEngine::surfaced(&activations, 0.1),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn recency_of_now_is_one() {
        let ts = "2026-06-01T00:00:00Z";
        let now = mnemos_core::parse_timestamp_rfc3339(ts);
        let got = StimulationEngine::compute_recency_weight(ts, now);
        assert!((got - 1.0).abs() < 1e-9, "recency of now ≈ 1.0: {got}");
    }

    #[test]
    fn recency_decays_and_floors() {
        let ts = "2026-05-02T00:00:00Z"; // 30 days before 2026-06-01
        let now = mnemos_core::parse_timestamp_rfc3339("2026-06-01T00:00:00Z");
        let got = StimulationEngine::compute_recency_weight(ts, now);
        assert!((got - (-0.3f64).exp()).abs() < 1e-9, "30d decay: {got}");

        // Very old timestamps floor at 0.01.
        let ancient = StimulationEngine::compute_recency_weight(
            "1970-01-01T00:00:00Z",
            now,
        );
        assert_eq!(ancient, 0.01);
        // Unparseable timestamps also floor (parse → 0.0).
        assert_eq!(
            StimulationEngine::compute_recency_weight("not-a-timestamp", now),
            0.01
        );
    }

    #[test]
    fn neighbors_query_builds_typed_request() {
        let req = neighbors_query(7, "Recalls".to_string()).expect("builds");
        assert_eq!(req.query_name(), Some("neighbors_query"));
        assert!(matches!(
            req.request_type(),
            helix_db::QueryRequestType::Read
        ));
        let params = req.parameters().expect("parameters present");
        assert!(matches!(
            params.get("node_id"),
            Some(helix_db::QueryValue::I64(7))
        ));
        assert!(matches!(
            params.get("edge_label"),
            Some(helix_db::QueryValue::String(s)) if s == "Recalls"
        ));
    }

    #[test]
    fn parse_handles_id_shapes() {
        let v: serde_json::Value = serde_json::json!({
            "neighbors": [3, {"$id": 5}, {"id": 9}, "11", 12.0, {"$id": -1}, null]
        });
        assert_eq!(parse_neighbor_ids(&v), vec![3, 5, 9, 11, 12]);
        let missing: serde_json::Value = serde_json::json!({"other": []});
        assert!(parse_neighbor_ids(&missing).is_empty());
    }

    #[tokio::test]
    #[ignore = "requires live HelixDB at http://localhost:6969"]
    async fn live_neighbors_fetch() {
        let storage =
            Storage::new("http://localhost:6969", "mnemos").expect("client builds");
        let e = engine();
        let ids = e
            .neighbors(&storage, 1, "Recalls")
            .await
            .expect("live query works");
        let _ = ids;
    }
}
