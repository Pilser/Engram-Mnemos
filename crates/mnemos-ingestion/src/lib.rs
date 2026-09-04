#![recursion_limit = "256"]
//! mnemos-ingestion: full ingestion pipeline.
//!
//! `IngestionPipeline` runs emotional tagging → importance scoring →
//! embedding → [`Storage`] writes: one `Engram` node (embedding stored
//! directly on the node) plus one `Concept` node per extracted concept,
//! linked by `Recalls` and `AbstractsTo` edges.
//!
//! All `HelixDB` queries are built with the `#[query]` macro from
//! `helix_db::dsl::prelude::*` and executed via
//! `storage.client().query(request).send().await`. Responses are parsed
//! as `serde_json::Value` with the node `$id` extracted by pointer;
//! failures map to [`mnemos_core::MnemosError::Storage`].
//!
//! Grounded in `helix-db 3.0.0` / `helix-ast 0.1.0` SDK surface
//! (`g().add_n`, `g().n(...).add_e`, `NodeRef::param`, `write_batch`):
//! edge labels are static AST strings, so each label gets its own
//! `#[query]` fn and the private `connect_edge` helper dispatches on the
//! label. `#[query]` params support `i64` but not `u64`, hence the
//! `u64` → `i64` conversion at the helper boundary.

use chrono::Utc;
use helix_db::dsl::prelude::*;
use mnemos_contradiction::ContradictionDetector;
use mnemos_core::{EngramId, EngramType, ExtractedConcept, MnemosError};
use mnemos_embedding_trait::EmbeddingProvider;
use mnemos_ml_trait::{ConceptExtractor, EmotionalTagger, ImportanceScorer};
use mnemos_storage::Storage;

/// Default Ebbinghaus decay rate for freshly ingested engrams.
pub const DEFAULT_DECAY_RATE: f64 = 0.05;

/// Compression level stored for raw (uncompressed) engrams.
pub const COMPRESSION_LEVEL_RAW: i64 = 0;

/// Initial activation count for a new engram.
pub const INITIAL_ACTIVATION_COUNT: i64 = 0;

/// Full ingestion pipeline: ML providers plus the storage backend.
///
/// `Storage` is held by value (it is not `Clone`). An optional
/// [`ContradictionDetector`] (see
/// [`Self::with_contradiction_detector`]) scans each linked concept
/// best-effort after ingest.
pub struct IngestionPipeline {
    tagger: Box<dyn EmotionalTagger>,
    scorer: Box<dyn ImportanceScorer>,
    extractor: Box<dyn ConceptExtractor>,
    embedder: Box<dyn EmbeddingProvider>,
    storage: Storage,
    detector: Option<ContradictionDetector>,
}

impl IngestionPipeline {
    /// Assemble the pipeline from its providers and storage backend.
    ///
    /// No contradiction detector is attached; use
    /// [`Self::with_contradiction_detector`] to add one.
    #[must_use]
    pub fn new(
        tagger: Box<dyn EmotionalTagger>,
        scorer: Box<dyn ImportanceScorer>,
        extractor: Box<dyn ConceptExtractor>,
        embedder: Box<dyn EmbeddingProvider>,
        storage: Storage,
    ) -> Self {
        Self {
            tagger,
            scorer,
            extractor,
            embedder,
            storage,
            detector: None,
        }
    }

    /// Attach a contradiction detector; its per-concept scan runs
    /// best-effort after linking (failures are recorded via telemetry and
    /// never fail ingest).
    #[must_use]
    pub fn with_contradiction_detector(mut self, detector: ContradictionDetector) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Ingest one memory.
    ///
    /// Steps: tag → score → embed → `create_full_engram` (Engram node with
    /// the embedding vec directly on the node, RFC 3339 timestamp,
    /// `activation_count` 0, `contradiction_flag` false, `decay_rate`
    /// [`DEFAULT_DECAY_RATE`]) → parse `$id` → extract concepts → per
    /// concept: `get_concept_by_name` (reuse on name match) else
    /// `create_concept` → `Recalls` edge + `AbstractsTo` edge →
    /// `source_count + 1` → best-effort contradiction scan.
    ///
    /// Equivalent to `ingest_with_importance(text, engram_type, None)`.
    ///
    /// # Errors
    ///
    /// Returns [`mnemos_core::MnemosError`] when any ML provider fails or
    /// when a `HelixDB` request fails or its response lacks a node `$id`.
    pub async fn ingest(
        &self,
        text: &str,
        engram_type: EngramType,
    ) -> mnemos_core::Result<EngramId> {
        self.ingest_with_importance(text, engram_type, None).await
    }

    /// Ingest one memory with an optional importance override.
    ///
    /// `Some(v)` clamps `v` into `[0, 1]` and skips the scorer LLM call;
    /// `None` scores via [`ImportanceScorer`] (the Tool-Call-Protocol
    /// `store` `auto` path; explicit `low`/`medium`/`high` map to
    /// `0.2`/`0.5`/`0.85` at the caller and arrive here as `Some`).
    /// The outcome (ok + engram id in detail, or err) is recorded via
    /// `mnemos_telemetry::global` (`"mnemos-ingestion"` / `"ingest"`).
    ///
    /// # Errors
    ///
    /// Returns [`mnemos_core::MnemosError`] when any ML provider fails or
    /// when a `HelixDB` request fails or its response lacks a node `$id`.
    pub async fn ingest_with_importance(
        &self,
        text: &str,
        engram_type: EngramType,
        importance: Option<f64>,
    ) -> mnemos_core::Result<EngramId> {
        let outcome = self.ingest_inner(text, engram_type, importance).await;
        match &outcome {
            Ok(id) => mnemos_telemetry::global().record(
                "mnemos-ingestion",
                "ingest",
                true,
                &format!("engram_id={id}"),
            ),
            Err(error) => mnemos_telemetry::global().record(
                "mnemos-ingestion",
                "ingest",
                false,
                &error.to_string(),
            ),
        }
        outcome
    }

    /// The ingest workhorse shared by [`Self::ingest`] and
    /// [`Self::ingest_with_importance`] (telemetry wraps at that layer).
    async fn ingest_inner(
        &self,
        text: &str,
        engram_type: EngramType,
        importance: Option<f64>,
    ) -> mnemos_core::Result<EngramId> {
        let emotional_charge = self.tagger.tag(text).await?;
        let importance_score = match importance {
            Some(override_score) => clamp_importance(override_score),
            None => self.scorer.score(text).await?,
        };
        let embedding = self.embedder.embed(text).await?;
        let timestamp = Utc::now().to_rfc3339();

        let request = create_full_engram(
            text.to_string(),
            emotional_charge,
            importance_score,
            embedding,
            timestamp,
            engram_type.as_str().to_string(),
            INITIAL_ACTIVATION_COUNT,
            false,
            DEFAULT_DECAY_RATE,
            COMPRESSION_LEVEL_RAW,
        )
        .map_err(storage_error)?;
        let response: serde_json::Value =
            self.storage.client().query(request).send().await.map_err(storage_error)?;
        let engram_id = parse_node_id(&response)?;

        let concepts = self.extractor.extract(text).await?;
        for concept in &concepts {
            let (concept_id, known_source_count) =
                self.get_or_create_concept(concept).await?;
            connect_edge(&self.storage, engram_id, concept_id, "Recalls").await?;
            connect_edge(&self.storage, engram_id, concept_id, "AbstractsTo").await?;
            self.bump_source_count(concept_id, known_source_count).await?;
            self.run_contradiction_hook(concept_id).await;
        }

        Ok(engram_id)
    }

    /// Reuse the `Concept` node with this name when it exists, else create
    /// it (`source_count` starts at `0`; the increment happens in
    /// [`Self::bump_source_count`] after both edges link).
    ///
    /// Returns the concept id plus the `source_count` already observed in
    /// the get/create response when available (`None` → fetch in the bump).
    async fn get_or_create_concept(
        &self,
        concept: &ExtractedConcept,
    ) -> mnemos_core::Result<(EngramId, Option<i64>)> {
        let request = get_concept_by_name(concept.name.clone()).map_err(storage_error)?;
        let response: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(storage_error)?;
        if let Some(found) = parse_concept_lookup(&response) {
            return Ok(found);
        }

        let formation_date = Utc::now().to_rfc3339();
        let request = create_concept(
            concept.name.clone(),
            concept.confidence,
            formation_date,
            0,
        )
        .map_err(storage_error)?;
        let response: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(storage_error)?;
        let concept_id = parse_node_id(&response)?;
        Ok((concept_id, parse_source_count(&response)))
    }

    /// Read-modify-write `source_count + 1` after both edges link.
    /// Prefers the count already seen in the get/create response; fetches
    /// it (defaulting a missing property to `0`) otherwise.
    async fn bump_source_count(
        &self,
        concept_id: EngramId,
        known_source_count: Option<i64>,
    ) -> mnemos_core::Result<()> {
        let current = if let Some(count) = known_source_count { count } else {
            let request =
                get_concept_source_count(to_i64(concept_id)?).map_err(storage_error)?;
            let response: serde_json::Value = self
                .storage
                .client()
                .query(request)
                .send()
                .await
                .map_err(storage_error)?;
            parse_source_count(&response).unwrap_or(0)
        };
        let request = set_concept_source_count(to_i64(concept_id)?, current.saturating_add(1))
            .map_err(storage_error)?;
        let _: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(storage_error)?;
        Ok(())
    }

    /// Scan one linked concept for contradictions. Best-effort by design:
    /// detector failures are recorded via telemetry (`"mnemos-ingestion"` /
    /// `"ingest.contradiction_hook"`) and never fail ingest; a missing
    /// detector is a silent no-op.
    async fn run_contradiction_hook(&self, concept_id: EngramId) {
        if let Some(detector) = &self.detector {
            if let Err(error) = detector.scan_concept(&self.storage, concept_id).await {
                mnemos_telemetry::global().record(
                    "mnemos-ingestion",
                    "ingest.contradiction_hook",
                    false,
                    &error.to_string(),
                );
            }
        }
    }
}

/// Create the full `Engram` node: all scalar properties plus the embedding
/// vector stored directly on the node.
#[query]
fn create_full_engram(
    episode_raw: String,
    emotional_charge: f64,
    importance_score: f64,
    embedding: Vec<f32>,
    timestamp: String,
    engram_type: String,
    activation_count: i64,
    contradiction_flag: bool,
    decay_rate: f64,
    compression_level: i64,
) {
    write_batch()
        .var_as(
            "engram",
            g().add_n(
                "Engram",
                vec![
                    ("episode_raw", episode_raw),
                    ("emotional_charge", emotional_charge),
                    ("importance_score", importance_score),
                    ("embedding", embedding),
                    ("timestamp", timestamp),
                    ("engram_type", engram_type),
                    ("activation_count", activation_count),
                    ("contradiction_flag", contradiction_flag),
                    ("decay_rate", decay_rate),
                    ("compression_level", compression_level),
                ],
            ),
        )
        .returning(["engram"])
}

/// Create one `Concept` node for an extracted concept.
///
/// `source_count` starts at `0`; the caller bumps it to `1` after linking
/// both edges (see `IngestionPipeline::bump_source_count`), so fresh and
/// reused concepts share one increment path.
#[query]
fn create_concept(
    name: String,
    confidence: f64,
    formation_date: String,
    source_count: i64,
) {
    write_batch()
        .var_as(
            "concept",
            g().add_n(
                "Concept",
                vec![
                    ("name", name),
                    ("confidence", confidence),
                    ("formation_date", formation_date),
                    ("source_count", source_count),
                ],
            ),
        )
        .returning(["concept"])
}

/// Look up one `Concept` by exact name (dedup read).
///
/// DSL methods grep-verified in `helix-ast 0.1.0` (`traversal.rs`):
/// `n_with_label_where` (label + `SourcePredicate`, a `Predicate` alias)
/// with a `String` runtime value — the same shape as `neighbors_query`'s
/// `Predicate::eq("$label", edge_label)` in `mnemos-stimulation` — then
/// `limit` (`i64: Into<StreamBound>`) and a `value_map` projection.
/// `#[query]` params are `i64`, never `u64`.
#[query]
fn get_concept_by_name(name: String) {
    read_batch()
        .var_as(
            "concept",
            g().n_with_label_where("Concept", SourcePredicate::eq("name", name))
                .limit(1_i64)
                .value_map(Some(vec!["$id", "name", "source_count"])),
        )
        .returning(["concept"])
}

/// Read back one concept's `source_count` (dedup-increment fallback when
/// the get/create response did not already carry it).
#[query]
fn get_concept_source_count(concept_id: i64) {
    // See `get_engram_activation` in mnemos-retrieval: the binding keeps
    // the macro-generated ident used; the value travels as `NodeRef::param`.
    let _ = &concept_id;
    read_batch()
        .var_as(
            "concept",
            g().n(NodeRef::param("concept_id"))
                .value_map(Some(vec!["$id", "source_count"])),
        )
        .returning(["concept"])
}

/// Write back one concept's `source_count` (dedup increment).
#[query]
fn set_concept_source_count(concept_id: i64, source_count: i64) {
    // See `get_concept_source_count` for the binding note.
    let _ = &concept_id;
    write_batch()
        .var_as(
            "updated",
            g().n(NodeRef::param("concept_id"))
                .set_property("source_count", source_count),
        )
        .returning(["updated"])
}

/// Create a `Recalls` edge from an engram to a concept.
///
/// Node refs bind by request-parameter name (`NodeRef::param`); the
/// generated `Expr` bindings are referenced by those names, hence the
/// explicit use below.
#[query]
fn connect_recalls_edge(from_id: i64, to_id: i64) {
    let _ = (&from_id, &to_id);
    write_batch()
        .var_as(
            "edge",
            g().n(NodeRef::param("from_id")).add_e(
                "Recalls",
                NodeRef::param("to_id"),
                Vec::<(String, PropertyInput)>::new(),
            ),
        )
        .returning(["edge"])
}

/// Create an `AbstractsTo` edge from an engram to a concept.
///
/// See [`connect_recalls_edge`] for the parameter-binding note.
#[query]
fn connect_abstracts_to_edge(from_id: i64, to_id: i64) {
    let _ = (&from_id, &to_id);
    write_batch()
        .var_as(
            "edge",
            g().n(NodeRef::param("from_id")).add_e(
                "AbstractsTo",
                NodeRef::param("to_id"),
                Vec::<(String, PropertyInput)>::new(),
            ),
        )
        .returning(["edge"])
}

/// Dispatch an engram → concept edge write by label.
///
/// `add_e` labels are static AST strings, so this plain (non-`#[query]`)
/// helper routes to the matching `#[query]` fn. Only `"Recalls"` and
/// `"AbstractsTo"` are supported.
///
/// # Errors
///
/// Returns [`mnemos_core::MnemosError`] on unknown labels, id overflow,
/// or `HelixDB` request failure.
async fn connect_edge(
    storage: &Storage,
    from_id: u64,
    to_id: u64,
    label: &str,
) -> mnemos_core::Result<()> {
    let request = match label {
        "Recalls" => connect_recalls_edge(to_i64(from_id)?, to_i64(to_id)?),
        "AbstractsTo" => connect_abstracts_to_edge(to_i64(from_id)?, to_i64(to_id)?),
        other => {
            return Err(MnemosError::Internal(format!(
                "unknown edge label: {other}"
            )));
        }
    }
    .map_err(storage_error)?;
    let _: serde_json::Value = storage
        .client()
        .query(request)
        .send()
        .await
        .map_err(storage_error)?;
    Ok(())
}

/// Map any displayable query/client failure into `MnemosError::Storage`.
fn storage_error<E: std::fmt::Display>(error: E) -> MnemosError {
    MnemosError::Storage(error.to_string())
}

/// Convert a node id to the `i64` query param the `#[query]` macro accepts
/// (the macro rejects `u64`; `HelixDB` ids always fit in `i64`).
fn to_i64(id: u64) -> mnemos_core::Result<i64> {
    i64::try_from(id)
        .map_err(|_| MnemosError::Internal(format!("node id overflow: {id}")))
}

/// Clamp an explicit importance override into `[0, 1]`; `NaN` maps to
/// `0.0` (`None` never reaches this helper — it takes the scorer path).
fn clamp_importance(override_score: f64) -> f64 {
    if override_score.is_nan() {
        0.0
    } else {
        override_score.clamp(0.0, 1.0)
    }
}

/// Coerce a JSON id (`u64`, `i64`, or numeric string) to `u64`.
fn json_id(value: &serde_json::Value) -> Option<u64> {
    if let Some(id) = value.as_u64() {
        return Some(id);
    }
    if let Some(id) = value.as_i64() {
        return u64::try_from(id).ok();
    }
    if let Some(text) = value.as_str() {
        return text.parse::<u64>().ok();
    }
    None
}

/// Extract a created node's id from a `HelixDB` JSON response.
///
/// Handles `{"$id": …}` directly as well as batched shapes like
/// `{"engram": [{"$id": …}]}`, falling back to `"id"` keys.
fn parse_node_id(response: &serde_json::Value) -> mnemos_core::Result<EngramId> {
    if let Some(id) = response.get("$id").and_then(json_id) {
        return Ok(id);
    }
    if let Some(id) = response.get("id").and_then(json_id) {
        return Ok(id);
    }
    if let Some(object) = response.as_object() {
        for value in object.values() {
            if let Some(id) = value.get("$id").and_then(json_id) {
                return Ok(id);
            }
            if let Some(id) = value.get("id").and_then(json_id) {
                return Ok(id);
            }
            if let Some(items) = value.as_array() {
                for item in items {
                    if let Some(id) = item.get("$id").and_then(json_id) {
                        return Ok(id);
                    }
                    if let Some(id) = item.get("id").and_then(json_id) {
                        return Ok(id);
                    }
                }
            }
        }
    }
    if let Some(items) = response.as_array() {
        for item in items {
            if let Some(id) = item.get("$id").and_then(json_id) {
                return Ok(id);
            }
        }
    }
    Err(MnemosError::Storage(format!(
        "response missing node $id: {response}"
    )))
}

/// Coerce a JSON `source_count` (`i64`, `u64`, or finite `f64`) to `i64`.
fn json_source_count(value: &serde_json::Value) -> Option<i64> {
    if let Some(count) = value.as_i64() {
        return Some(count);
    }
    if let Some(count) = value.as_u64() {
        return i64::try_from(count).ok();
    }
    if let Some(count) = value.as_f64() {
        if count.is_finite() {
            return Some(count as i64);
        }
    }
    None
}

/// Extract the first `Concept` row (`$id` + optional `source_count`) from a
/// `get_concept_by_name` response; `None` means "no such concept".
///
/// Handles `{"concept": [{...}]}`, a bare `{"concept": {...}}` object, and
/// empty/missing/null bindings (→ `None`).
fn parse_concept_lookup(response: &serde_json::Value) -> Option<(EngramId, Option<i64>)> {
    let row = match response.get("concept") {
        Some(serde_json::Value::Array(items)) => items.first()?,
        Some(row) if row.is_object() => row,
        _ => return None,
    };
    let id = row
        .get("$id")
        .and_then(json_id)
        .or_else(|| row.get("id").and_then(json_id))?;
    Some((id, row.get("source_count").and_then(json_source_count)))
}

/// Extract `source_count` from a `get_concept_source_count` (or fresh
/// `create_concept`) response; `None` when the property is absent.
fn parse_source_count(response: &serde_json::Value) -> Option<i64> {
    match response.get("concept") {
        Some(serde_json::Value::Array(items)) => items
            .first()
            .and_then(|row| row.get("source_count"))
            .and_then(json_source_count),
        Some(row) if row.is_object() => {
            row.get("source_count").and_then(json_source_count)
        }
        _ => response
            .get("source_count")
            .and_then(json_source_count),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use mnemos_core::ExtractedConcept;
    use serde_json::json;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn parse_node_id_from_batched_response() {
        let response = json!({"engram": [{"$id": 42, "episode_raw": "hi"}]});
        assert_eq!(parse_node_id(&response).unwrap(), 42);
    }

    #[test]
    fn parse_node_id_direct() {
        assert_eq!(parse_node_id(&json!({"$id": 7})).unwrap(), 7);
        assert_eq!(parse_node_id(&json!({"id": 9})).unwrap(), 9);
        assert_eq!(parse_node_id(&json!([{"$id": 11}])).unwrap(), 11);
    }

    #[test]
    fn parse_node_id_missing_is_storage_error() {
        let err = parse_node_id(&json!({"engram": []})).unwrap_err();
        assert!(matches!(err, MnemosError::Storage(_)));
    }

    #[test]
    fn query_builders_produce_write_requests() {
        let request = create_full_engram(
            "the sky is blue".to_string(),
            0.25,
            0.8,
            vec![0.1, 0.2],
            "2026-09-03T00:00:00Z".to_string(),
            EngramType::Episodic.as_str().to_string(),
            INITIAL_ACTIVATION_COUNT,
            false,
            DEFAULT_DECAY_RATE,
            COMPRESSION_LEVEL_RAW,
        )
        .expect("engram query builds");
        assert_eq!(request.query_name(), Some("create_full_engram"));

        let request = create_concept(
            "sky".to_string(),
            0.9,
            "2026-09-03T00:00:00Z".to_string(),
            0,
        )
        .expect("concept query builds");
        assert_eq!(request.query_name(), Some("create_concept"));

        let request = connect_recalls_edge(1, 2).expect("recalls edge query builds");
        assert_eq!(request.query_name(), Some("connect_recalls_edge"));

        let request =
            connect_abstracts_to_edge(1, 2).expect("abstracts-to edge query builds");
        assert_eq!(request.query_name(), Some("connect_abstracts_to_edge"));
    }

    #[test]
    fn concept_dedup_queries_build_typed_requests() {
        // No live DB: proves the #[query] wiring (params, kind, name).
        let request = get_concept_by_name("sky".to_string()).expect("lookup builds");
        assert_eq!(request.query_name(), Some("get_concept_by_name"));
        assert!(matches!(
            request.request_type(),
            helix_db::QueryRequestType::Read
        ));
        let params = request.parameters().expect("parameters present");
        assert!(matches!(
            params.get("name"),
            Some(helix_db::QueryValue::String(s)) if s == "sky"
        ));

        let request = get_concept_source_count(7).expect("count read builds");
        assert_eq!(request.query_name(), Some("get_concept_source_count"));
        assert!(matches!(
            request.request_type(),
            helix_db::QueryRequestType::Read
        ));
        let params = request.parameters().expect("parameters present");
        assert!(matches!(
            params.get("concept_id"),
            Some(helix_db::QueryValue::I64(7))
        ));

        let request = set_concept_source_count(7, 3).expect("count write builds");
        assert_eq!(request.query_name(), Some("set_concept_source_count"));
        let params = request.parameters().expect("parameters present");
        assert!(matches!(
            params.get("concept_id"),
            Some(helix_db::QueryValue::I64(7))
        ));
        assert!(matches!(
            params.get("source_count"),
            Some(helix_db::QueryValue::I64(3))
        ));
    }

    #[test]
    fn clamp_importance_bounds_override() {
        assert_eq!(clamp_importance(0.9), 0.9);
        assert_eq!(clamp_importance(0.0), 0.0);
        assert_eq!(clamp_importance(1.0), 1.0);
        assert_eq!(clamp_importance(-0.5), 0.0);
        assert_eq!(clamp_importance(1.7), 1.0);
        assert_eq!(clamp_importance(f64::NAN), 0.0);
        assert_eq!(clamp_importance(f64::INFINITY), 1.0);
        assert_eq!(clamp_importance(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn parse_concept_lookup_shapes() {
        let found = json!({"concept": [{"$id": 5, "name": "sky", "source_count": 3}]});
        assert_eq!(parse_concept_lookup(&found), Some((5, Some(3))));

        let no_count = json!({"concept": [{"$id": 6, "name": "sky"}]});
        assert_eq!(parse_concept_lookup(&no_count), Some((6, None)));

        let single = json!({"concept": {"$id": 7, "source_count": 1}});
        assert_eq!(parse_concept_lookup(&single), Some((7, Some(1))));

        assert_eq!(parse_concept_lookup(&json!({"concept": []})), None);
        assert_eq!(parse_concept_lookup(&json!({"concept": null})), None);
        assert_eq!(parse_concept_lookup(&json!({})), None);
        assert_eq!(parse_concept_lookup(&json!({"concept": [{}]})), None);
    }

    #[test]
    fn parse_source_count_shapes() {
        assert_eq!(
            parse_source_count(&json!({"concept": [{"source_count": 4}]})),
            Some(4)
        );
        assert_eq!(
            parse_source_count(&json!({"concept": {"source_count": 2}})),
            Some(2)
        );
        assert_eq!(parse_source_count(&json!({"concept": [{}]})), None);
        assert_eq!(parse_source_count(&json!({"concept": []})), None);
        assert_eq!(parse_source_count(&json!({})), None);
    }

    /// Compile-time pin of the FROZEN `mnemos-contradiction` contract the
    /// hook codes against (parallel crate): `new(Box<dyn LlmProvider>)`
    /// plus `scan_concept(&Storage, u64)`. Never runs — merely type-checks
    /// the hook call site, so a contract drift fails here first.
    fn _hook_call<'a>(
        detector: &'a ContradictionDetector,
        storage: &'a Storage,
        concept_id: EngramId,
    ) -> impl std::future::Future<Output = mnemos_core::Result<Vec<(EngramId, EngramId)>>> + 'a
    {
        detector.scan_concept(storage, concept_id)
    }

    #[test]
    fn contradiction_detector_matches_frozen_contract() {
        let _ = _hook_call;
    }

    /// The detector builder takes a detector by value and returns the pipeline.
    #[test]
    fn detector_builder_signature() {
        fn _builder(
            pipeline: IngestionPipeline,
            detector: ContradictionDetector,
        ) -> IngestionPipeline {
            pipeline.with_contradiction_detector(detector)
        }
        let _ = _builder;
    }

    #[tokio::test]
    async fn connect_edge_rejects_unknown_label_without_io() {
        // `Storage::new` performs no network I/O; the label check fails first.
        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let err = connect_edge(&storage, 1, 2, "Bogus").await.unwrap_err();
        assert!(matches!(err, MnemosError::Internal(_)));
    }

    struct MockTagger;
    struct MockScorer;
    struct MockExtractor;
    struct MockEmbedder;

    #[async_trait]
    impl EmotionalTagger for MockTagger {
        async fn tag(&self, _text: &str) -> mnemos_core::Result<f64> {
            Ok(0.25)
        }
    }

    #[async_trait]
    impl ImportanceScorer for MockScorer {
        async fn score(&self, _text: &str) -> mnemos_core::Result<f64> {
            Ok(0.8)
        }
    }

    /// Panics if the scorer LLM is consulted: proves an explicit importance
    /// override skips the scorer call entirely.
    struct PanickingScorer;

    #[async_trait]
    impl ImportanceScorer for PanickingScorer {
        async fn score(&self, _text: &str) -> mnemos_core::Result<f64> {
            panic!("scorer must be skipped when importance is overridden");
        }
    }

    /// Counts scorer calls: proves the `None` (auto) path still consults
    /// the scorer before any storage I/O.
    struct CountingScorer {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ImportanceScorer for CountingScorer {
        async fn score(&self, _text: &str) -> mnemos_core::Result<f64> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(0.8)
        }
    }

    #[async_trait]
    impl ConceptExtractor for MockExtractor {
        async fn extract(&self, _text: &str) -> mnemos_core::Result<Vec<ExtractedConcept>> {
            Ok(vec![ExtractedConcept {
                name: "sky".to_string(),
                confidence: 0.9,
            }])
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbedder {
        async fn embed(&self, _text: &str) -> mnemos_core::Result<Vec<f32>> {
            Ok(vec![0.1; 8])
        }
    }

    #[tokio::test]
    async fn ingest_with_importance_skips_scorer_when_overridden() {
        // No live DB: the panicking scorer would fire before any I/O, so
        // reaching the (failing) storage write proves `Some(0.9)` was used
        // without consulting the scorer.
        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let pipeline = IngestionPipeline::new(
            Box::new(MockTagger),
            Box::new(PanickingScorer),
            Box::new(MockExtractor),
            Box::new(MockEmbedder),
            storage,
        );
        let err = pipeline
            .ingest_with_importance("The sky is blue.", EngramType::Episodic, Some(0.9))
            .await
            .unwrap_err();
        assert!(matches!(err, MnemosError::Storage(_)));
    }

    #[tokio::test]
    async fn ingest_without_override_consults_scorer() {
        // No live DB: the scorer runs before the (failing) storage write,
        // so exactly one call plus a storage error proves the auto path.
        let calls = Arc::new(AtomicUsize::new(0));
        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let pipeline = IngestionPipeline::new(
            Box::new(MockTagger),
            Box::new(CountingScorer {
                calls: Arc::clone(&calls),
            }),
            Box::new(MockExtractor),
            Box::new(MockEmbedder),
            storage,
        );
        let err = pipeline
            .ingest("The sky is blue.", EngramType::Episodic)
            .await
            .unwrap_err();
        assert!(matches!(err, MnemosError::Storage(_)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn contradiction_hook_without_detector_is_noop() {
        // No detector attached and no live DB needed: the hook returns
        // without touching the network (a network attempt would fail).
        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let pipeline = IngestionPipeline::new(
            Box::new(MockTagger),
            Box::new(MockScorer),
            Box::new(MockExtractor),
            Box::new(MockEmbedder),
            storage,
        );
        pipeline.run_contradiction_hook(1).await;
    }

    #[tokio::test]
    #[ignore = "needs live HelixDB"]
    async fn pipeline_ingest_roundtrip() {
        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let pipeline = IngestionPipeline::new(
            Box::new(MockTagger),
            Box::new(MockScorer),
            Box::new(MockExtractor),
            Box::new(MockEmbedder),
            storage,
        );
        let _id = pipeline
            .ingest("The sky is blue.", EngramType::Episodic)
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "needs live HelixDB"]
    async fn pipeline_dedup_reuses_concept_and_bumps_source_count() {
        // `MockExtractor` always yields the single concept "sky": two
        // ingests must reuse one Concept node and bump it twice. The
        // before/after delta keeps the test robust to re-runs against a
        // non-empty DB.
        async fn lookup(storage: &Storage) -> Option<(EngramId, Option<i64>)> {
            let request = get_concept_by_name("sky".to_string()).unwrap();
            let response: serde_json::Value =
                storage.client().query(request).send().await.unwrap();
            parse_concept_lookup(&response)
        }

        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let before = lookup(&storage).await.map_or(0, |(_, c)| c.unwrap_or(0));

        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let pipeline = IngestionPipeline::new(
            Box::new(MockTagger),
            Box::new(MockScorer),
            Box::new(MockExtractor),
            Box::new(MockEmbedder),
            storage,
        );
        let first = pipeline
            .ingest("The sky is blue.", EngramType::Episodic)
            .await
            .unwrap();
        let second = pipeline
            .ingest("The sky is blue.", EngramType::Episodic)
            .await
            .unwrap();
        assert_ne!(first, second);

        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let (concept_id, count) = lookup(&storage).await.expect("sky concept exists");
        assert_eq!(
            count.unwrap_or(0),
            before + 2,
            "two ingests must bump source_count twice on concept {concept_id}"
        );
    }

    #[tokio::test]
    #[ignore = "needs live HelixDB"]
    async fn pipeline_ingest_without_detector_skips_hook() {
        // No detector attached: the contradiction hook is skipped and
        // ingest succeeds. (Attaching a real detector needs an
        // `LlmProvider` mock from the parallel `mnemos-contradiction`
        // crate; the hook call site itself is pinned by
        // `contradiction_detector_matches_frozen_contract`.)
        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let pipeline = IngestionPipeline::new(
            Box::new(MockTagger),
            Box::new(MockScorer),
            Box::new(MockExtractor),
            Box::new(MockEmbedder),
            storage,
        );
        let _id = pipeline
            .ingest_with_importance("The sky is blue.", EngramType::Episodic, Some(0.9))
            .await
            .unwrap();
    }
}
