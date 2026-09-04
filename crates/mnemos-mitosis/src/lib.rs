#![recursion_limit = "256"]
//! mnemos-mitosis: HDBSCAN-driven concept splitting.
//!
//! When a `Concept` accumulates too many recalling engrams, [`MitosisSplitter`]
//! clusters their embeddings via HDBSCAN, names each cluster via LLM, and
//! creates child `Concept` nodes with `SpawnedFrom` edges. Noise engrams stay
//! with the parent.
//!
//! ## Grounding (helix-db 3.0.0 / helix-ast 0.1.0, grep-verified)
//!
//! * `#[query]` from `helix_db::dsl::prelude::*`; params are `i64`, never `u64`
//!   (node ids cross as `i64` via `i64::try_from`).
//! * Edge-drop ops verified in `helix-ast/src/traversal.rs`:
//!   `drop_edge_labeled(to, label)` and `drop_edge_by_id(EdgeRef::id(...))`.
//! * Edge traversal: `out_e(Some("Recalls"))` / `in_e(Some(...))` land on
//!   edges; `value_map` projects `$id` (edge id) and other props.
//! * `Hdbscan::new(&data, hyper_params)` + `.cluster()` → `Vec<i32>` where
//!   `-1` = noise, `0..n-1` = cluster ids (`hdbscan` 0.12.0, `src/hdbscan.rs`).

use std::collections::HashMap;

use chrono::Utc;
use helix_db::dsl::prelude::*;
use mnemos_core::{compute_centroid, ConceptId, EngramId, MnemosError, Result};
use mnemos_llm_trait::LlmProvider;
use mnemos_storage::Storage;
use serde::{Deserialize, Serialize};

/// Minimum confidence to accept an LLM-generated child name.
const MIN_CHILD_CONFIDENCE: f64 = 0.5;

/// Maximum LLM retry attempts per cluster name.
const NAMING_MAX_ATTEMPTS: usize = 3;

/// HDBSCAN-driven concept splitter.
///
/// Clusters overloaded concepts' engram embeddings, names each cluster via
/// LLM, and creates child `Concept` nodes with `SpawnedFrom` edges.
pub struct MitosisSplitter {
    llm: Box<dyn LlmProvider>,
}

impl MitosisSplitter {
    /// Build a splitter backed by `llm`.
    #[must_use]
    pub fn new(llm: Box<dyn LlmProvider>) -> Self {
        Self { llm }
    }

    /// Pure HDBSCAN wrapper: cluster `embeddings` with `min_cluster_size`.
    ///
    /// Returns one `i64` label per input embedding:
    /// - `-1` = noise (outlier, stays with parent)
    /// - `0..n_clusters-1` = cluster assignment
    ///
    /// Empty input returns an empty vec. On any HDBSCAN error, every point is
    /// labeled noise (`-1`).
    ///
    /// ## hdbscan 0.12 API mapping
    /// `Hdbscan::cluster()` returns `Vec<i32>` where `-1` is noise and
    /// non-negative values are cluster ids. We map `i32` → `i64` for the
    /// public API; the noise label stays `-1`.
    pub fn cluster_embeddings(embeddings: &[Vec<f32>], min_cluster_size: usize) -> Vec<i64> {
        if embeddings.is_empty() {
            return Vec::new();
        }
        let hyper_params = hdbscan::HdbscanHyperParams::builder()
            .min_cluster_size(min_cluster_size.max(2))
            .build();
        let clusterer = hdbscan::Hdbscan::new(embeddings, hyper_params);
        match clusterer.cluster() {
            Ok(labels) => labels.into_iter().map(i64::from).collect(),
            Err(_) => vec![-1_i64; embeddings.len()],
        }
    }

    /// Split one overloaded concept into child concepts.
    ///
    /// Steps:
    /// 1. Fetch engrams recalling `concept_id` (with embeddings + episode_raw).
    /// 2. Cluster embeddings via [`Self::cluster_embeddings`].
    /// 3. Edge cases per reference doc: 0–1 clusters → no split (return zeros);
    ///    cap children to `max_children` largest clusters; clusters smaller
    ///    than `min_cluster_size` are noise (HDBSCAN handles this).
    /// 4. Per cluster: LLM-name it, create child `Concept`, link `SpawnedFrom`
    ///    edge, reassign `Recalls` + `AbstractsTo` edges (drop old → create new).
    /// 5. Update `source_counts` (parent = noise count, each child = cluster size).
    /// 6. Store centroid embedding on each child.
    /// 7. Record telemetry (`"mnemos-mitosis"` / `"split_concept"`).
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when any HelixDB request fails.
    pub async fn split_concept(
        &self,
        storage: &Storage,
        concept_id: u64,
        parent_name: &str,
        min_cluster_size: usize,
        max_children: usize,
    ) -> Result<MitosisReport> {
        let outcome = self
            .split_concept_inner(storage, concept_id, parent_name, min_cluster_size, max_children)
            .await;
        let (ok, detail) = match &outcome {
            Ok(report) => (
                true,
                format!(
                    "children={} reassigned={} noise={}",
                    report.children_created, report.reassigned, report.noise_kept
                ),
            ),
            Err(e) => (false, e.to_string()),
        };
        mnemos_telemetry::global().record("mnemos-mitosis", "split_concept", ok, &detail);
        outcome
    }

    /// Inner workhorse (telemetry wraps at [`Self::split_concept`]).
    async fn split_concept_inner(
        &self,
        storage: &Storage,
        concept_id: u64,
        parent_name: &str,
        min_cluster_size: usize,
        max_children: usize,
    ) -> Result<MitosisReport> {
        let concept_i64 = to_i64(concept_id)?;

        // 1. Fetch engrams recalling this concept.
        let request = get_engrams_recalling_concept_with_embeddings(concept_i64)
            .map_err(storage_error)?;
        let response: serde_json::Value = storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(storage_error)?;
        let engrams: Vec<EngramWithEmbedding> = response_rows(&response, "engrams");

        if engrams.len() < 2 {
            return Ok(MitosisReport {
                concept_id,
                children_created: 0,
                reassigned: 0,
                noise_kept: 0,
            });
        }

        // 2. Cluster embeddings.
        let embeddings: Vec<Vec<f32>> = engrams.iter().map(|e| e.embedding.clone()).collect();
        let labels = Self::cluster_embeddings(&embeddings, min_cluster_size);

        // 3. Group engrams by cluster label.
        let mut cluster_map: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            cluster_map.entry(label).or_default().push(i);
        }

        // Noise (-1) stays with parent.
        let noise_indices = cluster_map.remove(&-1).unwrap_or_default();

        // Edge case: 0 or 1 clusters → no split needed.
        if cluster_map.len() <= 1 {
            return Ok(MitosisReport {
                concept_id,
                children_created: 0,
                reassigned: 0,
                noise_kept: noise_indices.len() as u64,
            });
        }

        // Sort clusters by size descending; keep top max_children.
        let mut cluster_sizes: Vec<(i64, usize)> = cluster_map
            .iter()
            .map(|(k, v)| (*k, v.len()))
            .collect();
        cluster_sizes.sort_by(|a, b| b.1.cmp(&a.1));

        let formation_date = Utc::now().to_rfc3339();
        let mut children_created = 0_u64;
        let mut reassigned = 0_u64;

        for (rank, (cluster_id, _size)) in cluster_sizes.iter().enumerate() {
            if rank >= max_children {
                break;
            }
            let member_indices = &cluster_map[cluster_id];
            let cluster_size = member_indices.len() as i64;

            // Snippets for the LLM naming prompt.
            let snippets: Vec<String> = member_indices
                .iter()
                .map(|&i| engrams[i].episode_raw.clone())
                .collect();

            // Name the cluster (with fallback).
            let child_name = name_cluster(&*self.llm, parent_name, &snippets, rank + 1).await;

            // Centroid of the cluster's embeddings.
            let centroid = compute_centroid(
                &member_indices
                    .iter()
                    .map(|&i| engrams[i].embedding.clone())
                    .collect::<Vec<_>>(),
            );

            // Create child concept (with centroid embedding).
            let child_request = create_child_concept(
                child_name,
                0.8,
                formation_date.clone(),
                cluster_size,
                centroid,
            )
            .map_err(storage_error)?;
            let child_response: serde_json::Value = storage
                .client()
                .query(child_request)
                .send()
                .await
                .map_err(storage_error)?;
            let child_id = parse_node_id(&child_response)?;

            // SpawnedFrom edge: parent → child.
            let spawned_request =
                connect_spawned_from_edge(concept_i64, to_i64(child_id)?).map_err(storage_error)?;
            let _: serde_json::Value = storage
                .client()
                .query(spawned_request)
                .send()
                .await
                .map_err(storage_error)?;

            // Reassign edges for each engram in this cluster.
            for &member_idx in member_indices {
                let engram_i64 = to_i64(engrams[member_idx].id)?;

                // Drop old edges to parent.
                let drop_recalls = drop_recalls_edge(engram_i64, concept_i64).map_err(storage_error)?;
                let _: serde_json::Value = storage
                    .client()
                    .query(drop_recalls)
                    .send()
                    .await
                    .map_err(storage_error)?;

                let drop_abstracts =
                    drop_abstracts_to_edge(engram_i64, concept_i64).map_err(storage_error)?;
                let _: serde_json::Value = storage
                    .client()
                    .query(drop_abstracts)
                    .send()
                    .await
                    .map_err(storage_error)?;

                // Create new edges to child.
                let add_recalls =
                    connect_recalls_edge(engram_i64, to_i64(child_id)?).map_err(storage_error)?;
                let _: serde_json::Value = storage
                    .client()
                    .query(add_recalls)
                    .send()
                    .await
                    .map_err(storage_error)?;

                let add_abstracts =
                    connect_abstracts_to_edge(engram_i64, to_i64(child_id)?).map_err(storage_error)?;
                let _: serde_json::Value = storage
                    .client()
                    .query(add_abstracts)
                    .send()
                    .await
                    .map_err(storage_error)?;

                reassigned += 1;
            }

            children_created += 1;
        }

        // Parent source_count = noise + capped-out cluster engrams.
        let capped_count: i64 = cluster_sizes
            .iter()
            .skip(max_children)
            .map(|(_, s)| *s as i64)
            .sum();
        let noise_count = noise_indices.len() as i64 + capped_count;
        let set_parent =
            set_concept_source_count(concept_i64, noise_count).map_err(storage_error)?;
        let _: serde_json::Value = storage
            .client()
            .query(set_parent)
            .send()
            .await
            .map_err(storage_error)?;

        Ok(MitosisReport {
            concept_id,
            children_created,
            reassigned,
            noise_kept: noise_count as u64,
        })
    }
}

/// Outcome of one mitosis split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitosisReport {
    /// The parent concept that was split.
    pub concept_id: ConceptId,
    /// Number of child concepts created.
    pub children_created: u64,
    /// Number of engrams reassigned to children.
    pub reassigned: u64,
    /// Number of engrams that stayed with the parent (noise).
    pub noise_kept: u64,
}

/// LLM naming response shape.
#[derive(Debug, Deserialize)]
struct NamingResponse {
    name: String,
    confidence: f64,
}

/// Engram row returned by `get_engrams_recalling_concept_with_embeddings`.
#[derive(Debug, Deserialize)]
struct EngramWithEmbedding {
    #[serde(rename = "$id")]
    id: EngramId,
    episode_raw: String,
    embedding: Vec<f32>,
}

/// Name a cluster via LLM with a repair-retry loop.
///
/// Prompt per reference doc: "Given these {n} memory snippets about
/// '{parent}', generate a concise, specific concept name (2-6 words)...
/// Return JSON {name, confidence}".
///
/// Falls back to `"{parent} #{index}"` if all attempts fail or confidence is
/// below [`MIN_CHILD_CONFIDENCE`].
async fn name_cluster(
    llm: &dyn LlmProvider,
    parent_name: &str,
    snippets: &[String],
    index: usize,
) -> String {
    let fallback = format!("{parent_name} #{index}");
    let snippets_text = snippets.join("\n");
    let prompt = format!(
        "Given these {} memory snippets about '{}', generate a concise, specific \
         concept name (2-6 words):\n\n{}\n\nReturn JSON: {{\"name\": \"...\", \"confidence\": 0.0-1.0}}",
        snippets.len(),
        parent_name,
        snippets_text,
    );
    let repair_hint = "Return ONLY a JSON object with \"name\" (string) and \"confidence\" (number 0-1) fields.";
    let mut current = prompt.clone();

    for _ in 0..NAMING_MAX_ATTEMPTS {
        match llm.chat(&current).await {
            Ok(raw) => {
                if let Some(json_str) = extract_json(&raw) {
                    if let Ok(resp) = serde_json::from_str::<NamingResponse>(json_str) {
                        if resp.confidence >= MIN_CHILD_CONFIDENCE {
                            return resp.name;
                        }
                    }
                }
                current = format!(
                    "{prompt}\n\nYour previous response was not valid JSON or had low confidence. \
                     {repair_hint}\nPrevious response:\n{raw}"
                );
            }
            Err(_) => {
                current = format!("{prompt}\n\nPrevious attempt failed. {repair_hint}");
            }
        }
    }
    fallback
}

/// Slice the first `{…}` span so prose-wrapped JSON still parses.
fn extract_json(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end > start {
        Some(&raw[start..=end])
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// HelixDB queries
// ---------------------------------------------------------------------------

/// Fetch engrams recalling `concept_id` with embeddings + episode_raw.
///
/// Traverses incoming `Recalls` edges from the concept (edge direction is
/// `Engram -Recalls-> Concept`, see `mnemos-ingestion`).
#[query]
fn get_engrams_recalling_concept_with_embeddings(concept_id: i64) {
    let _ = &concept_id;
    read_batch()
        .var_as(
            "engrams",
            g().n(NodeRef::param("concept_id"))
                .in_(Some("Recalls"))
                .value_map(Some(vec!["$id", "episode_raw", "embedding"])),
        )
        .returning(["engrams"])
}

/// Create a child `Concept` with a centroid `embedding`.
#[query]
fn create_child_concept(
    name: String,
    confidence: f64,
    formation_date: String,
    source_count: i64,
    embedding: Vec<f32>,
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
                    ("embedding", embedding),
                ],
            ),
        )
        .returning(["concept"])
}

/// Create a `SpawnedFrom` edge: parent `Concept` → child `Concept`.
#[query]
fn connect_spawned_from_edge(parent_id: i64, child_id: i64) {
    let _ = (&parent_id, &child_id);
    write_batch()
        .var_as(
            "edge",
            g().n(NodeRef::param("parent_id")).add_e(
                "SpawnedFrom",
                NodeRef::param("child_id"),
                Vec::<(String, PropertyInput)>::new(),
            ),
        )
        .returning(["edge"])
}

/// Create a `Recalls` edge: engram → concept.
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

/// Create an `AbstractsTo` edge: engram → concept.
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

/// Drop the `Recalls` edge from `from_id` to `to_id`.
///
/// Grep-verified in `helix-ast 0.1.0/src/traversal.rs`:
/// `drop_edge_labeled(to, label)` on `Traversal<OnNodes, M>`.
#[query]
fn drop_recalls_edge(from_id: i64, to_id: i64) {
    let _ = (&from_id, &to_id);
    write_batch()
        .var_as(
            "dropped",
            g().n(NodeRef::param("from_id"))
                .drop_edge_labeled(NodeRef::param("to_id"), "Recalls"),
        )
        .returning(["dropped"])
}

/// Drop the `AbstractsTo` edge from `from_id` to `to_id`.
#[query]
fn drop_abstracts_to_edge(from_id: i64, to_id: i64) {
    let _ = (&from_id, &to_id);
    write_batch()
        .var_as(
            "dropped",
            g().n(NodeRef::param("from_id"))
                .drop_edge_labeled(NodeRef::param("to_id"), "AbstractsTo"),
        )
        .returning(["dropped"])
}

/// Read back one concept's `source_count`.
#[query]
fn get_concept_source_count(concept_id: i64) {
    let _ = &concept_id;
    read_batch()
        .var_as(
            "concept",
            g().n(NodeRef::param("concept_id"))
                .value_map(Some(vec!["$id", "source_count"])),
        )
        .returning(["concept"])
}

/// Write back one concept's `source_count`.
#[query]
fn set_concept_source_count(concept_id: i64, source_count: i64) {
    let _ = &concept_id;
    write_batch()
        .var_as(
            "updated",
            g().n(NodeRef::param("concept_id"))
                .set_property("source_count", source_count),
        )
        .returning(["updated"])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map any displayable query/client failure into `MnemosError::Storage`.
fn storage_error<E: std::fmt::Display>(error: E) -> MnemosError {
    MnemosError::Storage(error.to_string())
}

/// Convert a node id to the `i64` query param the `#[query]` macro accepts.
fn to_i64(id: u64) -> Result<i64> {
    i64::try_from(id).map_err(|_| MnemosError::Internal(format!("node id overflow: {id}")))
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

/// Extract a created node's id from a HelixDB JSON response.
fn parse_node_id(response: &serde_json::Value) -> Result<EngramId> {
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
            if let Some(items) = value.as_array() {
                for item in items {
                    if let Some(id) = item.get("$id").and_then(json_id) {
                        return Ok(id);
                    }
                }
            }
        }
    }
    Err(MnemosError::Storage(format!(
        "response missing node $id: {response}"
    )))
}

/// Deserialize the row array under `key` (tolerates a single object, missing,
/// or null bindings by yielding zero or more rows).
fn response_rows<T>(response: &serde_json::Value, key: &str) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    match response.get(key) {
        Some(serde_json::Value::Array(items)) => items
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    // --- cluster_embeddings tests (no DB) ---

    #[test]
    fn cluster_embeddings_two_blobs_plus_outlier() {
        // Two tight 3-dim blobs far apart, plus one distant outlier.
        let embeddings = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.0, 0.0],
            vec![0.0, 0.1, 0.0],
            vec![0.0, 0.0, 0.1],
            vec![0.1, 0.1, 0.0],
            vec![10.0, 10.0, 10.0],
            vec![10.1, 10.0, 10.0],
            vec![10.0, 10.1, 10.0],
            vec![10.0, 10.0, 10.1],
            vec![10.1, 10.1, 10.0],
            vec![50.0, 50.0, 50.0], // outlier / noise
        ];
        let labels = MitosisSplitter::cluster_embeddings(&embeddings, 3);
        assert_eq!(labels.len(), embeddings.len());

        // The outlier must be noise (-1).
        assert_eq!(labels[10], -1, "distant outlier should be noise");

        // There should be exactly 2 distinct non-noise cluster ids.
        let clusters: std::collections::HashSet<i64> =
            labels.iter().filter(|&&l| l >= 0).copied().collect();
        assert_eq!(clusters.len(), 2, "expected 2 clusters, got {clusters:?}");

        // Points within each blob share a label.
        let blob0_label = labels[0];
        for i in 0..5 {
            assert_eq!(labels[i], blob0_label, "blob0 point {i} label mismatch");
        }
        let blob1_label = labels[5];
        for i in 5..10 {
            assert_eq!(labels[i], blob1_label, "blob1 point {i} label mismatch");
        }
        assert_ne!(blob0_label, blob1_label, "blobs must differ");
    }

    #[test]
    fn cluster_embeddings_single_blob_one_label() {
        let embeddings = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
        ];
        let labels = MitosisSplitter::cluster_embeddings(&embeddings, 3);
        // Single blob → all in one cluster (one non-noise label), or all noise
        // if HDBSCAN can't form a cluster. Either way, at most 1 cluster id.
        let clusters: std::collections::HashSet<i64> =
            labels.iter().filter(|&&l| l >= 0).copied().collect();
        assert!(clusters.len() <= 1, "single blob → ≤1 cluster, got {clusters:?}");
    }

    #[test]
    fn cluster_embeddings_empty_input() {
        let labels = MitosisSplitter::cluster_embeddings(&[], 3);
        assert!(labels.is_empty());
    }

    // --- naming fallback test ---

    struct GarbageLlm;
    #[async_trait]
    impl LlmProvider for GarbageLlm {
        async fn chat(&self, _prompt: &str) -> Result<String> {
            Ok("not json at all".to_string())
        }
        async fn chat_with_system(&self, _system: &str, _user: &str) -> Result<String> {
            Ok("not json at all".to_string())
        }
    }

    #[tokio::test]
    async fn naming_fallback_on_garbage_llm() {
        let llm = GarbageLlm;
        let name = name_cluster(&llm, "databases", &["postgres".to_string()], 1).await;
        assert_eq!(name, "databases #1", "garbage LLM → fallback name");
    }

    struct LowConfidenceLlm;
    #[async_trait]
    impl LlmProvider for LowConfidenceLlm {
        async fn chat(&self, _prompt: &str) -> Result<String> {
            Ok(r#"{"name":"some name","confidence":0.1}"#.to_string())
        }
        async fn chat_with_system(&self, _system: &str, _user: &str) -> Result<String> {
            Ok(r#"{"name":"some name","confidence":0.1}"#.to_string())
        }
    }

    #[tokio::test]
    async fn naming_fallback_on_low_confidence() {
        let llm = LowConfidenceLlm;
        let name = name_cluster(&llm, "databases", &["postgres".to_string()], 2).await;
        assert_eq!(name, "databases #2", "low confidence → fallback name");
    }

    struct GoodLlm;
    #[async_trait]
    impl LlmProvider for GoodLlm {
        async fn chat(&self, _prompt: &str) -> Result<String> {
            Ok(r#"{"name":"PostgreSQL internals","confidence":0.9}"#.to_string())
        }
        async fn chat_with_system(&self, _system: &str, _user: &str) -> Result<String> {
            Ok(r#"{"name":"PostgreSQL internals","confidence":0.9}"#.to_string())
        }
    }

    #[tokio::test]
    async fn naming_accepts_valid_high_confidence() {
        let llm = GoodLlm;
        let name = name_cluster(&llm, "databases", &["postgres WAL".to_string()], 1).await;
        assert_eq!(name, "PostgreSQL internals");
    }

    // --- query builder tests (no DB) ---

    #[test]
    fn query_builders_produce_requests() {
        let req =
            get_engrams_recalling_concept_with_embeddings(7).expect("builds");
        assert_eq!(req.query_name(), Some("get_engrams_recalling_concept_with_embeddings"));
        assert!(matches!(
            req.request_type(),
            helix_db::QueryRequestType::Read
        ));

        let req = create_child_concept(
            "child".to_string(),
            0.8,
            "2026-09-03T00:00:00Z".to_string(),
            5,
            vec![0.1, 0.2],
        )
        .expect("builds");
        assert_eq!(req.query_name(), Some("create_child_concept"));

        let req = connect_spawned_from_edge(1, 2).expect("builds");
        assert_eq!(req.query_name(), Some("connect_spawned_from_edge"));

        let req = drop_recalls_edge(1, 2).expect("builds");
        assert_eq!(req.query_name(), Some("drop_recalls_edge"));

        let req = drop_abstracts_to_edge(1, 2).expect("builds");
        assert_eq!(req.query_name(), Some("drop_abstracts_to_edge"));

        let req = set_concept_source_count(7, 3).expect("builds");
        assert_eq!(req.query_name(), Some("set_concept_source_count"));
        let params = req.parameters().expect("params");
        assert!(matches!(
            params.get("source_count"),
            Some(helix_db::QueryValue::I64(3))
        ));
    }

    // --- live split test (ignored, needs HelixDB) ---

    #[tokio::test]
    #[ignore = "needs live HelixDB at http://localhost:6969"]
    async fn split_concept_live_roundtrip() {
        let storage = Storage::new("http://localhost:6969", "mnemos").unwrap();
        let splitter = MitosisSplitter::new(Box::new(GoodLlm));
        let report = splitter
            .split_concept(&storage, 1, "databases", 3, 5)
            .await
            .expect("split against live DB");
        let _ = report;
    }
}
