#![recursion_limit = "256"]
//! `mnemos-cli`: unified facade over the ingestion, retrieval, and
//! consolidation pipelines plus aggregate memory statistics.
//!
//! [`Cli`] owns one [`IngestionPipeline`], one [`RetrievalPipeline`] (behind
//! an async [`Mutex`] because `recall`/`reward` take `&mut self`), one
//! [`ConsolidationPipeline`], and its own [`Storage`] handle, which is the
//! only backend [`Cli::stats`] queries.
//!
//! All `HelixDB` access goes through [`Storage::client`] (the `helix-db 3.0.0`
//! HTTP client). The stats query is built with the `#[query]` macro from
//! `helix_db::dsl::prelude::*` and uses the `.count()` terminal
//! (grep-verified in `helix-ast-0.1.0/src/traversal.rs` as
//! `pub fn count(self) -> Traversal<Terminal, M>`). `#[query]` params
//! support `i64` but not `u64`; [`get_memory_stats`] takes no params, so no
//! id conversion is needed at this boundary.

use helix_db::dsl::prelude::*;
use mnemos_consolidation::ConsolidationPipeline;
use mnemos_contradiction::ContradictionDetector;
use mnemos_core::{
    ConsolidationReport, EngramId, EngramType, MemoryStats, MnemosError, ResonanceResult, Result,
};
use mnemos_ingestion::IngestionPipeline;
use mnemos_mitosis::MitosisSplitter;
use mnemos_retrieval::RetrievalPipeline;
use mnemos_storage::Storage;
use tokio::sync::Mutex;

/// Count `Engram` / `Concept` / `Identity` nodes plus contradicted engrams
/// in a single read batch.
///
/// Every binding ends in the `.count()` terminal, so `HelixDB` returns one
/// scalar per binding under its variable name.
#[query]
fn get_memory_stats() {
    read_batch()
        .var_as("engrams", g().n_with_label("Engram").count())
        .var_as("concepts", g().n_with_label("Concept").count())
        .var_as("identities", g().n_with_label("Identity").count())
        .var_as(
            "contradicted",
            g().n_with_label("Engram")
                .where_(Predicate::eq("contradiction_flag", true))
                .count(),
        )
        .returning(["engrams", "concepts", "identities", "contradicted"])
}

/// Fetch concepts with high `source_count` for mitosis splitting.
#[query]
fn get_mitosis_candidates_cli(min_source_count: i64) {
    let _ = &min_source_count;
    read_batch()
        .var_as(
            "concepts",
            g().n_with_label("Concept")
                .where_(Predicate::gte("source_count", min_source_count))
                .value_map(Some(vec!["$id", "name", "source_count"])),
        )
        .returning(["concepts"])
}

/// Fetch concepts that could crystallize into identities (high `source_count`, not yet linked).
#[query]
fn get_concepts_for_identity(min_source_count: i64) {
    let _ = &min_source_count;
    read_batch()
        .var_as(
            "concepts",
            g().n_with_label("Concept")
                .where_(Predicate::gte("source_count", min_source_count))
                .value_map(Some(vec!["$id", "name", "confidence", "source_count"])),
        )
        .returning(["concepts"])
}

/// Create an Identity trait node.
#[query]
fn create_identity(trait_name: String, value: f64, stability: f64, last_updated: String) {
    write_batch()
        .var_as(
            "identity",
            g().add_n(
                "Identity",
                vec![
                    ("trait", trait_name),
                    ("value", value),
                    ("stability", stability),
                    ("last_updated", last_updated),
                ],
            ),
        )
        .returning(["identity"])
}

/// Link Concept -> Identity via Defines edge.
#[query]
fn connect_defines_edge(concept_id: i64, identity_id: i64) {
    let _ = (&concept_id, &identity_id);
    write_batch()
        .var_as(
            "edge",
            g().n(NodeRef::param("concept_id")).add_e(
                "Defines",
                NodeRef::param("identity_id"),
                Vec::<(String, PropertyInput)>::new(),
            ),
        )
        .returning(["edge"])
}

/// Build the vector-index request for `Engram.embedding` in plain Rust.
///
/// The `#[query]` macro is deliberately NOT used here: its codegen only
/// accepts `bool/i64/f64/String/Vec` params and has no `NonZeroUsize`
/// conversion, while the index dimension is a runtime value from env
/// (`LlmConfig::embedding_dim`). `QueryRequest::write` + `set_query_name`
/// produce the identical request shape. Idempotent server-side
/// (`create_index_if_not_exists`).
fn create_engram_vector_index_request(
    dim: std::num::NonZeroUsize,
) -> helix_db::QueryRequest {
    let batch = write_batch().var_as(
        "index",
        g().create_vector_index_nodes(
            "Engram",
            "embedding",
            dim,
            VectorDistanceMetric::Cosine,
            None::<&str>,
        ),
    ).returning(["index"]);
    let mut request = helix_db::QueryRequest::write(batch);
    request.set_query_name("create_engram_vector_index");
    request
}

/// Unified CLI facade over all memory pipelines.
pub struct Cli {
    ingestion: IngestionPipeline,
    retrieval: Mutex<RetrievalPipeline>,
    consolidation: ConsolidationPipeline,
    storage: Storage,
    /// Optional contradiction detector for protocol flows.
    ///
    /// `None` by default; set via
    /// [`Cli::with_contradiction_detector`]. The detector is stored but not
    /// currently consulted by [`Cli`] itself: `contradiction_check` needs an
    /// embedder plus a detector plus storage-backed vector search, which
    /// [`Cli`] does not own (its embedder lives inside the ingestion and
    /// retrieval pipelines). That check is therefore implemented in the
    /// `mnemos-mcp-protocol` crate, which owns all three. This field exists
    /// so a detector handle can travel with the facade for future use.
    detector: Option<ContradictionDetector>,
    /// Optional mitosis splitter for aggressive consolidation.
    mitosis: Option<MitosisSplitter>,
    /// Last-seen `source_count` per concept id. A concept whose count hasn't
    /// grown since the previous aggressive cycle is skipped (no new engrams →
    /// no new clusters, identities, or contradictions to find), so steady-state
    /// 24/7 cycles cost ~zero LLM calls. Lives only while the daemon lives.
    seen_source_counts: Mutex<std::collections::HashMap<u64, i64>>,
    /// Concept ids already crystallized into identities this daemon lifetime.
    /// Prevents duplicate `Identity` nodes when a concept stays overloaded
    /// across cycles (its count keeps growing but it needs only one trait).
    crystallized: Mutex<std::collections::HashSet<u64>>,
}

impl Cli {
    /// Assemble the facade from its pipelines and a stats [`Storage`]
    /// handle. `retrieval` is wrapped in an async [`Mutex`] because
    /// [`RetrievalPipeline::recall`] and [`RetrievalPipeline::reward`] take
    /// `&mut self`.
    #[must_use]
    pub fn new(
        ingestion: IngestionPipeline,
        retrieval: RetrievalPipeline,
        consolidation: ConsolidationPipeline,
        storage: Storage,
    ) -> Self {
        Self {
            ingestion,
            retrieval: Mutex::new(retrieval),
            consolidation,
            storage,
            detector: None,
            mitosis: None,
            seen_source_counts: Mutex::new(std::collections::HashMap::new()),
            crystallized: Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Attach a [`ContradictionDetector`] to the facade (builder style).
    ///
    /// Stored for future protocol flows; no existing method changes
    /// behaviour when it is present.
    #[must_use]
    pub fn with_contradiction_detector(mut self, d: ContradictionDetector) -> Self {
        self.detector = Some(d);
        self
    }

    /// Attach a [`MitosisSplitter`] for aggressive consolidation.
    #[must_use]
    pub fn with_mitosis_splitter(mut self, splitter: MitosisSplitter) -> Self {
        self.mitosis = Some(splitter);
        self
    }

    /// Ingest one episodic memory.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError`] when any ML provider fails or when a `HelixDB`
    /// write fails.
    pub async fn ingest(&self, text: &str) -> Result<EngramId> {
        self.ingestion.ingest(text, EngramType::Episodic).await
    }

    /// Ingest one memory with an explicit type and optional importance
    /// override (protocol `store` path).
    ///
    /// Delegates to the frozen ingestion contract
    /// `IngestionPipeline::ingest_with_importance(&self, text, engram_type,
    /// importance: Option<f64>) -> Result<EngramId>`, which is landing in
    /// parallel: `Some(score)` skips the scorer, `None` lets the pipeline
    /// score it. If that method is absent on disk this crate fails to
    /// compile by design — no alternate ingestion path is attempted here.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError`] when any ML provider fails or when a `HelixDB`
    /// write fails.
    pub async fn ingest_full(
        &self,
        text: &str,
        engram_type: EngramType,
        importance: Option<f64>,
    ) -> Result<EngramId> {
        let start = std::time::Instant::now();
        let out = self
            .ingestion
            .ingest_with_importance(text, engram_type, importance)
            .await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match &out {
            Ok(_) => mnemos_telemetry::global().record_with_latency(
                "mnemos-cli",
                "ingest_full",
                true,
                "",
                ms,
            ),
            Err(e) => mnemos_telemetry::global().record_with_latency(
                "mnemos-cli",
                "ingest_full",
                false,
                &e.to_string(),
                ms,
            ),
        }
        out
    }

    /// Recall the top-`limit` memories for `query`.
    ///
    /// Always routes through the spreading activation wave
    /// (`recall_stimulated`: seed CRR → spread across
    /// `Reinforces`/`TemporalSequence`/`Contradicts` via learned edge weights
    /// → merge real rows) so rewarded weights visibly shape ranking.
    /// Set `MNEMOS_RECALL_WAVE=0/false/no/off` to revert to single-pass CRR.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError`] when embedding, query building, the vector
    /// search, or candidate decoding fails.
    pub async fn recall(&self, query: &str, limit: usize) -> Result<Vec<ResonanceResult>> {
        if Self::wave_enabled() {
            self.retrieval
                .lock()
                .await
                .recall_stimulated(query, limit)
                .await
        } else {
            self.retrieval.lock().await.recall(query, limit).await
        }
    }

    /// Protocol `recall`: oversampled recall formatted as memory snippets.
    ///
    /// Fetches `limit * 2` candidates via [`Cli::recall`] (oversample for
    /// resonance scoring, per the Tool-Call Protocol doc), truncates to
    /// `limit`, and formats each as:
    /// `[Memory: {first 60 chars} (relevance: {resonance:.2})]\n{episode}
    /// [importance: {x:.2}]`.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError`] when embedding, the vector search, or
    /// candidate decoding fails.
    pub async fn recall_protocol(&self, query: &str, limit: usize) -> Result<String> {
        let start = std::time::Instant::now();
        let out = self.recall_protocol_inner(query, limit).await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match &out {
            Ok(_) => mnemos_telemetry::global().record_with_latency(
                "mnemos-cli",
                "recall_protocol",
                true,
                "",
                ms,
            ),
            Err(e) => mnemos_telemetry::global().record_with_latency(
                "mnemos-cli",
                "recall_protocol",
                false,
                &e.to_string(),
                ms,
            ),
        }
        out
    }

    /// Inner recall-protocol implementation (kept separate so telemetry
    /// wraps exactly one call site).
    async fn recall_protocol_inner(&self, query: &str, limit: usize) -> Result<String> {
        let oversampled = self.recall(query, limit.saturating_mul(2)).await?;
        let top = oversampled.into_iter().take(limit);
        let mut out = String::new();
        for r in top {
            let snippet: String = r.episode_raw.chars().take(60).collect();
            out.push_str(&format!(
                "[Memory: {} (relevance: {:.2})]\n{} [importance: {:.2}]\n",
                snippet, r.resonance_score, r.episode_raw, r.importance_score
            ));
        }
        // Ledger is on by default — every recall notes that rewarding is available.
        let recall_id = self.retrieval.lock().await.last_recall_id();
        if let Some(id) = recall_id {
            out.push_str(&format!(
                "\n[Note: consider rewarding this recall via recall_id={id} with the reward tool]\n"
            ));
        } else {
            out.push_str("\n[Note: consider rewarding this recall with the reward tool]\n");
        }
        Ok(out)
    }

    // NOTE: `Cli` intentionally exposes NO `contradiction_check` method.
    // Checking a claim needs a claim embedding (embedder) + an emotional
    // tone tag (tagger) + storage-backed vector search + a
    // `ContradictionDetector`, and `Cli` owns none of the three providers
    // directly (its embedder/tagger handles live inside the ingestion and
    // retrieval pipelines). The protocol `contradiction_check` tool is
    // therefore implemented in the `mnemos-mcp-protocol` crate, which owns
    // `detector + embedder + tagger + storage` and calls into [`Cli`]
    // only for recall/consolidation. See that crate's docs.

    /// Apply a scalar reward `score` to edge weights via Adam.
    ///
    /// An empty `attributions` slice falls back to the seed contributions
    /// recorded by the latest [`Cli::recall`].
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError`] for forward compatibility with fallible
    /// weight stores (currently infallible).
    pub async fn reward(&self, attributions: &[f64], score: f64) -> Result<()> {
        self.retrieval.lock().await.reward(attributions, score)
    }

    /// Parallel-safe reward via ledger `recall_id` (always on).
    ///
    /// The `recall_id` must be from a prior `recall`/`recall_protocol` in this
    /// process. Falls back to `reward([], score)` when the id is unknown.
    pub async fn reward_with_id(&self, recall_id: u64, score: f64) -> Result<()> {
        self.retrieval.lock().await.reward_with_id(recall_id, score)
    }

    /// Last ledger recall id, if at least one `recall` ran (always on now).
    pub async fn last_recall_id(&self) -> Option<u64> {
        self.retrieval.lock().await.last_recall_id()
    }

    /// Whether spreading-activation recall is enabled (always on unless
    /// `MNEMOS_RECALL_WAVE=0/false/no/off` explicitly disables it).
    fn wave_enabled() -> bool {
        match std::env::var("MNEMOS_RECALL_WAVE").ok().as_deref().map(str::trim) {
            None | Some("") => true,
            Some(v) => !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            ),
        }
    }

    /// Run one consolidation ("sleep") cycle.
    ///
    /// Runs the aggressive path (mitosis splits + contradiction resolution +
    /// identity crystallization) so capabilities stay utilized. Use
    /// `consolidation.consolidate()` directly for decay-only.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when any `HelixDB` query fails.
    pub async fn consolidate(&self) -> Result<ConsolidationReport> {
        self.consolidate_aggressive(true).await
    }

    /// Protocol `consolidate` with an `aggressive` flag.
    ///
    /// When `aggressive` is false, identical to [`Cli::consolidate`].
    /// When true, also runs mitosis splitting (via [`MitosisSplitter`]
    /// if attached) and identity crystallization for overloaded concepts.
    /// Contradiction resolution is best-effort when a detector is attached.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when any `HelixDB` query fails.
    pub async fn consolidate_aggressive(
        &self,
        aggressive: bool,
    ) -> Result<ConsolidationReport> {
        let start = std::time::Instant::now();
        let out = self.consolidate_aggressive_inner(aggressive).await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match &out {
            Ok(_) => mnemos_telemetry::global().record_with_latency(
                "mnemos-cli",
                "consolidate_aggressive",
                true,
                "",
                ms,
            ),
            Err(e) => mnemos_telemetry::global().record_with_latency(
                "mnemos-cli",
                "consolidate_aggressive",
                false,
                &e.to_string(),
                ms,
            ),
        }
        out
    }

    async fn consolidate_aggressive_inner(&self, aggressive: bool) -> Result<ConsolidationReport> {
        let mut report = self.consolidation.consolidate().await?;
        if !aggressive {
            return Ok(report);
        }
        // Aggressive extras: mitosis + identity + contradiction scan (all best-effort).
        // Cost guard: concepts whose source_count hasn't grown since the last
        // cycle are skipped — no new engrams means no new clusters, identities,
        // or contradictions, so steady-state 24/7 cycles spend ~zero LLM calls.
        let candidates = self.fetch_mitosis_candidates(10).await.unwrap_or_default();
        let mut mitosis_children = 0u64;
        let mut identities_created = 0u64;
        let mut skipped_unchanged = 0u64;
        for cand in &candidates {
            {
                let seen = self.seen_source_counts.lock().await;
                if seen.get(&cand.id).copied().unwrap_or(-1) >= cand.source_count {
                    skipped_unchanged += 1;
                    continue;
                }
            }
            // Mitosis splitting if splitter attached.
            if let Some(splitter) = &self.mitosis {
                match splitter
                    .split_concept(&self.storage, cand.id, &cand.name, 3, 5)
                    .await
                {
                    Ok(r) => mitosis_children += r.children_created,
                    Err(e) => {
                        mnemos_telemetry::global().record(
                            "mnemos-cli",
                            "consolidate_aggressive.mitosis",
                            false,
                            &e.to_string(),
                        );
                    }
                }
            }
            // Identity crystallization: once per concept per daemon lifetime
            // (no duplicate traits when a concept stays overloaded).
            if self.crystallized.lock().await.insert(cand.id) {
                if let Ok(identity_id) = self.crystallize_identity(cand).await {
                    // Link concept -> identity if we created one.
                    let _ = self.link_defines(cand.id, identity_id).await;
                    identities_created += 1;
                }
            }
            // Contradiction scan if detector attached.
            if let Some(detector) = &self.detector {
                if let Err(e) = detector.scan_concept(&self.storage, cand.id).await {
                    mnemos_telemetry::global().record(
                        "mnemos-cli",
                        "consolidate_aggressive.contradiction",
                        false,
                        &e.to_string(),
                    );
                } else {
                    // Re-count contradictions after scan for report.
                    if let Ok(stats) = self.stats().await {
                        report.contradictions_linked = stats.contradictions;
                    }
                }
            }
            // Remember this count so unchanged concepts skip LLM work next cycle.
            self.seen_source_counts
                .lock()
                .await
                .insert(cand.id, cand.source_count);
        }
        mnemos_telemetry::global().record(
            "mnemos-cli",
            "consolidate_aggressive",
            true,
            &format!(
                "candidates={} mitosis_children={} identities={} skipped_unchanged={}",
                candidates.len(),
                mitosis_children,
                identities_created,
                skipped_unchanged,
            ),
        );
        Ok(report)
    }

    async fn fetch_mitosis_candidates(
        &self,
        min_source_count: i64,
    ) -> Result<Vec<ConceptCandidate>> {
        let request = get_mitosis_candidates_cli(min_source_count)
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        let response: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        Ok(response_rows_concept(&response, "concepts"))
    }

    async fn crystallize_identity(&self, cand: &ConceptCandidate) -> Result<u64> {
        let trait_name = cand.name.clone();
        let value = (0.5 + cand.confidence * 0.5).clamp(0.0, 1.0);
        let stability: f64 = 0.6;
        let last_updated = chrono::Utc::now().to_rfc3339();
        let request = create_identity(trait_name, value, stability, last_updated)
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        let response: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        parse_node_id(&response)
    }

    async fn link_defines(&self, concept_id: u64, identity_id: u64) -> Result<()> {
        let cid = i64::try_from(concept_id)
            .map_err(|_| MnemosError::Storage(format!("concept id overflow: {concept_id}")))?;
        let iid = i64::try_from(identity_id)
            .map_err(|_| MnemosError::Storage(format!("identity id overflow: {identity_id}")))?;
        let request = connect_defines_edge(cid, iid).map_err(|e| MnemosError::Storage(e.to_string()))?;
        let _: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Create the vector index on `Engram.embedding` (one-time setup).
    ///
    /// Idempotent (`create_index_if_not_exists`). `dimension` must match the
    /// stored vectors exactly — it comes from env (`LlmConfig::embedding_dim`),
    /// never from a CLI flag. Any positive dimension works.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when `dimension` is 0, or the
    /// `HelixDB` request fails.
    pub async fn setup_vector_index(&self, dimension: usize) -> Result<String> {
        let start = std::time::Instant::now();
        let dim = std::num::NonZeroUsize::new(dimension).ok_or_else(|| {
            MnemosError::Storage(format!("dimension must be > 0: {dimension}"))
        })?;
        let request = create_engram_vector_index_request(dim);
        let response: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        mnemos_telemetry::global().record_with_latency(
            "mnemos-cli",
            "setup_vector_index",
            true,
            &format!("dimension={dimension}"),
            ms,
        );
        Ok(format!("vector index on Engram.embedding (dim {dimension}): {response}"))
    }

    /// Aggregate node counts via the single [`get_memory_stats`] query
    /// against the facade's own [`Storage`] handle.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when the query fails to build or
    /// the `HelixDB` request fails.
    pub async fn stats(&self) -> Result<MemoryStats> {
        let start = std::time::Instant::now();
        let request = get_memory_stats().map_err(|e| MnemosError::Storage(e.to_string()))?;
        let response: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        let stats = MemoryStats {
            total_engrams: count_binding(&response, "engrams"),
            contradictions: count_binding(&response, "contradicted"),
            concepts: count_binding(&response, "concepts"),
            identities: count_binding(&response, "identities"),
        };
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        mnemos_telemetry::global().record_with_latency("mnemos-cli", "stats", true, &format!("engrams={} concepts={}", stats.total_engrams, stats.concepts), ms);
        mnemos_telemetry::global().record_system_stats(serde_json::to_value(&stats).unwrap_or_default());
        Ok(stats)
    }
}

/// Extract one `.count()` binding as `u64`.
///
/// Tolerates the scalar arriving bare (`{"engrams": 5}`), wrapped in a
/// single-element array, or nested under a `"count"` key; anything else
/// (missing, null, multi-row arrays) falls back to the array length or `0`
/// so stats never fail on response-shape drift.
fn count_binding(response: &serde_json::Value, key: &str) -> u64 {
    count_value(response.get(key).unwrap_or(&serde_json::Value::Null))
}

/// Coerce one count-shaped JSON value to `u64`.
fn count_value(value: &serde_json::Value) -> u64 {
    match value {
        serde_json::Value::Number(n) => n
            .as_u64()
            .or_else(|| n.as_i64().and_then(|i| u64::try_from(i).ok()))
            .unwrap_or(0),
        serde_json::Value::Array(items) => match items.as_slice() {
            [] => 0,
            [single] => count_value(single),
            many => u64::try_from(many.len()).unwrap_or(0),
        },
        serde_json::Value::Object(map) => map.get("count").map_or(0, count_value),
        _ => 0,
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ConceptCandidate {
    #[serde(rename = "$id")]
    id: u64,
    name: String,
    #[serde(default)]
    confidence: f64,
    #[serde(default)]
    source_count: i64,
}

fn response_rows_concept(response: &serde_json::Value, key: &str) -> Vec<ConceptCandidate> {
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

fn parse_node_id(response: &serde_json::Value) -> Result<u64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use mnemos_core::{ConsolidationConfig, ExtractedConcept, StimulationConfig};
    use mnemos_edge_weights::EdgeWeights;
    use mnemos_embedding_trait::EmbeddingProvider;
    use mnemos_ml_trait::{ConceptExtractor, EmotionalTagger, ImportanceScorer};
    use mnemos_stimulation::StimulationEngine;
    use serde_json::json;

    struct MockTagger;
    struct MockScorer;
    struct MockExtractor;
    struct MockEmbedder;

    #[async_trait]
    impl EmotionalTagger for MockTagger {
        async fn tag(&self, _text: &str) -> Result<f64> {
            Ok(0.25)
        }
    }

    #[async_trait]
    impl ImportanceScorer for MockScorer {
        async fn score(&self, _text: &str) -> Result<f64> {
            Ok(0.8)
        }
    }

    #[async_trait]
    impl ConceptExtractor for MockExtractor {
        async fn extract(&self, _text: &str) -> Result<Vec<ExtractedConcept>> {
            Ok(vec![ExtractedConcept {
                name: "sky".to_string(),
                confidence: 0.9,
            }])
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.1; 8])
        }
    }

    /// Build a facade over mock providers. `Storage::new` performs no
    /// network I/O, so this is a pure constructor with no live DB.
    fn test_cli() -> Cli {
        let ingestion = IngestionPipeline::new(
            Box::new(MockTagger),
            Box::new(MockScorer),
            Box::new(MockExtractor),
            Box::new(MockEmbedder),
            Storage::new("http://localhost:6969", "mnemos").expect("storage builds"),
        );
        let retrieval = RetrievalPipeline::new(
            Storage::new("http://localhost:6969", "mnemos").expect("storage builds"),
            StimulationEngine::new(StimulationConfig::default(), EdgeWeights::defaults()),
            EdgeWeights::defaults(),
            Box::new(MockEmbedder),
        );
        let consolidation = ConsolidationPipeline::new(
            Storage::new("http://localhost:6969", "mnemos").expect("storage builds"),
            ConsolidationConfig::default(),
        );
        Cli::new(
            ingestion,
            retrieval,
            consolidation,
            Storage::new("http://localhost:6969", "mnemos").expect("storage builds"),
        )
    }

    #[test]
    fn constructs_with_mock_providers_without_db() {
        let _cli = test_cli();
    }

    #[test]
    fn memory_stats_query_builds() {
        // No live DB: proves the #[query] wiring (name + read kind).
        let request = get_memory_stats().expect("stats query builds");
        assert_eq!(request.query_name(), Some("get_memory_stats"));
        assert!(matches!(
            request.request_type(),
            helix_db::QueryRequestType::Read
        ));
    }

    #[test]
    fn count_binding_handles_response_shapes() {
        assert_eq!(count_binding(&json!({"engrams": 5}), "engrams"), 5);
        assert_eq!(count_binding(&json!({"engrams": [7]}), "engrams"), 7);
        assert_eq!(
            count_binding(&json!({"engrams": [{"count": 3}]}), "engrams"),
            3
        );
        assert_eq!(
            count_binding(&json!({"engrams": {"count": 4}}), "engrams"),
            4
        );
        assert_eq!(count_binding(&json!({}), "engrams"), 0);
        assert_eq!(count_binding(&json!({"engrams": null}), "engrams"), 0);
    }

    #[tokio::test]
    #[ignore = "needs live HelixDB at http://localhost:6969"]
    async fn live_ingest_roundtrip() {
        let cli = test_cli();
        let _id = cli.ingest("The sky is blue.").await.expect("ingest");
    }

    #[tokio::test]
    #[ignore = "needs live HelixDB at http://localhost:6969"]
    async fn live_recall_and_reward() {
        let cli = test_cli();
        let results = cli.recall("blue sky", 5).await.expect("recall");
        assert!(results.len() <= 5);
        cli.reward(&[0.5; 8], 1.0).await.expect("explicit reward");
        cli.reward(&[], 1.0).await.expect("fallback reward");
    }

    #[tokio::test]
    #[ignore = "needs live HelixDB at http://localhost:6969"]
    async fn live_consolidate_reports() {
        let cli = test_cli();
        let report = cli.consolidate().await.expect("consolidate");
        assert_eq!(report.contradictions_linked, 0);
    }

    #[tokio::test]
    #[ignore = "needs live HelixDB at http://localhost:6969"]
    async fn live_stats_counts() {
        let cli = test_cli();
        let stats = cli.stats().await.expect("stats");
        let _ = stats.total_engrams;
    }
}
