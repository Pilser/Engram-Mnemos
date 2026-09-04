#![recursion_limit = "256"]
//! `mnemos-retrieval`: single-pass Cognitive Resonance Retrieval (CRR).
//!
//! Pipeline: embed query -> vector search (`search_engrams_full`) ->
//! [`compute_resonance`] -> sort desc -> truncate -> bump
//! `activation_count` (best-effort) -> stash seed attributions for
//! [`RetrievalPipeline::reward`].
//!
//! # CRR formula
//!
//! ```text
//! resonance = semantic_sim
//!     * recency_weight
//!     * (1 + |emotional_charge|)
//!     * identity_alignment (Engram -> Recalls -> Concepts -> Defines ->
//!       Identity traits; 1.0 when unlinked)
//!     * (contradiction_flag ? 0.5 : 1.0)
//! ```
//!
//! `recency_weight = exp(-decay_rate * days_elapsed).max(0.01)` with
//! `days_elapsed = (now - timestamp) / 86400`, using the candidate's own
//! per-engram `decay_rate` (doc 06), reimplemented locally via
//! [`mnemos_core::parse_timestamp_rfc3339`]. [`compute_resonance`] never
//! touches stimulation internals.
//!
//! # `HelixDB` grounding (verified against helix-db 3.0.0 / helix-ast 0.1.0)
//!
//! * `#[query]` is imported via `helix_db::dsl::prelude::*` (re-exported
//!   from `helix_ast::prelude` plus `helix_dsl_macros::query`).
//! * Grep of `helix-ast-0.1.0/src/traversal.rs` shows `vector_search_with`
//!   exists only as a mid-stream ranker on `Traversal<OnNodes>` /
//!   `Traversal<OnEdges>`; it is NOT a source op on `Traversal<Empty>`, so
//!   the reference doc's `g().vector_search_with("Engram", ...)` does not
//!   compile. Substitution: the source-stage equivalent
//!   `vector_search_nodes_with(label, property, query_vector:
//!   impl Into<PropertyInput>, k: impl Into<StreamBound>, tenant_value:
//!   Option<PropertyInput>) -> Traversal<OnNodes>`, which likewise returns
//!   `Engram` nodes directly (embedding lives on the node, no `HasVector`
//!   traversal). `Vec<f32>: Into<PropertyInput>` (via `PropertyValue::F32Array`)
//!   and `i64: Into<StreamBound>` are both implemented.
//! * `Client::query` signature (helix-db `src/lib.rs`):
//!   `pub fn query<R: for<'de> Deserialize<'de>>(&self, request: QueryRequest)
//!   -> QueryExecutionRequest<'_, 'static, R>` with
//!   `async fn send(self) -> Result<R, HelixError>`; responses are decoded
//!   as `serde_json::Value` and `candidates` extracted per the reference doc.
//! * No `HelixDbSource`, `Client::open*`, or embedded APIs are used: all DB
//!   access goes through [`mnemos_storage::Storage::client`].
//! * `#[query]` params support `Vec<f32>` / `i64` but NOT `u64`, so node ids
//!   cross the query boundary as `i64` (`i64::try_from` at the call site).
//!
//! # Stimulation contract (sibling crate may still be in flight)
//!
//! Only this surface of `mnemos_stimulation::StimulationEngine` is used:
//! `new(config, weights)`, `initial_activation(sem, rec, emo) -> f64`,
//! `transfer(idx, act) -> f64`, `neighbors(&self, storage, node_id, label)`.
//! Assumed shapes: `new(StimulationConfig, EdgeWeights)`, and
//! `initial_activation` callable as a method on the stored engine. `recall`
//! is single-pass CRR, so `transfer`/`neighbors` (multi-hop spreading) are
//! intentionally unused here and reserved for a future wave loop.

use helix_db::dsl::prelude::*;
use mnemos_core::{EngramCandidate, EngramId, MnemosError, ResonanceResult, Result};
use mnemos_edge_weights::{EdgeWeights, IDX_RECALLS};
use mnemos_embedding_trait::EmbeddingProvider;
use mnemos_stimulation::StimulationEngine;
use mnemos_storage::Storage;
use std::collections::HashMap;

/// Vector search over `Engram.embedding`, projecting every field
/// [`compute_resonance`] needs plus `$id` / `$distance`.
///
/// See the crate docs for why this uses `vector_search_nodes_with` instead
/// of the doc's `g().vector_search_with(...)` spelling.
#[query]
fn search_engrams_full(query_embedding: Vec<f32>, limit: i64) {
    read_batch()
        .var_as(
            "candidates",
            g()
                .vector_search_nodes_with("Engram", "embedding", query_embedding, limit, None)
                .value_map(Some(vec![
                    "$id",
                    "$distance",
                    "episode_raw",
                    "emotional_charge",
                    "importance_score",
                    "decay_rate",
                    "activation_count",
                    "timestamp",
                    "engram_type",
                    "compression_level",
                    "contradiction_flag",
                ])),
        )
        .returning(["candidates"])
}

/// Read back one engram's `activation_count` (recall activation bump).
#[query]
fn get_engram_activation(engram_id: i64) {
    // `engram_id` binds via `NodeRef::param("engram_id")` below; the macro's
    // `Expr` binding is only referenced here to silence `unused_variables`.
    let _ = &engram_id;
    read_batch()
        .var_as(
            "engram",
            g().n(NodeRef::param("engram_id"))
                .value_map(Some(vec!["$id", "activation_count"])),
        )
        .returning(["engram"])
}

/// Write back one engram's `activation_count` (recall activation bump).
#[query]
fn set_engram_activation(engram_id: i64, activation_count: i64) {
    // See `get_engram_activation` for why this binding is referenced.
    let _ = &engram_id;
    write_batch()
        .var_as(
            "updated",
            g().n(NodeRef::param("engram_id"))
                .set_property("activation_count", activation_count),
        )
        .returning(["updated"])
}

/// Cognitive chain for one engram: `Engram -Recalls-> Concept -Defines->
/// Identity`, projecting each linked trait's `value` + `stability`.
///
/// Single chained traversal (no intermediate round-trips). Chainability is
/// grep-verified in `helix-ast`'s `traversal.rs`: `Traversal<OnNodes>::out`
/// returns `Traversal<OnNodes>`, so `.out(..).out(..)` chains, and
/// `value_map` is the terminal. Edge labels are static strings baked into
/// the AST (`out` takes `Option<impl Into<String>>`), so `Some("Recalls")` /
/// `Some("Defines")` are used directly; only the node id crosses the query
/// boundary as `i64` (see crate docs).
#[query]
fn engram_identity_traits(engram_id: i64) {
    // `engram_id` binds via `NodeRef::param("engram_id")` below; the macro's
    // `Expr` binding is only referenced here to silence `unused_variables`.
    let _ = &engram_id;
    read_batch()
        .var_as(
            "traits",
            g().n(NodeRef::param("engram_id"))
                .out(Some("Recalls"))
                .out(Some("Defines"))
                .value_map(Some(vec!["value", "stability"])),
        )
        .returning(["traits"])
}

/// Five-factor CRR scoring over vector-search candidates.
///
/// * `semantic_sims[i]` should be `1.0 - candidates[i].distance`; a missing
///   entry defaults to `0.5` (neutral).
/// * `alignments[i]` is the graph-derived identity alignment for
///   `candidates[i]` (see [`identity_alignment_for`]); a missing entry
///   defaults to `1.0` (neutral).
/// * `contradiction_flag` halves the score (`* 0.5`), otherwise `* 1.0`.
#[must_use]
pub fn compute_resonance_with_alignment(
    candidates: &[EngramCandidate],
    semantic_sims: &[f64],
    now_unix_secs: f64,
    alignments: &[f64],
) -> Vec<ResonanceResult> {
    let mut results: Vec<ResonanceResult> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            // Factor 1: semantic similarity (from vector search).
            let semantic = semantic_sims.get(i).copied().unwrap_or(0.5);

            // Factor 2: recency — per-engram Ebbinghaus weight
            // (doc 06: exp(-decay_rate * days), floored; same decay_rate
            // the consolidation loop thresholds on).
            let engram_ts = mnemos_core::parse_timestamp_rfc3339(&c.timestamp);
            let days_elapsed = (now_unix_secs - engram_ts) / 86400.0;
            let recency = (-c.decay_rate * days_elapsed).exp().max(0.01);

            // Factor 3: emotional charge amplification.
            let emotional_boost = 1.0 + c.emotional_charge.abs();

            // Factor 4: identity alignment (graph-derived; neutral when absent).
            let identity_alignment = alignments.get(i).copied().unwrap_or(1.0);

            // Factor 5: contradiction penalty.
            let contradiction_factor = if c.contradiction_flag { 0.5 } else { 1.0 };

            let resonance =
                semantic * recency * emotional_boost * identity_alignment * contradiction_factor;

            ResonanceResult {
                engram_id: c.id,
                resonance_score: resonance,
                episode_raw: c.episode_raw.clone(),
                emotional_charge: c.emotional_charge,
                importance_score: c.importance_score,
                identity_alignment,
                semantic_sim: semantic,
                recency_weight: recency,
            }
        })
        .collect::<Vec<_>>();
    // Doc Step 4: highest resonance first (NaN-safe; recall() re-sorts anyway).
    results.sort_by(|a, b| {
        b.resonance_score
            .partial_cmp(&a.resonance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Five-factor CRR scoring with neutral identity alignment.
///
/// Backward-compatible wrapper over [`compute_resonance_with_alignment`]
/// (all-`1.0` alignments); prefer the `_with_alignment` form once the
/// identity graph is reachable.
#[must_use]
pub fn compute_resonance(
    candidates: &[EngramCandidate],
    semantic_sims: &[f64],
    now_unix_secs: f64,
) -> Vec<ResonanceResult> {
    let neutral = vec![1.0; candidates.len()];
    compute_resonance_with_alignment(candidates, semantic_sims, now_unix_secs, &neutral)
}

/// Graph-derived identity alignment for one engram (doc 06, Factor 4 +
/// Step 3: `GetCognitiveChain`, engram -> concepts via `Recalls` ->
/// identities via `Defines`; alignment weighted by trait `value` and
/// `stability`). No embeddings needed.
///
/// Best-effort: any failure (query build, transport, decode) records a
/// `recall.identity_alignment` telemetry event and returns `1.0` (neutral).
/// An engram with no linked traits also scores `1.0`.
pub async fn identity_alignment_for(storage: &Storage, engram_id: u64) -> f64 {
    let node_param = i64::try_from(engram_id).unwrap_or(i64::MAX);
    let request = match engram_identity_traits(node_param) {
        Ok(request) => request,
        Err(e) => {
            mnemos_telemetry::global().record(
                "mnemos-retrieval",
                "recall.identity_alignment",
                false,
                &format!("build engram_identity_traits: {e}"),
            );
            return 1.0;
        }
    };
    let response: serde_json::Value = match storage
        .client()
        .query(request)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => {
            mnemos_telemetry::global().record(
                "mnemos-retrieval",
                "recall.identity_alignment",
                false,
                &format!("engram_identity_traits: {e}"),
            );
            return 1.0;
        }
    };
    alignment_from_traits_response(&response)
}

/// Pure fold over an `engram_identity_traits` response envelope:
/// `sum(value * stability) / count`, `1.0` when no usable traits, clamped
/// to `[0, 1]`. Traits missing a numeric `value`/`stability` are skipped.
fn alignment_from_traits_response(response: &serde_json::Value) -> f64 {
    let traits = match response
        .get("traits")
        .and_then(serde_json::Value::as_array)
    {
        Some(traits) => traits,
        None => return 1.0,
    };
    let mut sum = 0.0;
    let mut count = 0_u32;
    for t in traits {
        let (Some(value), Some(stability)) = (
            t.get("value").and_then(number_as_f64),
            t.get("stability").and_then(number_as_f64),
        ) else {
            continue;
        };
        sum += value * stability;
        count += 1;
    }
    if count == 0 {
        return 1.0;
    }
    (sum / f64::from(count)).clamp(0.0, 1.0)
}

/// A JSON number (`f64` or integer) as `f64`; `None` otherwise.
fn number_as_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
}

/// Single-pass CRR recall pipeline with reward-driven edge-weight learning.
pub struct RetrievalPipeline {
    storage: Storage,
    stimulation: StimulationEngine,
    edge_weights: EdgeWeights,
    embedder: Box<dyn EmbeddingProvider>,
    /// Seed contributions from the latest [`Self::recall`], folded uniformly
    /// into the `IDX_RECALLS` slot (seeds arrive via vector search over
    /// `Engram.embedding`, i.e. the Recalls pathway). Used as the fallback
    /// attribution source by [`Self::reward`].
    last_attributions: Vec<f64>,
    /// Recall ledger for parallel-safe reward (always on).
    /// `recall_id → attributions` isolated per recall; `next_recall_id` is
    /// monotonic. Replaces the old single `last_attributions` fallback.
    ledger: std::collections::HashMap<u64, Vec<f64>>,
    next_recall_id: u64,
}

impl RetrievalPipeline {
    /// Assemble the pipeline. The stimulation engine is stored opaquely;
    /// only its documented 4-method surface is ever called.
    #[must_use]
    pub fn new(
        storage: Storage,
        stimulation: StimulationEngine,
        edge_weights: EdgeWeights,
        embedder: Box<dyn EmbeddingProvider>,
    ) -> Self {
        let mut loaded = edge_weights;
        // Opt-in persistence: load weights file if present (atomic writes on reward).
        if Self::ledger_enabled() {
            if let Some(path) = Self::weights_file_path() {
                if let Ok(data) = std::fs::read_to_string(&path) {
                    if let Ok(w) = serde_json::from_str::<EdgeWeights>(&data) {
                        loaded = w;
                    }
                }
            }
        }
        Self {
            storage,
            stimulation,
            edge_weights: loaded,
            embedder,
            last_attributions: Vec::new(),
            ledger: std::collections::HashMap::new(),
            next_recall_id: 1,
        }
    }

    /// Whether the recall ledger is enabled — always on (parallel-safe by default).
    fn ledger_enabled() -> bool {
        true
    }

    fn weights_file_path() -> Option<String> {
        let p = std::env::var("MNEMOS_WEIGHTS_FILE")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "./data/helix/mnemos-weights.json".to_string());
        Some(p)
    }

    fn persist_weights(&self) {
        if !Self::ledger_enabled() {
            return;
        }
        let Some(path) = Self::weights_file_path() else {
            return;
        };
        if let Ok(json) = serde_json::to_string_pretty(&self.edge_weights) {
            if let Some(parent) = std::path::Path::new(&path).parent() {
                let _ = std::fs::create_dir_all(parent);
                let tmp = format!("{path}.tmp");
                if std::fs::write(&tmp, json).is_ok() {
                    let _ = std::fs::rename(&tmp, &path);
                }
            }
        }
    }

    /// Recall top-`limit` memories for `query`.
    ///
    /// Embeds the query, runs `search_engrams_full`, fetches per-candidate
    /// identity alignment via [`identity_alignment_for`], scores with
    /// [`compute_resonance_with_alignment`] (`semantic = 1 - distance`),
    /// sorts descending,
    /// truncates, bumps `activation_count` on recalled engrams (best-effort:
    /// per-item failures are ignored), and records seed attributions for
    /// [`Self::reward`].
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError`] when embedding, query building, the vector
    /// search, or candidate decoding fails.
    pub async fn recall(&mut self, query: &str, limit: usize) -> Result<Vec<ResonanceResult>> {
        // Step 1: embed the query client-side (no server-side Embed()).
        let query_embedding = match self.embedder.embed(query).await {
            Ok(v) => v,
            Err(e) => {
                mnemos_telemetry::global().record(
                    "mnemos-retrieval",
                    "recall",
                    false,
                    &format!("embed: {e}"),
                );
                return Err(e);
            }
        };

        // Step 2: vector search returns Engram nodes directly.
        let fetch = i64::try_from(limit).unwrap_or(i64::MAX).max(0);
        let request = match search_engrams_full(query_embedding, fetch) {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("build search_engrams_full: {e}");
                mnemos_telemetry::global().record("mnemos-retrieval", "recall", false, &msg);
                return Err(MnemosError::Storage(msg));
            }
        };
        let response: serde_json::Value = match self
            .storage
            .client()
            .query(request)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("search_engrams_full: {e}");
                mnemos_telemetry::global().record("mnemos-retrieval", "recall", false, &msg);
                return Err(MnemosError::Storage(msg));
            }
        };

        // Step 3: deserialize candidates.
        let candidates: Vec<EngramCandidate> = match serde_json::from_value(
            response
                .get("candidates")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        ) {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("decode candidates: {e}");
                mnemos_telemetry::global().record("mnemos-retrieval", "recall", false, &msg);
                return Err(MnemosError::Storage(msg));
            }
        };

        // Step 4: semantic similarities (1.0 - cosine distance).
        let semantic_sims: Vec<f64> =
            candidates.iter().map(|c| 1.0 - c.distance).collect();

        // Step 5: per-candidate identity alignment (graph-derived via the
        // cognitive chain; best-effort neutral on failure) + resonance
        // scoring. Candidates are bounded by the seed limit (~30), so
        // sequential awaits are fine.
        let now_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        let mut alignments = Vec::with_capacity(candidates.len());
        for c in &candidates {
            alignments.push(identity_alignment_for(&self.storage, c.id).await);
        }
        let mut results = compute_resonance_with_alignment(
            &candidates,
            &semantic_sims,
            now_unix_secs,
            &alignments,
        );

        // Step 6: sort highest-first, take top N.
        results.sort_by(|a, b| {
            b.resonance_score
                .partial_cmp(&a.resonance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        // Step 7: activate recalled engrams (best-effort).
        for r in &results {
            self.bump_activation(r.engram_id).await;
        }

        // Step 8: record seed contributions — uniform per recalled engram
        // (each seed contributed equally to this recall wave), strength from
        // `StimulationEngine::initial_activation`, averaged into IDX_RECALLS.
        let mut total = 0.0;
        let mut count = 0.0;
        for r in &results {
            total += self.stimulation.initial_activation(
                r.semantic_sim,
                r.recency_weight,
                r.emotional_charge,
            );
            count += 1.0;
        }
        let mut attr = vec![0.0; 8];
        if count > 0.0 {
            attr[IDX_RECALLS] = total / count;
        }
        self.last_attributions.clone_from(&attr);
        // Opt-in ledger: isolate per-recall attributions for parallel-safe reward.
        if Self::ledger_enabled() {
            let id = self.next_recall_id;
            self.next_recall_id = self.next_recall_id.wrapping_add(1);
            self.ledger.insert(id, attr);
            // Keep ledger bounded (evict oldest when > 1024).
            if self.ledger.len() > 1024 {
                if let Some(k) = self.ledger.keys().next().copied() {
                    self.ledger.remove(&k);
                }
            }
            // Telemetry hint for the agent.
            mnemos_telemetry::global().record(
                "mnemos-retrieval",
                "recall.ledger",
                true,
                &format!("recall_id={id} consider rewarding this recall"),
            );
        }

        mnemos_telemetry::global().record(
            "mnemos-retrieval",
            "recall",
            true,
            &format!("results={}", results.len()),
        );
        Ok(results)
    }

    /// Recall with spreading activation (Stimulation Layer wave loop).
    ///
    /// 1. Seed via single-pass [`Self::recall`] (without bumping activation twice).
    /// 2. Spread `max_iterations` waves: for each surfaced engram above `tau`,
    ///    fetch neighbors via `Reinforces` / `TemporalSequence` / `Contradicts`,
    ///    transfer activation via `stimulation.transfer`, decay by `gamma`,
    ///    and surface new engrams. This is the doc 07 wave loop in minimal form.
    /// 3. Merge wave-discovered engrams (attenuated) with seed results, resort.
    ///
    /// Best-effort: neighbor fetch failures are skipped via telemetry.
    pub async fn recall_stimulated(&mut self, query: &str, limit: usize) -> Result<Vec<ResonanceResult>> {
        // Seed phase: reuse recall but avoid double activation bump by calling internal helper.
        // For simplicity, call recall and then do one spreading wave from its top results.
        let mut seed_results = self.recall(query, limit).await?;
        if seed_results.is_empty() || self.stimulation.config().max_iterations == 0 {
            return Ok(seed_results);
        }
        let tau = self.stimulation.config().tau_threshold;
        let gamma = self.stimulation.config().gamma_decay;
        let mut seen: HashMap<u64, f64> = seed_results
            .iter()
            .map(|r| (r.engram_id, r.resonance_score))
            .collect();
        let mut wave_ids: Vec<u64> = seed_results
            .iter()
            .filter(|r| r.resonance_score > tau)
            .map(|r| r.engram_id)
            .collect();

        for _ in 0..self.stimulation.config().max_iterations {
            let mut next_wave: Vec<(u64, f64)> = Vec::new();
            for &engram_id in &wave_ids {
                let base_act = seen.get(&engram_id).copied().unwrap_or(0.0);
                if base_act <= tau {
                    continue;
                }
                // Spread via engram->engram edges (Reinforces, TemporalSequence, Contradicts).
                for (label, idx) in [
                    ("Reinforces", mnemos_edge_weights::IDX_REINFORCES),
                    ("TemporalSequence", mnemos_edge_weights::IDX_TEMPORAL_SEQ),
                    ("Contradicts", mnemos_edge_weights::IDX_CONTRADICTS),
                ] {
                    let neighbors = match self
                        .stimulation
                        .neighbors(&self.storage, engram_id, label)
                        .await
                    {
                        Ok(ids) => ids,
                        Err(e) => {
                            mnemos_telemetry::global().record(
                                "mnemos-retrieval",
                                "recall_stimulated.neighbors",
                                false,
                                &e.to_string(),
                            );
                            Vec::new()
                        }
                    };
                    for nid in neighbors {
                        if seen.contains_key(&nid) {
                            continue;
                        }
                        let transferred = self.stimulation.transfer(idx, base_act);
                        let decayed = StimulationEngine::apply_decay(transferred, gamma);
                        if decayed > tau {
                            next_wave.push((nid, decayed));
                        }
                    }
                }
            }
            if next_wave.is_empty() {
                break;
            }
            // Insert wave results into seen and prepare next iteration.
            let mut next_ids = Vec::new();
            for (id, act) in next_wave {
                seen.insert(id, act);
                next_ids.push(id);
            }
            wave_ids = next_ids;
        }
        // Merge: fetch episode data for wave-discovered engrams not in seed?
        // For now, synthesize minimal ResonanceResults for wave nodes by fetching
        // their engram data via a simple read (best-effort, skip on failure).
        // To keep this method testable offline, we surface wave ids as attenuated scores
        // without extra DB reads if the seed already covered them; otherwise we append
        // synthetic entries with attenuated resonance.
        for (id, act) in &seen {
            if seed_results.iter().any(|r| &r.engram_id == id) {
                continue;
            }
            // Synthetic wave-discovered entry (attenuated, no DB fetch to stay offline-friendly).
            seed_results.push(ResonanceResult {
                engram_id: *id,
                resonance_score: *act * 0.5, // wave attenuation
                episode_raw: format!("[wave-discovered engram {id}]"),
                emotional_charge: 0.0,
                importance_score: 0.5,
                identity_alignment: 1.0,
                semantic_sim: 0.5,
                recency_weight: 1.0,
            });
        }
        // Resort and truncate to limit.
        seed_results.sort_by(|a, b| {
            b.resonance_score
                .partial_cmp(&a.resonance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        seed_results.truncate(limit);
        Ok(seed_results)
    }

    /// Apply a scalar `reward` to edge weights via Adam.
    ///
    /// An empty `attributions` slice falls back to the seed contributions
    /// recorded by the latest [`Self::recall`]; otherwise the given slice is
    /// used (and stored for future fallback calls).
    ///
    /// # Errors
    ///
    /// Currently infallible (returns `Result` for forward compatibility with
    /// fallible weight stores).
    pub fn reward(&mut self, attributions: &[f64], reward: f64) -> Result<()> {
        if attributions.is_empty() {
            let last = self.last_attributions.clone();
            self.edge_weights.adam_update(&last, reward);
        } else {
            self.edge_weights.adam_update(attributions, reward);
            self.last_attributions.clone_from(&attributions.to_vec());
        }
        self.persist_weights();
        mnemos_telemetry::global().record_weights(serde_json::to_value(&self.edge_weights).unwrap_or_default());
        Ok(())
    }

    /// Parallel-safe reward: `recall_id` must be from a prior `recall`.
    /// Falls back to `reward([], score)` when the id is missing (telemetry records the miss).
    pub fn reward_with_id(&mut self, recall_id: u64, score: f64) -> Result<()> {
        if !Self::ledger_enabled() {
            return self.reward(&[], score);
        }
        let Some(attributions) = self.ledger.remove(&recall_id) else {
            mnemos_telemetry::global().record(
                "mnemos-retrieval",
                "reward.ledger_miss",
                false,
                &format!("recall_id={recall_id} not found (expired or already rewarded)"),
            );
            return self.reward(&[], score);
        };
        self.edge_weights.adam_update(&attributions, score);
        self.last_attributions.clone_from(&attributions);
        self.persist_weights();
        mnemos_telemetry::global().record_weights(serde_json::to_value(&self.edge_weights).unwrap_or_default());
        mnemos_telemetry::global().record(
            "mnemos-retrieval",
            "reward.ledger_hit",
            true,
            &format!("recall_id={recall_id} score={score}"),
        );
        Ok(())
    }

    /// Last ledger recall id, if ledger is enabled and at least one recall ran.
    #[must_use]
    pub fn last_recall_id(&self) -> Option<u64> {
        if !Self::ledger_enabled() || self.ledger.is_empty() {
            return None;
        }
        // next_recall_id is one past the last inserted, so last is next-1 if present.
        let candidate = self.next_recall_id.wrapping_sub(1);
        if self.ledger.contains_key(&candidate) {
            Some(candidate)
        } else {
            self.ledger.keys().next().copied()
        }
    }

    /// Read-modify-write `activation_count + 1`. All failures are swallowed
    /// by design (recall must not fail because activation bookkeeping did).
    async fn bump_activation(&self, engram_id: EngramId) {
        let node_param = i64::try_from(engram_id).unwrap_or(i64::MAX);
        let Ok(read_req) = get_engram_activation(node_param) else {
            return;
        };
        let Ok(current) = self
            .storage
            .client()
            .query::<serde_json::Value>(read_req)
            .send()
            .await
        else {
            return;
        };
        let count = current
            .get("engram")
            .and_then(|v| v.get(0))
            .and_then(|r| r.get("activation_count"))
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        let Ok(write_req) = set_engram_activation(node_param, count.saturating_add(1)) else {
            return;
        };
        let _ = self
            .storage
            .client()
            .query::<serde_json::Value>(write_req)
            .send()
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_core::{Result as CoreResult, StimulationConfig};
    use std::future::Future;
    use std::pin::Pin;

    fn candidate(id: EngramId, distance: f64, contradiction_flag: bool) -> EngramCandidate {
        EngramCandidate {
            id,
            episode_raw: format!("episode {id}"),
            emotional_charge: 0.3,
            importance_score: 0.8,
            decay_rate: 0.01,
            activation_count: 0,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            engram_type: "episodic".to_string(),
            compression_level: 0,
            contradiction_flag,
            distance,
        }
    }

    /// One day after the fixed candidate timestamp.
    const NOW: f64 = 1_767_225_600.0 + 86400.0;

    #[test]
    fn higher_semantic_similarity_wins() {
        let candidates = vec![candidate(1, 0.1, false), candidate(2, 0.4, false)];
        let sims = vec![0.9, 0.6];
        let results = compute_resonance(&candidates, &sims, NOW);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].engram_id, 1);
        assert!(results[0].resonance_score > results[1].resonance_score);
    }

    #[test]
    fn contradiction_penalty_halves_score() {
        let plain = vec![candidate(1, 0.2, false)];
        let flagged = vec![candidate(1, 0.2, true)];
        let sims = vec![0.8];
        let a = compute_resonance(&plain, &sims, NOW);
        let b = compute_resonance(&flagged, &sims, NOW);
        assert!(a[0].resonance_score > 0.0);
        let ratio = b[0].resonance_score / a[0].resonance_score;
        assert!((ratio - 0.5).abs() < 1e-12, "ratio was {ratio}");
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(compute_resonance(&[], &[], NOW).is_empty());
    }

    #[test]
    fn missing_semantic_entry_defaults_to_neutral() {
        let results = compute_resonance(&[candidate(1, 0.2, false)], &[], NOW);
        assert_eq!(results.len(), 1);
        assert!((results[0].semantic_sim - 0.5).abs() < 1e-12);
    }

    #[test]
    fn high_alignment_wins_semantic_tie() {
        let candidates = vec![candidate(1, 0.2, false), candidate(2, 0.2, false)];
        let sims = vec![0.8, 0.8];
        let results = compute_resonance_with_alignment(&candidates, &sims, NOW, &[0.2, 0.9]);
        assert_eq!(results.len(), 2);
        // Same semantics/recency/emotion: the aligned engram scores higher.
        assert_eq!(results[0].engram_id, 2);
        assert!((results[0].identity_alignment - 0.9).abs() < 1e-12);
        assert!((results[1].identity_alignment - 0.2).abs() < 1e-12);
        assert!(results[0].resonance_score > results[1].resonance_score);
    }

    #[test]
    fn empty_alignments_default_to_neutral() {
        let candidates = vec![candidate(1, 0.2, false)];
        let sims = vec![0.8];
        let aligned = compute_resonance_with_alignment(&candidates, &sims, NOW, &[]);
        let neutral = compute_resonance(&candidates, &sims, NOW);
        assert!((aligned[0].identity_alignment - 1.0).abs() < 1e-12);
        assert!((aligned[0].resonance_score - neutral[0].resonance_score).abs() < 1e-12);
    }

    #[test]
    fn alignment_response_averages_value_times_stability() {
        let response: serde_json::Value = serde_json::json!({
            "traits": [
                { "value": 0.8, "stability": 0.9 },
                { "value": 0.6, "stability": 0.4 },
            ]
        });
        let want = f64::midpoint(0.8 * 0.9, 0.6 * 0.4);
        let got = alignment_from_traits_response(&response);
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    #[test]
    fn alignment_response_defaults_and_clamps() {
        let empty: serde_json::Value = serde_json::json!({ "traits": [] });
        assert_eq!(alignment_from_traits_response(&empty), 1.0);
        let missing: serde_json::Value = serde_json::json!({ "other": [] });
        assert_eq!(alignment_from_traits_response(&missing), 1.0);
        // Traits without numeric value/stability are skipped -> neutral.
        let unusable: serde_json::Value =
            serde_json::json!({ "traits": [{ "value": "high" }, {}] });
        assert_eq!(alignment_from_traits_response(&unusable), 1.0);
        // Out-of-range means clamp to [0, 1].
        let big: serde_json::Value =
            serde_json::json!({ "traits": [{ "value": 2.0, "stability": 2.0 }] });
        assert_eq!(alignment_from_traits_response(&big), 1.0);
        let negative: serde_json::Value =
            serde_json::json!({ "traits": [{ "value": -2.0, "stability": 1.0 }] });
        assert_eq!(alignment_from_traits_response(&negative), 0.0);
    }

    /// Fixed-vector embedder. Hand-written `Future` impl so unit tests need
    /// no extra dependencies (`async-trait` is not in `[dev-dependencies]`).
    /// Signature mirrors the `async-trait 0.1` expansion exactly: one
    /// lifetime per elided reference (`&self` -> `'life0`, `text` ->
    /// `'life1`) plus `'async_trait` on the boxed future.
    struct FixedEmbedder(Vec<f32>);

    impl EmbeddingProvider for FixedEmbedder {
        fn embed<'life0, 'life1, 'async_trait>(
            &'life0 self,
            _text: &'life1 str,
        ) -> Pin<Box<dyn Future<Output = CoreResult<Vec<f32>>> + Send + 'async_trait>>
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            let v = self.0.clone();
            Box::pin(async move { Ok(v) })
        }
    }

    #[test]
    fn search_query_builds_typed_read_request() {
        // No live DB: proves the #[query] wiring (params, kind, name).
        let req = search_engrams_full(vec![0.1, 0.2], 5).expect("builds");
        assert_eq!(req.query_name(), Some("search_engrams_full"));
        assert!(matches!(
            req.request_type(),
            helix_db::QueryRequestType::Read
        ));
        let params = req.parameters().expect("parameters present");
        assert!(matches!(
            params.get("limit"),
            Some(helix_db::QueryValue::I64(5))
        ));
        assert!(matches!(
            params.get("query_embedding"),
            Some(helix_db::QueryValue::Array(v)) if v.len() == 2
        ));
    }

    #[test]
    fn identity_traits_query_builds_typed_read_request() {
        // No live DB: proves the cognitive-chain #[query] wiring (params,
        // kind, name). Node ids cross the query boundary as i64.
        let req = engram_identity_traits(7).expect("builds");
        assert_eq!(req.query_name(), Some("engram_identity_traits"));
        assert!(matches!(
            req.request_type(),
            helix_db::QueryRequestType::Read
        ));
        let params = req.parameters().expect("parameters present");
        assert!(matches!(
            params.get("engram_id"),
            Some(helix_db::QueryValue::I64(7))
        ));
    }

    /// Live round-trip: vector search -> CRR -> activation bump -> reward.
    /// Requires a running `HelixDB` with the MNEMOS schema + vector index.
    #[tokio::test]
    #[ignore = "needs live HelixDB at HELIX_URL (default http://localhost:6969)"]
    async fn recall_and_reward_against_live_db() {
        let storage = Storage::from_config(&mnemos_core::StorageConfig::default())
            .await
            .expect("storage builds without network I/O");
        let stimulation =
            StimulationEngine::new(StimulationConfig::default(), EdgeWeights::defaults());
        let mut pipe = RetrievalPipeline::new(
            storage,
            stimulation,
            EdgeWeights::defaults(),
            Box::new(FixedEmbedder(vec![0.1; 8])),
        );
        let results = pipe.recall("a joyful celebration", 5).await.expect("recall");
        assert!(results.len() <= 5);
        pipe.reward(&[0.5; 8], 1.0).expect("explicit-attribution reward");
        pipe.reward(&[], 1.0).expect("stored-attribution reward");
    }

    /// Live cognitive chain: Engram -> Recalls -> Concepts -> Defines ->
    /// Identity traits folded to a unit-range alignment. Requires a running
    /// `HelixDB` with the MNEMOS schema (an engram with id 1 need not exist:
    /// unlinked ids fall back to neutral 1.0).
    #[tokio::test]
    #[ignore = "needs live HelixDB at HELIX_URL (default http://localhost:6969)"]
    async fn live_identity_alignment_is_unit_range() {
        let storage = Storage::from_config(&mnemos_core::StorageConfig::default())
            .await
            .expect("storage builds without network I/O");
        let alignment = identity_alignment_for(&storage, 1).await;
        assert!(
            (0.0..=1.0).contains(&alignment),
            "alignment in [0,1]: {alignment}"
        );
    }
}
