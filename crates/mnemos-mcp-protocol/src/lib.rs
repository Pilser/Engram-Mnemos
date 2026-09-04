#![recursion_limit = "256"]
//! `mnemos-mcp-protocol`: MCP server exposing the 4 Tool-Call Protocol tools.
//!
//! See `__reference/.../Explanation-docs/query-layer/01-Tool-Call-Protocol.md`.
//! One MCP tool per protocol operation, served over stdio:
//!
//! | Tool | Params | Returns (JSON string) |
//! |------|--------|---------------------|
//! | `recall` | `{ query*, limit?=5 (cap 50), type?=auto, since?=all }` | `{ "query", "limit", "memories" }` (formatted snippets from [`Cli::recall_protocol`]) |
//! | `store` | `{ content*, importance?=auto, type?=auto }` | `{ "engram_id": … }` |
//! | `contradiction_check` | `{ claim* }` | `{ "found", "conflicts" }` or `{ "found": false, "message": "no contradictions" }` |
//! | `consolidate` | `{ aggressive?=false }` | `{ "pruned", "compressed", "promoted" }` |
//!
//! `type` / `since` on `recall` are accepted and documented as filters for
//! future use; they do not change retrieval today. Every tool returns its
//! payload as a JSON-encoded string; failures are reported as
//! [`rmcp::ErrorData`] internal errors and recorded via telemetry (op per
//! tool name). Argument mapping is pure (free functions below) so it can be
//! unit-tested without a server, a database, or pipelines.
//!
//! [`Cli::recall_protocol`]: mnemos_cli::Cli::recall_protocol

use std::sync::Arc;
use std::time::Instant;

use helix_db::dsl::prelude::*;
use mnemos_cli::Cli;
use mnemos_contradiction::ContradictionDetector;
use mnemos_core::{EngramId, EngramType};
use mnemos_embedding_trait::EmbeddingProvider;
use mnemos_ml_trait::EmotionalTagger;
use mnemos_storage::Storage;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// `limit` used for `recall` when the caller omits it.
pub const DEFAULT_RECALL_LIMIT: usize = 5;

/// Hard cap for `recall` `limit` (oversample fetches `limit * 2`).
pub const MAX_RECALL_LIMIT: usize = 50;

/// Candidates fetched for `contradiction_check` (protocol doc: limit 5,
/// `min_importance` 0.0 — include all).
const CONTRADICTION_SEARCH_LIMIT: i64 = 5;

/// Vector search over `Engram.embedding` for the `contradiction_check` tool.
///
/// Same shape as `mnemos-retrieval`'s `search_engrams_full` (source-stage
/// `vector_search_nodes_with`, `Engram` nodes directly, no `HasVector`
/// traversal), plus the `embedding` projection the similarity filter needs
/// and `importance_score` for the report. There is no importance filter in
/// the query itself: `min_importance` 0.0 means every candidate is included.
#[query]
fn search_engrams_for_contradiction(query_embedding: Vec<f32>, limit: i64) {
    read_batch()
        .var_as(
            "candidates",
            g()
                .vector_search_nodes_with("Engram", "embedding", query_embedding, limit, None)
                .value_map(Some(vec![
                    "$id",
                    "episode_raw",
                    "emotional_charge",
                    "importance_score",
                    "embedding",
                ])),
        )
        .returning(["candidates"])
}

/// One candidate row for the contradiction filter (tolerant: missing
/// properties default so partially projected rows still decode).
#[derive(Debug, Clone, Deserialize)]
struct ContradictionCandidate {
    #[serde(rename = "$id")]
    id: EngramId,
    #[serde(default)]
    episode_raw: String,
    #[serde(default)]
    emotional_charge: f64,
    #[serde(default)]
    importance_score: f64,
    #[serde(default)]
    embedding: Vec<f32>,
}

/// Deserialize the row array under `key` (tolerates a single object, missing,
/// or null bindings by yielding zero or one rows; unparseable rows are
/// skipped).
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

/// Effective `recall` limit: omitted → 5, otherwise capped at 50.
#[must_use]
pub fn effective_recall_limit(raw: Option<usize>) -> usize {
    raw.unwrap_or(DEFAULT_RECALL_LIMIT).min(MAX_RECALL_LIMIT)
}

/// Map a `store` `importance` string to an explicit score.
///
/// `None` / `"auto"` → `None` (the pipeline scores it); `"low"` → 0.2,
/// `"medium"` → 0.5, `"high"` → 0.85 (per the protocol doc); anything else →
/// `Some(0.5)` (neutral, matching the doc's fallback).
#[must_use]
pub fn map_importance(raw: Option<&str>) -> Option<f64> {
    match raw {
        None => None,
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "auto" => None,
            "low" => Some(0.2),
            "medium" => Some(0.5),
            "high" => Some(0.85),
            _ => Some(0.5),
        },
    }
}

/// Parse a `store` `type` string to [`EngramType`].
///
/// `None` / `"auto"` → [`EngramType::Episodic`]; otherwise parses
/// `episodic` / `semantic` / `procedural` case-insensitively, falling back
/// to [`EngramType::Episodic`] on unknown values.
#[must_use]
pub fn parse_engram_type(raw: Option<&str>) -> EngramType {
    match raw {
        None => EngramType::Episodic,
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "auto" | "episodic" => EngramType::Episodic,
            "semantic" => EngramType::Semantic,
            "procedural" => EngramType::Procedural,
            _ => EngramType::Episodic,
        },
    }
}

/// Backend failure mapped to an MCP internal error.
fn internal(err: impl std::fmt::Display) -> McpError {
    McpError::internal_error(err.to_string(), None)
}

/// Record one tool outcome with latency.
fn record_tool(op: &'static str, ok: bool, detail: &str, latency_ms: u64) {
    mnemos_telemetry::global().record_with_latency("mnemos-mcp-protocol", op, ok, detail, latency_ms);
}

/// Params for `recall`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct RecallParams {
    /// Natural-language description of what to remember.
    query: String,
    /// Max results (default 5, capped at 50).
    #[serde(default)]
    limit: Option<usize>,
    /// Memory scope filter (`semantic` / `temporal` / `auto`). Accepted for
    /// future use; does not change retrieval today.
    #[serde(default, rename = "type")]
    memory_type: Option<String>,
    /// Time range (`1d` / `7d` / `30d` / `1y` / `all`). Accepted for future
    /// use; does not change retrieval today.
    #[serde(default)]
    since: Option<String>,
}

/// Params for `store`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct StoreParams {
    /// What to remember (self-contained: understandable months from now).
    content: String,
    /// `low` (0.2) / `medium` (0.5) / `high` (0.85) / `auto` (system scores).
    #[serde(default)]
    importance: Option<String>,
    /// `episodic` / `semantic` / `procedural` / `auto` (→ episodic).
    #[serde(default, rename = "type")]
    memory_type: Option<String>,
}

/// Params for `contradiction_check`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ContradictionCheckParams {
    /// The claim about to be made.
    claim: String,
}

/// Params for `consolidate`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ConsolidateParams {
    /// Run full mitosis + contradiction resolution (default false).
    #[serde(default)]
    aggressive: Option<bool>,
}

/// MCP server exposing the 4 Tool-Call Protocol tools.
///
/// Owns everything `contradiction_check` needs — `detector` + `embedder` +
/// `tagger` + `storage` — plus the shared [`Cli`] for recall/consolidation.
/// (`Cli` itself owns no embedder/tagger/detector directly, so the check
/// lives here rather than on [`Cli`].)
pub struct ProtocolTools {
    cli: Arc<Cli>,
    detector: ContradictionDetector,
    embedder: Box<dyn EmbeddingProvider>,
    storage: Storage,
    tagger: Box<dyn EmotionalTagger>,
    tool_router: ToolRouter<Self>,
}

impl ProtocolTools {
    /// Assemble the tool server. `storage` backs the `contradiction_check`
    /// vector search; `tagger` supplies the claim's emotional tone for the
    /// opposite-sign filter.
    #[must_use]
    pub fn new(
        cli: Arc<Cli>,
        detector: ContradictionDetector,
        embedder: Box<dyn EmbeddingProvider>,
        storage: Storage,
        tagger: Box<dyn EmotionalTagger>,
    ) -> Self {
        Self {
            cli,
            detector,
            embedder,
            storage,
            tagger,
            tool_router: Self::tool_router(),
        }
    }

    /// `contradiction_check` implementation: embed the claim, vector-search
    /// candidates (limit 5, `min_importance` 0.0), keep opposite-sign pairs
    /// with similarity > 0.7 via
    /// [`ContradictionDetector::embedding_candidate`], verify the survivors
    /// with [`ContradictionDetector::verify_pair`], and format the report.
    async fn contradiction_check_inner(&self, claim: &str) -> mnemos_core::Result<String> {
        // Step 1: claim embedding + tone, in parallel with nothing else
        // (both are cheap provider calls; sequential keeps borrowck simple).
        let claim_embedding = self.embedder.embed(claim).await?;
        let claim_tone = self.tagger.tag(claim).await?;

        // Step 2: vector-search candidates (limit 5, min_importance 0.0).
        let request = search_engrams_for_contradiction(claim_embedding.clone(), CONTRADICTION_SEARCH_LIMIT)
            .map_err(|e| mnemos_core::MnemosError::Storage(format!("build search: {e}")))?;
        let response: serde_json::Value = self
            .storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| mnemos_core::MnemosError::Storage(format!("search: {e}")))?;
        let candidates: Vec<ContradictionCandidate> =
            response_rows(&response, "candidates");

        // Step 3: embedding pre-filter (opposite signs + sim > 0.7).
        let mut prefiltered: Vec<&ContradictionCandidate> = Vec::new();
        for candidate in &candidates {
            if ContradictionDetector::embedding_candidate(
                claim_tone,
                &claim_embedding,
                candidate.emotional_charge,
                &candidate.embedding,
            )
            .is_some()
            {
                prefiltered.push(candidate);
            }
        }

        // Step 4: LLM verification of the surviving pairs.
        let mut conflicts: Vec<serde_json::Value> = Vec::new();
        for candidate in prefiltered {
            let verdict = self.detector.verify_pair(claim, &candidate.episode_raw).await?;
            if verdict.contradicts {
                conflicts.push(serde_json::json!({
                    "engram_id": candidate.id,
                    "episode": candidate.episode_raw,
                    "emotional_charge": candidate.emotional_charge,
                    "importance": candidate.importance_score,
                    "explanation": verdict.explanation,
                }));
            }
        }

        // Step 5: formatted report or empty.
        if conflicts.is_empty() {
            Ok(serde_json::json!({"found": false, "message": "no contradictions"}).to_string())
        } else {
            Ok(serde_json::json!({"found": true, "count": conflicts.len(), "conflicts": conflicts})
                .to_string())
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ProtocolTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "MNEMOS Tool-Call Protocol: recall/store/contradiction_check/consolidate",
        )
    }
}

#[tool_router(router = tool_router)]
impl ProtocolTools {
    /// Access persistent memory about past conversations, decisions, and
    /// insights. Returns formatted memory snippets as JSON.
    #[tool(
        name = "recall",
        description = "Access persistent memory about past conversations, decisions, and insights. Use before answering questions that depend on context. Parameters: query (required), limit (optional, default 5, max 50), type (optional: semantic/temporal/auto, reserved for future filtering), since (optional: 1d/7d/30d/1y/all, reserved for future filtering)."
    )]
    async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<String, McpError> {
        let start = Instant::now();
        let limit = effective_recall_limit(params.limit);
        // `type`/`since` are accepted but reserved for future filtering.
        let _ = (params.memory_type.as_deref(), params.since.as_deref());
        let out = self.cli.recall_protocol(&params.query, limit).await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match &out {
            Ok(memories) => {
                record_tool("recall", true, "", ms);
                // Ledger is on by default — fetch the recall_id just recorded for parallel-safe reward.
                // Exposed as a top-level field so every transport (stdio + :4545/mcp) passes it identically.
                let recall_id = self.cli.last_recall_id().await;
                serde_json::to_string(&serde_json::json!({
                    "query": params.query,
                    "limit": limit,
                    "memories": memories,
                    "recall_id": recall_id,
                }))
                .map_err(internal)
            }
            Err(e) => {
                record_tool("recall", false, &e.to_string(), ms);
                Err(internal(e))
            }
        }
    }

    /// Preserve important information for future recall. Returns the new
    /// engram id as JSON.
    #[tool(
        name = "store",
        description = "Preserve important information for future recall. Use after significant insights, preferences, or decisions. Parameters: content (required), importance (optional: low/medium/high/auto, default auto), type (optional: episodic/semantic/procedural/auto, default auto)."
    )]
    async fn store(
        &self,
        Parameters(params): Parameters<StoreParams>,
    ) -> Result<String, McpError> {
        let start = Instant::now();
        let importance = map_importance(params.importance.as_deref());
        let engram_type = parse_engram_type(params.memory_type.as_deref());
        let out = self.cli.ingest_full(&params.content, engram_type, importance).await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match out {
            Ok(id) => {
                record_tool("store", true, "", ms);
                serde_json::to_string(&serde_json::json!({ "engram_id": id })).map_err(internal)
            }
            Err(e) => {
                record_tool("store", false, &e.to_string(), ms);
                Err(internal(e))
            }
        }
    }

    /// Check whether conflicting memories exist before stating a claim as
    /// fact. Returns a contradiction report as JSON.
    #[tool(
        name = "contradiction_check",
        description = "Before making a factual claim or giving advice that could contradict past information, check for conflicting memories. Parameters: claim (required)."
    )]
    async fn contradiction_check(
        &self,
        Parameters(params): Parameters<ContradictionCheckParams>,
    ) -> Result<String, McpError> {
        let start = Instant::now();
        let out = self.contradiction_check_inner(&params.claim).await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match &out {
            Ok(report) => {
                record_tool("contradiction_check", true, "", ms);
                Ok(report.clone())
            }
            Err(e) => {
                record_tool("contradiction_check", false, &e.to_string(), ms);
                Err(internal(e))
            }
        }
    }

    /// Run memory maintenance (decay old memories, promote recalled ones).
    /// Returns `{ pruned, compressed, promoted }` as JSON.
    #[tool(
        name = "consolidate",
        description = "Run memory maintenance — decay old memories, promote frequently recalled ones. Call at end of session or periodically. Parameters: aggressive (optional boolean, default false — full mitosis and contradiction resolution)."
    )]
    async fn consolidate(
        &self,
        Parameters(params): Parameters<ConsolidateParams>,
    ) -> Result<String, McpError> {
        let start = Instant::now();
        let aggressive = params.aggressive.unwrap_or(false);
        let out = self.cli.consolidate_aggressive(aggressive).await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        match out {
            Ok(report) => {
                record_tool("consolidate", true, "", ms);
                serde_json::to_string(&serde_json::json!({
                    "pruned": report.pruned,
                    "compressed": report.compressed,
                    "promoted": report.promoted,
                }))
                .map_err(internal)
            }
            Err(e) => {
                record_tool("consolidate", false, &e.to_string(), ms);
                Err(internal(e))
            }
        }
    }
}

/// Serve the 4 protocol tools over stdio.
///
/// Blocks until the client disconnects. Called by the `mnemos-app` binary.
///
/// Transport/handler shape mirrors `mnemos-mcp-server::run`.
///
/// # Errors
///
/// Returns [`mnemos_core::MnemosError::Internal`] if the stdio transport
/// fails to start or the service task fails.
pub async fn run(tools: ProtocolTools) -> mnemos_core::Result<()> {
    let service = tools
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|err| mnemos_core::MnemosError::Internal(err.to_string()))?;
    service
        .waiting()
        .await
        .map_err(|err| mnemos_core::MnemosError::Internal(err.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recall_limit_defaults_to_five_when_omitted() {
        assert_eq!(DEFAULT_RECALL_LIMIT, 5);
        assert_eq!(effective_recall_limit(None), 5);
        let params: RecallParams = serde_json::from_value(json!({ "query": "redwood" }))
            .expect("query-only params must deserialize");
        assert_eq!(params.limit, None);
        assert_eq!(effective_recall_limit(params.limit), 5);
    }

    #[test]
    fn recall_limit_caps_at_fifty() {
        assert_eq!(MAX_RECALL_LIMIT, 50);
        assert_eq!(effective_recall_limit(Some(3)), 3);
        assert_eq!(effective_recall_limit(Some(50)), 50);
        assert_eq!(effective_recall_limit(Some(500)), 50);
    }

    #[test]
    fn recall_accepts_type_and_since_filters() {
        let params: RecallParams = serde_json::from_value(json!({
            "query": "q",
            "limit": 3,
            "type": "semantic",
            "since": "7d",
        }))
        .expect("full recall params must deserialize");
        assert_eq!(params.memory_type.as_deref(), Some("semantic"));
        assert_eq!(params.since.as_deref(), Some("7d"));
        assert_eq!(effective_recall_limit(params.limit), 3);
    }

    #[test]
    fn importance_maps_per_doc() {
        assert_eq!(map_importance(None), None);
        assert_eq!(map_importance(Some("auto")), None);
        assert_eq!(map_importance(Some("low")), Some(0.2));
        assert_eq!(map_importance(Some("medium")), Some(0.5));
        assert_eq!(map_importance(Some("high")), Some(0.85));
    }

    #[test]
    fn importance_unknown_falls_back_to_neutral() {
        assert_eq!(map_importance(Some("urgent")), Some(0.5));
    }

    #[test]
    fn engram_type_parses_per_doc() {
        assert_eq!(parse_engram_type(None), EngramType::Episodic);
        assert_eq!(parse_engram_type(Some("auto")), EngramType::Episodic);
        assert_eq!(parse_engram_type(Some("episodic")), EngramType::Episodic);
        assert_eq!(parse_engram_type(Some("semantic")), EngramType::Semantic);
        assert_eq!(parse_engram_type(Some("procedural")), EngramType::Procedural);
    }

    #[test]
    fn store_params_deserialize_with_optionals_missing() {
        let params: StoreParams =
            serde_json::from_value(json!({ "content": "the sky is blue" }))
                .expect("content-only params must deserialize");
        assert_eq!(map_importance(params.importance.as_deref()), None);
        assert_eq!(
            parse_engram_type(params.memory_type.as_deref()),
            EngramType::Episodic
        );
    }

    #[test]
    fn tool_attrs_match_protocol_names() {
        assert_eq!(
            ProtocolTools::recall_tool_attr().name.as_ref(),
            "recall"
        );
        assert_eq!(ProtocolTools::store_tool_attr().name.as_ref(), "store");
        assert_eq!(
            ProtocolTools::contradiction_check_tool_attr().name.as_ref(),
            "contradiction_check"
        );
        assert_eq!(
            ProtocolTools::consolidate_tool_attr().name.as_ref(),
            "consolidate"
        );
    }

    #[test]
    fn consolidate_aggressive_defaults_to_false() {
        let params: ConsolidateParams = serde_json::from_value(json!({}))
            .expect("empty consolidate params must deserialize");
        assert!(!params.aggressive.unwrap_or(false));
        let params: ConsolidateParams =
            serde_json::from_value(json!({ "aggressive": true }))
                .expect("explicit consolidate params must deserialize");
        assert_eq!(params.aggressive, Some(true));
    }

    #[test]
    fn contradiction_params_require_claim() {
        let ok: ContradictionCheckParams =
            serde_json::from_value(json!({ "claim": "cloud is best" }))
                .expect("claim params must deserialize");
        assert_eq!(ok.claim, "cloud is best");
        assert!(serde_json::from_value::<ContradictionCheckParams>(json!({})).is_err());
    }

    /// Manual smoke test: serve the protocol tools over stdio.
    ///
    /// Requires live providers (a `Cli`, detector LLM, embedder, tagger,
    /// storage), so there is nothing to construct here. Run explicitly via
    /// `cargo test -p mnemos-mcp-protocol -- --ignored` with a binary harness.
    #[ignore = "needs live providers backed by storage; run manually"]
    #[tokio::test]
    async fn serve_stdio_smoke() {
        // Intentionally empty: `run` blocks on stdio, so this documents the
        // manual harness rather than executing it in CI.
    }
}
