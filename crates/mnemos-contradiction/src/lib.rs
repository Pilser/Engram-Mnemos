#![recursion_limit = "256"]
//! `mnemos-contradiction`: hybrid contradiction detection pipeline.
//!
//! See `__reference/AGI-Memory-Research/Explanation-docs/04-Contradiction-Detector.md`.
//!
//! Pipeline per concept:
//! 1. **Embedding filter** (fast, free, local): fetch engrams recalling the
//!    concept, keep pairs with opposite emotional signs, emotional gap
//!    `> 0.6`, and cosine similarity `> 0.70` ([`ContradictionDetector::embedding_candidate`]).
//! 2. **LLM verification** (accurate, few calls): each candidate pair is
//!    checked by [`ContradictionDetector::verify_pair`], which asks the LLM
//!    for `{"contradicts", "explanation"}` JSON with one repair retry.
//! 3. **Write-back**: confirmed pairs get a `Contradicts` edge plus
//!    `contradiction_flag = true` on both engrams (the CRR formula penalizes
//!    flagged engrams by `0.5x`).
//!
//! # `HelixDB` grounding (verified against helix-db 3.0.0 / helix-ast 0.1.0)
//!
//! * `#[query]` is imported via `helix_db::dsl::prelude::*` (re-exported
//!   from `helix_ast::prelude` plus `helix_dsl_macros::query`).
//! * `g().n(NodeRef::param(..))` binds node ids by request-parameter name;
//!   `#[query]` params support `i64` but NOT `u64`, so ids cross the query
//!   boundary as `i64` (`i64::try_from` at the call site).
//! * Concept -> engrams traversal: ingestion writes `Recalls` edges
//!   engram -> concept, so recalling engrams are the concept's incoming
//!   `Recalls` neighbors (`in_(Some("Recalls"))`; the label is a static AST
//!   string). `value_map` projects `$id` / `episode_raw` /
//!   `emotional_charge` / `embedding` (the embedding lives directly on the
//!   node).
//! * `Client::query` signature (helix-db `src/lib.rs`):
//!   `pub fn query<R: for<'de> Deserialize<'de>>(&self, request: QueryRequest)`
//!   with `async fn send(self) -> Result<R, HelixError>`; responses are
//!   decoded as `serde_json::Value`.
//! * No `HelixDbSource`, `Client::open`, or embedded APIs are used: all DB
//!   access goes through [`mnemos_storage::Storage::client`].

use helix_db::dsl::prelude::*;
use mnemos_core::{EngramId, MnemosError, Result, cosine_similarity};
use mnemos_llm_trait::LlmProvider;
use mnemos_storage::Storage;
use serde::{Deserialize, Serialize};

/// Opposite-sign emotional gap required between candidate engrams.
pub const EMOTIONAL_GAP_THRESHOLD: f64 = 0.6;
/// Minimum cosine similarity for two engrams to count as "same topic".
pub const SIMILARITY_THRESHOLD: f64 = 0.70;

/// Fetch engrams recalling a concept, with the fields the embedding filter
/// needs: `$id`, `episode_raw`, `emotional_charge`, and the on-node
/// `embedding` vector.
#[query]
fn engrams_recalling_concept(concept_id: i64) -> ReadBatch {
    // `concept_id` binds via `NodeRef::param("concept_id")` below; the
    // macro's `Expr` binding is only referenced here to silence
    // `unused_variables`.
    let _ = &concept_id;
    read_batch()
        .var_as(
            "engrams",
            g().n(NodeRef::param("concept_id"))
                .in_(Some("Recalls"))
                .value_map(Some(vec![
                    "$id",
                    "episode_raw",
                    "emotional_charge",
                    "embedding",
                ])),
        )
        .returning(["engrams"])
}

/// Create a `Contradicts` edge from one engram to another.
///
/// Edge labels are static AST strings, so the label is baked in (cf.
/// `mnemos-ingestion`'s per-label `connect_*_edge` fns).
#[query]
fn connect_contradicts_edge(from_id: i64, to_id: i64) -> WriteBatch {
    // See `engrams_recalling_concept` for why these bindings are referenced.
    let _ = (&from_id, &to_id);
    write_batch()
        .var_as(
            "edge",
            g().n(NodeRef::param("from_id")).add_e(
                "Contradicts",
                NodeRef::param("to_id"),
                Vec::<(String, PropertyInput)>::new(),
            ),
        )
        .returning(["edge"])
}

/// Set `contradiction_flag = true` on one engram.
#[query]
fn set_contradiction_flag(engram_id: i64) -> WriteBatch {
    // See `engrams_recalling_concept` for why this binding is referenced.
    let _ = &engram_id;
    write_batch()
        .var_as(
            "updated",
            g().n(NodeRef::param("engram_id"))
                .set_property("contradiction_flag", true),
        )
        .returning(["updated"])
}

/// LLM verdict on whether two statements contradict each other.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContradictionVerdict {
    pub contradicts: bool,
    pub explanation: String,
}

/// Hybrid contradiction detector: embedding filter plus LLM verification.
pub struct ContradictionDetector {
    llm: Box<dyn LlmProvider>,
}

impl ContradictionDetector {
    /// Assemble the detector over an LLM provider.
    #[must_use]
    pub fn new(llm: Box<dyn LlmProvider>) -> Self {
        Self { llm }
    }

    /// Pure embedding pre-filter: `Some(cosine_sim)` iff the emotional
    /// charges have different signs, their absolute gap exceeds `0.6`, and
    /// the embedding cosine similarity exceeds `0.70`; else `None`.
    ///
    /// A charge of exactly `0.0` is its own side: `0.0` vs. any nonzero
    /// charge counts as opposite signs.
    #[must_use]
    pub fn embedding_candidate(
        a_emo: f64,
        a_emb: &[f32],
        b_emo: f64,
        b_emb: &[f32],
    ) -> Option<f64> {
        if a_emo.signum() == b_emo.signum() {
            return None;
        }
        if (a_emo - b_emo).abs() <= EMOTIONAL_GAP_THRESHOLD {
            return None;
        }
        let sim = cosine_similarity(a_emb, b_emb);
        if sim <= SIMILARITY_THRESHOLD {
            return None;
        }
        Some(sim)
    }

    /// Ask the LLM whether two statements contradict each other about the
    /// same specific topic.
    ///
    /// Up to 2 attempts: a parse failure re-asks with a repair hint quoting
    /// the previous reply. Each parse failure is recorded via telemetry
    /// (`mnemos-contradiction` / `verify_pair`).
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError`] when the LLM call fails or when neither
    /// attempt yields parseable `{"contradicts", "explanation"}` JSON.
    pub async fn verify_pair(
        &self,
        text_a: &str,
        text_b: &str,
    ) -> Result<ContradictionVerdict> {
        let base = format!(
            "Compare these two statements. Do they contradict each other \
             about the same specific topic, or are they about different aspects? \
             Return JSON: {{\"contradicts\": true/false, \"explanation\": \"why or why not\"}}\n\n\
             Statement A: {text_a}\n\
             Statement B: {text_b}",
        );
        let mut prompt = base.clone();
        let mut last_err = String::new();
        for attempt in 1..=2 {
            let raw = self.llm.chat(&prompt).await?;
            match serde_json::from_str::<ContradictionVerdict>(extract_json(&raw)) {
                Ok(verdict) => return Ok(verdict),
                Err(e) => {
                    last_err = e.to_string();
                    mnemos_telemetry::global().record(
                        "mnemos-contradiction",
                        "verify_pair",
                        false,
                        &format!("attempt {attempt} parse failed: {last_err}"),
                    );
                    prompt = format!(
                        "{base}\n\nYour previous response was not valid JSON ({last_err}). \
                         Reply with ONLY a JSON object like \
                         {{\"contradicts\": true, \"explanation\": \"...\"}} and no other text.\n\
                         Previous response:\n{raw}"
                    );
                }
            }
        }
        Err(MnemosError::Llm(format!(
            "verify_pair: no parseable verdict after 2 attempts: {last_err}"
        )))
    }

    /// Scan one concept for contradictions (hybrid pipeline).
    ///
    /// Fetches engrams recalling `concept_id`, applies
    /// [`Self::embedding_candidate`] pairwise, verifies each candidate with
    /// [`Self::verify_pair`], and for every confirmed pair writes a
    /// `Contradicts` edge plus `contradiction_flag = true` on both engrams.
    /// Returns the confirmed `(EngramId, EngramId)` pairs. The scan outcome
    /// (engram / candidate / confirmed counts) is recorded via telemetry
    /// (`mnemos-contradiction` / `scan_concept`).
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when id conversion, query building,
    /// or any HelixDB request fails, and propagates [`Self::verify_pair`]
    /// LLM failures.
    pub async fn scan_concept(
        &self,
        storage: &Storage,
        concept_id: u64,
    ) -> Result<Vec<(EngramId, EngramId)>> {
        let request = engrams_recalling_concept(node_param(concept_id)?)
            .map_err(|e| MnemosError::Storage(format!("build engrams_recalling_concept: {e}")))?;
        let response: serde_json::Value = storage
            .client()
            .query(request)
            .send()
            .await
            .map_err(|e| MnemosError::Storage(format!("engrams_recalling_concept: {e}")))?;
        let rows: Vec<EngramRow> = response_rows(&response, "engrams");

        // Step 1: embedding filter (fast, free) narrows pairs to candidates.
        let mut candidate_pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                let a = &rows[i];
                let b = &rows[j];
                if Self::embedding_candidate(
                    a.emotional_charge,
                    &a.embedding,
                    b.emotional_charge,
                    &b.embedding,
                )
                .is_some()
                {
                    candidate_pairs.push((i, j));
                }
            }
        }
        let candidate_count = candidate_pairs.len();

        // Step 2: LLM verification (accurate, few calls) + write-back.
        let mut confirmed: Vec<(EngramId, EngramId)> = Vec::new();
        for (i, j) in candidate_pairs {
            let verdict = self
                .verify_pair(&rows[i].episode_raw, &rows[j].episode_raw)
                .await?;
            if verdict.contradicts {
                let a = rows[i].id;
                let b = rows[j].id;
                let request = connect_contradicts_edge(node_param(a)?, node_param(b)?)
                    .map_err(|e| {
                        MnemosError::Storage(format!("build connect_contradicts_edge: {e}"))
                    })?;
                let _: serde_json::Value = storage
                    .client()
                    .query(request)
                    .send()
                    .await
                    .map_err(|e| {
                        MnemosError::Storage(format!("connect_contradicts_edge: {e}"))
                    })?;
                for id in [a, b] {
                    let request = set_contradiction_flag(node_param(id)?).map_err(|e| {
                        MnemosError::Storage(format!("build set_contradiction_flag: {e}"))
                    })?;
                    let _: serde_json::Value = storage
                        .client()
                        .query(request)
                        .send()
                        .await
                        .map_err(|e| {
                            MnemosError::Storage(format!("set_contradiction_flag: {e}"))
                        })?;
                }
                confirmed.push((a, b));
            }
        }

        mnemos_telemetry::global().record(
            "mnemos-contradiction",
            "scan_concept",
            true,
            &format!(
                "concept={concept_id} engrams={} candidates={candidate_count} confirmed={}",
                rows.len(),
                confirmed.len()
            ),
        );
        Ok(confirmed)
    }
}

/// One engram row from `engrams_recalling_concept` (tolerant: missing
/// properties default so partially projected rows still decode).
#[derive(Debug, Clone, Deserialize)]
struct EngramRow {
    #[serde(rename = "$id")]
    id: EngramId,
    #[serde(default)]
    episode_raw: String,
    #[serde(default)]
    emotional_charge: f64,
    #[serde(default)]
    embedding: Vec<f32>,
}

/// Convert a `u64` node id to the `i64` query param the `#[query]` macro
/// accepts (the macro rejects `u64`; HelixDB ids always fit in `i64`).
fn node_param(id: u64) -> Result<i64> {
    i64::try_from(id).map_err(|_| MnemosError::Storage(format!("node id out of range: {id}")))
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

/// Slice the first `{`…`}` span so prose-wrapped JSON still parses.
fn extract_json(raw: &str) -> &str {
    match raw.find('{').zip(raw.rfind('}')) {
        Some((start, end)) => &raw[start..=end],
        None => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Fixed-reply LLM. Hand-written `chat` impls so unit tests need no
    /// extra async plumbing beyond `async-trait` (already a dependency).
    struct MockLlm {
        replies: std::sync::Mutex<Vec<String>>,
    }

    impl MockLlm {
        fn new(replies: Vec<String>) -> Self {
            Self {
                replies: std::sync::Mutex::new(replies),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockLlm {
        async fn chat(&self, _prompt: &str) -> Result<String> {
            let mut replies = self.replies.lock().expect("mock replies lock");
            assert!(!replies.is_empty(), "mock LLM ran out of replies");
            Ok(replies.remove(0))
        }

        async fn chat_with_system(&self, _system: &str, user: &str) -> Result<String> {
            self.chat(user).await
        }
    }

    /// Remote-work example from doc 04: gap 1.35, sim 0.89 → candidate.
    /// `b` is unit-norm with first component 0.89 (cosine vs `a` = 0.89).
    #[test]
    fn remote_work_pair_is_candidate() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.89f32, 0.4559];
        let sim = ContradictionDetector::embedding_candidate(0.7, &a, -0.65, &b);
        assert!(sim.is_some(), "expected Some, got None");
        assert!((sim.expect("some") - 0.89).abs() < 1e-3);
    }

    /// Coffee example from doc 04: opposite-side charges (0.8 vs 0.0) and a
    /// wide gap, but sim 0.42 < 0.70 → not a contradiction.
    #[test]
    fn coffee_pair_rejected_on_low_similarity() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.42f32, 0.9075];
        assert_eq!(
            ContradictionDetector::embedding_candidate(0.8, &a, 0.0, &b),
            None
        );
    }

    #[test]
    fn same_sign_is_rejected() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0];
        assert_eq!(
            ContradictionDetector::embedding_candidate(0.7, &a, 0.4, &b),
            None
        );
    }

    #[test]
    fn narrow_gap_is_rejected() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0];
        // Opposite signs but |0.3 - (-0.2)| = 0.5 <= 0.6.
        assert_eq!(
            ContradictionDetector::embedding_candidate(0.3, &a, -0.2, &b),
            None
        );
    }

    #[test]
    fn boundary_gap_is_rejected() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0];
        // Gap exactly 0.6 does not clear the `> 0.6` threshold.
        assert_eq!(
            ContradictionDetector::embedding_candidate(0.3, &a, -0.3, &b),
            None
        );
    }

    #[tokio::test]
    async fn verify_pair_parses_llm_json() {
        let llm = MockLlm::new(vec![
            r#"{"contradicts": true, "explanation": "A praises remote work while B blames it for killing collaboration."}"#
                .to_string(),
        ]);
        let detector = ContradictionDetector::new(Box::new(llm));
        let verdict = detector
            .verify_pair(
                "The team delivered 40% more features after going remote.",
                "Remote work has made collaboration impossible.",
            )
            .await
            .expect("verify_pair parses mock JSON");
        assert!(verdict.contradicts);
        assert!(verdict.explanation.contains("collaboration"));
    }

    #[tokio::test]
    async fn verify_pair_retries_after_prose_wrapped_reply() {
        let llm = MockLlm::new(vec![
            "not json at all".to_string(),
            r#"{"contradicts": false, "explanation": "different aspects of the topic."}"#
                .to_string(),
        ]);
        let detector = ContradictionDetector::new(Box::new(llm));
        let verdict = detector
            .verify_pair("statement one", "statement two")
            .await
            .expect("verify_pair succeeds on retry");
        assert!(!verdict.contradicts);
    }

    #[test]
    fn query_builders_produce_typed_requests() {
        // No live DB: proves the #[query] wiring (params, kind, name).
        let req = engrams_recalling_concept(7).expect("builds");
        assert_eq!(req.query_name(), Some("engrams_recalling_concept"));
        assert!(matches!(
            req.request_type(),
            helix_db::QueryRequestType::Read
        ));

        let req = connect_contradicts_edge(1, 2).expect("builds");
        assert_eq!(req.query_name(), Some("connect_contradicts_edge"));
        assert!(matches!(
            req.request_type(),
            helix_db::QueryRequestType::Write
        ));

        let req = set_contradiction_flag(3).expect("builds");
        assert_eq!(req.query_name(), Some("set_contradiction_flag"));
        assert!(matches!(
            req.request_type(),
            helix_db::QueryRequestType::Write
        ));
    }

    /// Live end-to-end scan: embedding filter → LLM verify → `Contradicts`
    /// edge + flags. Requires a running HelixDB with the MNEMOS schema and
    /// a concept with recalling engrams.
    #[tokio::test]
    #[ignore = "needs live HelixDB at HELIX_URL (default http://localhost:6969)"]
    async fn scan_concept_against_live_db() {
        let llm = MockLlm::new(vec![
            r#"{"contradicts": true, "explanation": "live check"}"#.to_string(),
        ]);
        let detector = ContradictionDetector::new(Box::new(llm));
        let storage = Storage::from_config(&mnemos_core::StorageConfig::default())
            .await
            .expect("storage builds without network I/O");
        let pairs = detector
            .scan_concept(&storage, 1)
            .await
            .expect("scan against live DB");
        let _ = pairs;
    }
}
