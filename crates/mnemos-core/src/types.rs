//! Canonical domain types. Field names mirror `HelixDB` property names
//! from `__reference/AGI-Memory-Research/helixdb-embedded-schema.md`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Unique id for an engram node (`HelixDB` `$id`).
pub type EngramId = u64;
/// Unique id for a concept node.
pub type ConceptId = u64;
/// Unique id for an identity node.
pub type IdentityId = u64;

/// Memory kind stored in `Engram.engram_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EngramType {
    #[default]
    Episodic,
    Semantic,
    Procedural,
}

impl EngramType {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
        }
    }
}

/// Core memory unit. Embedding lives directly on the node
/// (no separate vector node — see vector-direct-vs-separate-node doc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Engram {
    pub id: Option<EngramId>,
    pub episode_raw: String,
    /// -1.0 (very negative) to +1.0 (very positive).
    pub emotional_charge: f64,
    /// 0.0 to 1.0.
    pub importance_score: f64,
    /// Ebbinghaus decay parameter.
    pub decay_rate: f64,
    pub activation_count: i64,
    pub timestamp: DateTime<Utc>,
    pub engram_type: EngramType,
    /// 0 = raw, 1 = compressed, 2 = crystallized.
    pub compression_level: i64,
    pub contradiction_flag: bool,
    /// 1536-dim embedding, stored directly on node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// Knowledge concept linked from engrams via Recalls / `AbstractsTo`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id: Option<ConceptId>,
    pub name: String,
    pub confidence: f64,
    pub formation_date: DateTime<Utc>,
    pub source_count: i64,
    /// Optional centroid of recalling engrams' embeddings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// Personality trait fed by Concept via Defines edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub id: Option<IdentityId>,
    #[serde(rename = "trait")]
    pub trait_name: String,
    pub value: f64,
    /// Resistance to change (0.0-1.0).
    pub stability: f64,
    pub last_updated: DateTime<Utc>,
}

/// Concept extracted by LLM at ingestion time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedConcept {
    pub name: String,
    pub confidence: f64,
}

/// Candidate row returned from vector search, before CRR scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramCandidate {
    #[serde(rename = "$id")]
    pub id: EngramId,
    #[serde(default)]
    pub episode_raw: String,
    #[serde(default)]
    pub emotional_charge: f64,
    #[serde(default)]
    pub importance_score: f64,
    #[serde(default)]
    pub decay_rate: f64,
    #[serde(default)]
    pub activation_count: i64,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub engram_type: String,
    #[serde(default)]
    pub compression_level: i64,
    #[serde(default)]
    pub contradiction_flag: bool,
    #[serde(rename = "$distance", default)]
    pub distance: f64,
}

/// Five-factor CRR scoring output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceResult {
    pub engram_id: EngramId,
    pub resonance_score: f64,
    pub episode_raw: String,
    pub emotional_charge: f64,
    pub importance_score: f64,
    pub identity_alignment: f64,
    pub semantic_sim: f64,
    pub recency_weight: f64,
}

/// Aggregate counts for CLI stats / MCP tools.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStats {
    pub total_engrams: u64,
    pub contradictions: u64,
    pub concepts: u64,
    pub identities: u64,
}

/// Outcome of one consolidation ("sleep") cycle.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsolidationReport {
    pub pruned: u64,
    pub compressed: u64,
    pub promoted: u64,
    pub contradictions_linked: u64,
}

/// Embedding dimension used everywhere (`OpenAI` text-embedding-3-small).
pub const EMBEDDING_DIM: usize = 1536;

/// Compute cosine similarity in f64 precision.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let x = f64::from(a[i]);
        let y = f64::from(b[i]);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Mean-pool embeddings into a centroid.
#[must_use]
pub fn compute_centroid(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![0.0; EMBEDDING_DIM];
    }
    let dim = embeddings[0].len();
    let mut out = vec![0.0f32; dim];
    for emb in embeddings {
        for (i, v) in emb.iter().enumerate().take(dim) {
            out[i] += *v;
        }
    }
    let n = embeddings.len() as f32;
    for v in &mut out {
        *v /= n;
    }
    out
}

/// Parse RFC 3339 timestamp to unix seconds; 0.0 on failure.
#[must_use]
pub fn parse_timestamp_rfc3339(ts: &str) -> f64 {
    DateTime::parse_from_rfc3339(ts)
        .map_or(0.0, |dt| dt.timestamp() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0, 0.0, 0.5];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn centroid_averages() {
        let c = compute_centroid(&[vec![1.0, 3.0], vec![3.0, 1.0]]);
        assert!((c[0] - 2.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
    }
}
