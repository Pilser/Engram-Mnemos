//! `mnemos-reranker`: small learned reranker over CRR features (Option B).
//!
//! This is the *learning* layer that sits on top of the CRR base ("Option A").
//! It is deliberately tiny — a linear model (~10 weights, <1 KB) trained with
//! pairwise ranking loss on the feedback log (`feedback.jsonl`), so it can be
//! trained on the fly and served with no GPU.
//!
//! ## Why pairwise
//!
//! Feedback only labels what was *shown*, so pointwise training learns the
//! current ranking's position bias. Pairwise training uses the preference
//! signal directly: within one recall, a rewarded candidate should outrank a
//! non-rewarded one. That is far more robust with sparse labels.
//!
//! ## Safety
//!
//! The model is only *blended* into CRR when `alpha > 0` (env
//! `MNEMOS_RERANKER_ALPHA`, default `0.0`), so the base CRR keeps working until
//! a trained model has been validated offline. See [`blend`].

#![recursion_limit = "256"]

use serde::{Deserialize, Serialize};

/// Canonical feature order (must match [`features_from_result`] and the
/// trainer's vectorization). Length is the model input dimension.
pub const FEATURE_NAMES: [&str; 9] = [
    "semantic_sim",
    "recency_weight",
    "emotional_charge_abs",
    "importance_score",
    "identity_alignment",
    "reward_score",
    "reward_factor",
    "resonance_score",
    "position",
];

/// Model input dimension.
pub const FEATURE_DIM: usize = FEATURE_NAMES.len();

/// Feature vector for one recalled engram.
#[must_use]
pub fn features_from_result(r: &mnemos_core::ResonanceResult, position: usize) -> Vec<f64> {
    vec![
        r.semantic_sim,
        r.recency_weight,
        r.emotional_charge.abs(),
        r.importance_score,
        r.identity_alignment,
        r.reward_score,
        r.reward_factor,
        r.resonance_score,
        position as f64,
    ]
}

/// One preference pair: `positive` should outrank `negative`.
pub type PreferencePair = (Vec<f64>, Vec<f64>);

/// Linear reranker: `sigmoid(w · x + b)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerModel {
    /// Format version.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Number of preference pairs seen during batch training.
    #[serde(default)]
    pub trained_samples: u64,
    /// Number of reward events applied online (drives the auto-alpha ramp).
    #[serde(default)]
    pub reward_events: u64,
    /// Weights in [`FEATURE_NAMES`] order.
    pub weights: Vec<f64>,
    /// Bias term.
    #[serde(default)]
    pub bias: f64,
}

fn default_version() -> u32 {
    1
}

impl Default for RerankerModel {
    fn default() -> Self {
        Self {
            version: default_version(),
            trained_samples: 0,
            reward_events: 0,
            weights: vec![0.0; FEATURE_DIM],
            bias: 0.0,
        }
    }
}

impl RerankerModel {
    /// Untrained (neutral) model: every score is `0.5`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Shipped seed prior (used when no locally-trained model exists).
    ///
    /// Encodes only the obvious monotonicities so a fresh install has a
    /// sensible starting point: higher semantic similarity, higher learned
    /// reward, and earlier rank are more relevant. Local fine-tuning replaces
    /// it as soon as feedback exists.
    #[must_use]
    pub fn seed() -> Self {
        // Order: semantic, recency, |emotion|, importance, identity,
        //        reward_score, reward_factor, resonance, position
        Self {
            version: default_version(),
            trained_samples: 0,
            reward_events: 0,
            weights: vec![3.0, 0.5, 0.0, 0.5, 0.5, 2.0, 1.0, 1.0, -0.2],
            bias: -3.0,
        }
    }

    /// Record one reward event (once per reward, not per candidate).
    pub fn record_reward(&mut self) {
        self.reward_events = self.reward_events.saturating_add(1);
    }

    /// Resolve the blend alpha.
    ///
    /// `raw = "auto"` (or unset) ramps linearly with reward events up to
    /// `max_alpha` at `min_pairs` events, so the system tunes itself and the
    /// agent never manages it. A numeric `raw` is a fixed override.
    #[must_use]
    pub fn resolve_alpha(raw: Option<&str>, reward_events: u64, min_pairs: u64, max_alpha: f64) -> f64 {
        let mode = raw.map(str::trim).filter(|s| !s.is_empty()).unwrap_or("auto");
        if mode.eq_ignore_ascii_case("auto") {
            if min_pairs == 0 {
                return max_alpha.clamp(0.0, 1.0);
            }
            let progress = (reward_events as f64) / (min_pairs as f64);
            progress.clamp(0.0, 1.0) * max_alpha.clamp(0.0, 1.0)
        } else {
            mode.parse::<f64>()
                .ok()
                .filter(|v| (0.0..=1.0).contains(v))
                .unwrap_or(0.0)
        }
    }

    /// One online logistic SGD step: push the score toward the reward.
    ///
    /// `reward` is the same `-1..1` scale as the reward tool; it maps to a
    /// `0..1` label. Weights are clamped so a burst of feedback cannot blow
    /// the model up.
    pub fn online_update(&mut self, features: &[f64], reward: f64, lr: f64) {
        let label = ((reward.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0);
        let p = self.score(features);
        let grad = label - p;
        for (i, w) in self.weights.iter_mut().enumerate() {
            *w += lr * grad * features.get(i).copied().unwrap_or(0.0);
            *w = w.clamp(-10.0, 10.0);
        }
        self.bias = (self.bias + lr * grad).clamp(-20.0, 20.0);
    }

    /// Relevance probability in `(0, 1)`.
    #[must_use]
    pub fn score(&self, features: &[f64]) -> f64 {
        let mut z = self.bias;
        for (i, w) in self.weights.iter().enumerate() {
            z += w * features.get(i).copied().unwrap_or(0.0);
        }
        sigmoid(z)
    }

    /// Train with pairwise logistic (RankNet-style) loss.
    ///
    /// For each `(pos, neg)` pair the loss is
    /// `-log sigmoid(s_pos - s_neg)`; the gradient step moves weights toward
    /// the preferred example. `lr` is the learning rate, `epochs` the passes.
    pub fn train_pairwise(&mut self, pairs: &[PreferencePair], lr: f64, epochs: usize) {
        if pairs.is_empty() {
            return;
        }
        for _ in 0..epochs {
            for (pos, neg) in pairs {
                let s_pos = self.raw_score(pos);
                let s_neg = self.raw_score(neg);
                let p = sigmoid(s_pos - s_neg);
                // d/dw of -log sigmoid(s_pos - s_neg) = -(1 - p) * (pos - neg)
                let grad = 1.0 - p;
                for i in 0..self.weights.len() {
                    let x_pos = pos.get(i).copied().unwrap_or(0.0);
                    let x_neg = neg.get(i).copied().unwrap_or(0.0);
                    self.weights[i] += lr * grad * (x_pos - x_neg);
                }
                self.bias += lr * grad;
            }
            self.trained_samples = self.trained_samples.saturating_add(pairs.len() as u64);
        }
    }

    /// Raw (pre-sigmoid) score.
    fn raw_score(&self, features: &[f64]) -> f64 {
        let mut z = self.bias;
        for (i, w) in self.weights.iter().enumerate() {
            z += w * features.get(i).copied().unwrap_or(0.0);
        }
        z
    }

    /// Load from a JSON file; `None` when absent or unparsable.
    #[must_use]
    pub fn load(path: &str) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str::<Self>(&data).ok()
    }

    /// Save to a JSON file atomically (tmp → rename).
    ///
    /// # Errors
    ///
    /// Returns the IO/serialization error string on failure.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
}

/// Sigmoid with overflow guard.
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-60.0, 60.0)).exp())
}

/// Feature vector for one logged candidate (must match [`FEATURE_NAMES`] order).
#[must_use]
pub fn features_from_log(r: &serde_json::Value, position: usize) -> Vec<f64> {
    let g = |k: &str| r.get(k).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
    vec![
        g("semantic_sim"),
        g("recency_weight"),
        g("emotional_charge").abs(),
        g("importance_score"),
        g("identity_alignment"),
        g("reward_score"),
        g("reward_factor"),
        g("resonance_score"),
        position as f64,
    ]
}

/// Batch-train the reranker from a feedback JSONL log.
///
/// Labels are per-recall (one scalar reward covers the whole shown set), so
/// preference pairs are built *across* recalls: candidates from positively
/// rewarded recalls should outrank candidates from negatively rewarded ones.
/// Saves the model and returns a summary. Shared by the CLI (`train-reranker`)
/// and the HTTP mirror endpoint.
///
/// # Errors
///
/// Returns a message when the log cannot be read, both signs are absent, or
/// the model cannot be saved.
pub fn train_from_log(log_path: &str, model_path: &str) -> Result<serde_json::Value, String> {
    use std::collections::HashMap;
    let data = std::fs::read_to_string(log_path)
        .map_err(|e| format!("cannot read feedback log {log_path}: {e}"))?;
    let mut candidates: HashMap<u64, Vec<Vec<f64>>> = HashMap::new();
    let mut rewards: HashMap<u64, f64> = HashMap::new();
    for line in data.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match v.get("kind").and_then(serde_json::Value::as_str) {
            Some("recall") => {
                let Some(rid) = v.get("recall_id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                let rows: Vec<Vec<f64>> = v
                    .get("results")
                    .and_then(serde_json::Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .enumerate()
                            .map(|(i, r)| features_from_log(r, i))
                            .collect()
                    })
                    .unwrap_or_default();
                candidates.entry(rid).or_default().extend(rows);
            }
            Some("reward") => {
                if let (Some(rid), Some(rew)) = (
                    v.get("recall_id").and_then(serde_json::Value::as_u64),
                    v.get("reward").and_then(serde_json::Value::as_f64),
                ) {
                    rewards.insert(rid, rew);
                }
            }
            _ => {}
        }
    }
    const POS: f64 = 0.2;
    const NEG: f64 = -0.2;
    let positives: Vec<&Vec<f64>> = candidates
        .iter()
        .filter(|(rid, _)| rewards.get(rid).is_some_and(|r| *r > POS))
        .flat_map(|(_, rows)| rows.iter())
        .collect();
    let negatives: Vec<&Vec<f64>> = candidates
        .iter()
        .filter(|(rid, _)| rewards.get(rid).is_some_and(|r| *r < NEG))
        .flat_map(|(_, rows)| rows.iter())
        .collect();
    if positives.is_empty() || negatives.is_empty() {
        return Err(format!(
            "not enough feedback yet (positive recalls={}, negative recalls={}); need both signs",
            candidates
                .keys()
                .filter(|rid| rewards.get(rid).is_some_and(|r| *r > POS))
                .count(),
            candidates
                .keys()
                .filter(|rid| rewards.get(rid).is_some_and(|r| *r < NEG))
                .count(),
        ));
    }
    const MAX_PAIRS: usize = 200_000;
    let mut pairs: Vec<PreferencePair> = Vec::new();
    'outer: for p in &positives {
        for n in &negatives {
            pairs.push(((*p).clone(), (*n).clone()));
            if pairs.len() >= MAX_PAIRS {
                break 'outer;
            }
        }
    }
    let mut model = RerankerModel::load(model_path).unwrap_or_default();
    model.train_pairwise(&pairs, 0.05, 20);
    model.save(model_path)?;
    Ok(serde_json::json!({
        "model": model_path,
        "pairs": pairs.len(),
        "positive_candidates": positives.len(),
        "negative_candidates": negatives.len(),
        "trained_samples": model.trained_samples,
        "alpha_env": "MNEMOS_RERANKER_ALPHA (auto = ramp; 0.0 = shadow)",
    }))
}

/// Blend a CRR score with a reranker probability.
///
/// `p = 0.5` is neutral (returns `crr`); `p = 1.0` doubles it; `p = 0.0`
/// scales it by `(1 - alpha)`. With `alpha = 0.0` this is exactly `crr`, so
/// the base behaviour is preserved until a validated model is enabled.
#[must_use]
pub fn blend(crr: f64, p: f64, alpha: f64) -> f64 {
    let alpha = alpha.clamp(0.0, 1.0);
    crr * (1.0 - alpha + alpha * 2.0 * p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(semantic: f64, reward_score: f64, reward_factor: f64) -> mnemos_core::ResonanceResult {
        mnemos_core::ResonanceResult {
            engram_id: 1,
            resonance_score: semantic,
            episode_raw: "x".to_string(),
            emotional_charge: 0.0,
            importance_score: 0.5,
            identity_alignment: 1.0,
            semantic_sim: semantic,
            recency_weight: 1.0,
            reward_score,
            reward_factor,
        }
    }

    #[test]
    fn features_have_canonical_dim() {
        let f = features_from_result(&result(0.9, 0.0, 1.0), 3);
        assert_eq!(f.len(), FEATURE_DIM);
    }

    #[test]
    fn untrained_model_is_neutral() {
        let m = RerankerModel::new();
        assert!((m.score(&[0.5; FEATURE_DIM]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn alpha_zero_preserves_crr() {
        assert!((blend(0.7, 0.9, 0.0) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn neutral_probability_preserves_crr() {
        assert!((blend(0.7, 0.5, 1.0) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn pairwise_training_learns_preference() {
        // Positive has high semantic, negative low; the model should learn it.
        let pos = features_from_result(&result(0.95, 0.0, 1.0), 0);
        let neg = features_from_result(&result(0.30, 0.0, 1.0), 1);
        let pairs = vec![(pos.clone(), neg.clone()); 200];
        let mut m = RerankerModel::new();
        m.train_pairwise(&pairs, 0.1, 50);
        assert!(
            m.score(&pos) > m.score(&neg),
            "trained model must prefer the positive: {} vs {}",
            m.score(&pos),
            m.score(&neg)
        );
    }

    #[test]
    fn round_trips_through_json() {
        let mut m = RerankerModel::new();
        m.weights[0] = 0.5;
        let json = serde_json::to_string(&m).unwrap();
        let back: RerankerModel = serde_json::from_str(&json).unwrap();
        assert_eq!(back.weights[0], 0.5);
    }
}
