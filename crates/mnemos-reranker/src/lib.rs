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
    /// Number of preference pairs seen during training.
    #[serde(default)]
    pub trained_samples: u64,
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
