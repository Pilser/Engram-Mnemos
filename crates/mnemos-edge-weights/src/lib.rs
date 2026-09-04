//! Learnable edge weights with Adam optimizer.
//!
//! Nine scalar weights would govern stimulation; with direct-on-node vector
//! storage the `HasVector` bridge weight is eliminated, leaving 8:
//! 7 edge-type weights + recurrent re-injection weight.
//! See `__reference/.../Explanation-docs/08-Learnable-Edge-Weights.md`.

use serde::{Deserialize, Serialize};

/// Index of each weight in [`EdgeWeights::as_array`] / attribution slices.
pub const IDX_RECALLS: usize = 0;
/// Index of each weight in [`EdgeWeights::as_array`] / attribution slices.
pub const IDX_ABSTRACTS_TO: usize = 1;
/// Index of each weight in [`EdgeWeights::as_array`] / attribution slices.
pub const IDX_REINFORCES: usize = 2;
/// Index of each weight in [`EdgeWeights::as_array`] / attribution slices.
pub const IDX_TEMPORAL_SEQ: usize = 3;
/// Index of each weight in [`EdgeWeights::as_array`] / attribution slices.
pub const IDX_CONTRADICTS: usize = 4;
/// Index of each weight in [`EdgeWeights::as_array`] / attribution slices.
pub const IDX_DEFINES: usize = 5;
/// Index of each weight in [`EdgeWeights::as_array`] / attribution slices.
pub const IDX_SPAWNED_FROM: usize = 6;
/// Index of each weight in [`EdgeWeights::as_array`] / attribution slices.
pub const IDX_RECURRENT: usize = 7;

/// Learnable association strengths + Adam optimizer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeWeights {
    /// Engram → Concept association.
    pub recalls: f64,
    /// Engram → Concept abstraction.
    pub abstracts_to: f64,
    /// Engram ↔ Engram mutual support.
    pub reinforces: f64,
    /// Engram → Engram time chain.
    pub temporal_seq: f64,
    /// Engram ↔ Engram suppression (negative).
    pub contradicts: f64,
    /// Concept → Identity (slow path).
    pub defines: f64,
    /// Parent → Child hierarchy.
    pub spawned_from: f64,
    /// Re-injection weight per feedback iteration.
    pub recurrent: f64,
    /// Adam momentum per weight.
    pub m: [f64; 8],
    /// Adam velocity per weight.
    pub v: [f64; 8],
    /// Adam timestep.
    pub t: u64,
}

impl Default for EdgeWeights {
    fn default() -> Self {
        Self::defaults()
    }
}

impl EdgeWeights {
    /// Default weights before any learning.
    #[must_use]
    pub fn defaults() -> Self {
        Self {
            recalls: 0.70,
            abstracts_to: 0.50,
            reinforces: 0.60,
            temporal_seq: 0.30,
            contradicts: -0.40,
            defines: 0.20,
            spawned_from: 0.45,
            recurrent: 0.35,
            m: [0.0; 8],
            v: [0.0; 8],
            t: 0,
        }
    }

    /// Weights as an array in index order (see `IDX_*`).
    #[must_use]
    pub fn as_array(&self) -> [f64; 8] {
        [
            self.recalls,
            self.abstracts_to,
            self.reinforces,
            self.temporal_seq,
            self.contradicts,
            self.defines,
            self.spawned_from,
            self.recurrent,
        ]
    }

    /// Read weight by index.
    #[must_use]
    pub fn weight(&self, index: usize) -> f64 {
        self.as_array().get(index).copied().unwrap_or(0.0)
    }

    /// One Adam step from observed attributions and scalar reward.
    ///
    /// `attributions` maps 1:1 to [`Self::as_array`] order; entries below
    /// `0.001` are skipped (edge type not involved — implicit regularization).
    /// Zero reward is a no-op (don't learn from noise).
    pub fn adam_update(&mut self, attributions: &[f64], reward: f64) {
        if reward == 0.0 {
            return;
        }
        const LR: f64 = 0.01;
        const BETA1: f64 = 0.9;
        const BETA2: f64 = 0.999;
        const EPS: f64 = 1e-8;

        self.t = self.t.saturating_add(1);
        let t = self.t as i32;
        let mut arr = self.as_array();
        let mut m = self.m;
        let mut v = self.v;
        for (i, slot) in arr.iter_mut().enumerate() {
            let attr = attributions.get(i).copied().unwrap_or(0.0);
            if attr < 0.001 {
                continue;
            }
            let gradient = attr * reward;
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * gradient;
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * gradient * gradient;
            let m_hat = m[i] / (1.0 - BETA1.powi(t));
            let v_hat = v[i] / (1.0 - BETA2.powi(t));
            *slot += LR * m_hat / (v_hat.sqrt() + EPS);
            *slot = clamp_weight(i, *slot);
        }
        self.m = m;
        self.v = v;
        [
            self.recalls,
            self.abstracts_to,
            self.reinforces,
            self.temporal_seq,
            self.contradicts,
            self.defines,
            self.spawned_from,
            self.recurrent,
        ] = arr;
    }

    /// Tiny L2 decay toward defaults after each session (anti-overfit).
    pub fn regularize_toward_defaults(&mut self) {
        const K: f64 = 1e-4;
        let d = Self::defaults().as_array();
        let mut arr = self.as_array();
        for (i, slot) in arr.iter_mut().enumerate() {
            *slot = *slot * (1.0 - K) + d[i] * K;
        }
        [
            self.recalls,
            self.abstracts_to,
            self.reinforces,
            self.temporal_seq,
            self.contradicts,
            self.defines,
            self.spawned_from,
            self.recurrent,
        ] = arr;
    }
}

fn clamp_weight(index: usize, w: f64) -> f64 {
    match index {
        IDX_CONTRADICTS => w.clamp(-0.95, -0.05),
        IDX_RECURRENT => w.clamp(0.05, 0.75),
        _ => w.clamp(0.01, 0.98),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let w = EdgeWeights::defaults();
        assert_eq!(w.as_array(), [0.70, 0.50, 0.60, 0.30, -0.40, 0.20, 0.45, 0.35]);
    }

    #[test]
    fn zero_reward_is_noop() {
        let mut w = EdgeWeights::defaults();
        w.adam_update(&[1.0; 8], 0.0);
        assert_eq!(w.as_array(), EdgeWeights::defaults().as_array());
    }

    #[test]
    fn positive_reward_pushes_active_weight_up() {
        let mut w = EdgeWeights::defaults();
        let mut attr = [0.0; 8];
        attr[IDX_RECALLS] = 0.5;
        w.adam_update(&attr, 1.0);
        assert!(w.recalls > 0.70);
        // untouched weights stay put
        assert_eq!(w.abstracts_to, 0.50);
    }

    #[test]
    fn contradicts_stays_negative() {
        let mut w = EdgeWeights::defaults();
        let mut attr = [0.0; 8];
        attr[IDX_CONTRADICTS] = 1.0;
        for _ in 0..200 {
            w.adam_update(&attr, 1.0);
        }
        assert!(w.contradicts >= -0.95 && w.contradicts <= -0.05);
    }

    #[test]
    fn round_trips_through_json() {
        let w = EdgeWeights::defaults();
        let s = serde_json::to_string(&w).unwrap();
        let back: EdgeWeights = serde_json::from_str(&s).unwrap();
        assert_eq!(back.as_array(), w.as_array());
    }
}
