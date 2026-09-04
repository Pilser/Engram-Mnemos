//! Local (on-device) embedding provider backed by
//! [`fastembed`](https://crates.io/crates/fastembed) 6.0.2.
//!
//! # ⚠️ DIMENSION WARNING — READ BEFORE USE ⚠️
//!
//! The default local model produces **384**-dimensional vectors
//! ([`LOCAL_EMBEDDING_DIM`]), while [`mnemos_core::EMBEDDING_DIM`] (the
//! OpenAI-provider dimension) is **1536**.
//!
//! **Do NOT mix vector stores between providers.** A store populated with
//! 1536-dim OpenAI embeddings cannot be queried with 384-dim local
//! embeddings and vice versa. Pick one provider per store.
//!
//! # fastembed 6.0.2 API findings (verified against
//! `~/.cargo/registry/src/*/fastembed-6.0.2/src/`)
//!
//! - Struct: `fastembed::TextEmbedding` (defined in
//!   `src/text_embedding/init.rs`, re-exported at the crate root).
//! - Constructor: `TextEmbedding::try_new(options: TextInitOptions) ->
//!   Result<Self>` — **synchronous**; `TextInitOptions =
//!   InitOptionsWithLength<EmbeddingModel>`, built with
//!   `TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)`. It **downloads
//!   the model from the Hugging Face Hub on first use** (network access,
//!   honouring `HF_HOME` / cache dir), so [`LocalEmbeddingProvider::new`]
//!   and [`LocalEmbeddingProvider::with_model`] are fallible and map
//!   download/init failures to [`mnemos_core::MnemosError::Embedding`].
//! - Model enum: `fastembed::EmbeddingModel` (defined in
//!   `src/models/text_embedding.rs`). Small-model variants verified:
//!   `AllMiniLML6V2` (dim **384**, the default used here) and the crate's
//!   own `#[default]` variant `BGESmallENV15` (also dim 384).
//! - Inference: `embed<S: AsRef<str> + Send + Sync>(&mut self, texts: impl
//!   AsRef<[S]>, batch_size: Option<usize>) -> Result<Vec<Embedding>>`
//!   where `Embedding = Vec<f32>` (see `src/text_embedding/impl.rs`).
//!   Note the differences from the plan sketch: it is **synchronous**,
//!   takes **`&mut self`** (not `&self`), and is **batch-oriented** (a
//!   slice of texts, not a single `&str`).
//!
//! # Concurrency notes
//!
//! Because `fastembed`'s `embed` takes `&mut self` while
//! [`mnemos_embedding_trait::EmbeddingProvider::embed`] takes `&self`
//! (and requires `Send + Sync`), the model is held behind a mutex (field
//! `model`; the plan sketch's bare `fastembed::TextEmbedding` field cannot
//! satisfy the trait).
//!
//! An `std::sync::Mutex` (not `tokio::sync::Mutex`) is used deliberately:
//! the workspace `tokio` dependency does not enable the `sync` feature, and
//! the lock is never held across an `.await`, so an async mutex would buy
//! nothing.
//!
//! The synchronous ONNX call currently runs inline on the async executor
//! while the lock is held. A future improvement is to move inference into
//! `tokio::task::spawn_blocking` (with a dedicated worker) so long-running
//! local inference does not stall the runtime.

use async_trait::async_trait;
use mnemos_core::{MnemosError, Result};
use mnemos_embedding_trait::EmbeddingProvider;

/// Dimension of the default local model
/// ([`fastembed::EmbeddingModel::AllMiniLML6V2`]).
///
/// Deliberately **not** [`mnemos_core::EMBEDDING_DIM`] (1536): local and
/// OpenAI vectors must never share a store.
pub const LOCAL_EMBEDDING_DIM: usize = 384;

/// Local embedding provider using an on-device `fastembed` model.
///
/// Construct with [`Self::new`] (default small model) or
/// [`Self::with_model`]. Both download the model weights on first run and
/// therefore need network access exactly once (then the Hub cache is reused).
pub struct LocalEmbeddingProvider {
    /// Guarded because `fastembed::TextEmbedding::embed` takes `&mut self`.
    /// `std` (not `tokio`) mutex: no `tokio/sync` feature in the workspace,
    /// and the guard is never held across `.await`.
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    /// Actual output dimension of the chosen model (looked up offline via
    /// `TextEmbedding::get_model_info`, no download required).
    dimension: usize,
}

impl LocalEmbeddingProvider {
    /// Create a provider with the default small model
    /// ([`fastembed::EmbeddingModel::AllMiniLML6V2`], 384 dims).
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Embedding`] when the model cannot be
    /// downloaded or initialised (e.g. no network on first run).
    pub fn new() -> Result<Self> {
        Self::with_model(fastembed::EmbeddingModel::AllMiniLML6V2)
    }

    /// Create a provider with an explicit `fastembed` model.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Embedding`] when the model is unknown or
    /// cannot be downloaded/initialised.
    pub fn with_model(model: fastembed::EmbeddingModel) -> Result<Self> {
        // Offline static lookup — safe to do before the downloading constructor.
        let dimension = Self::dimension_of(&model)?;
        let embedding =
            fastembed::TextEmbedding::try_new(fastembed::TextInitOptions::new(model)).map_err(
                |e| MnemosError::Embedding(format!("fastembed init/download failed: {e}")),
            )?;
        Ok(Self {
            model: std::sync::Mutex::new(embedding),
            dimension,
        })
    }

    /// Look up a model's output dimension without downloading anything.
    ///
    /// Pure offline metadata query over fastembed's static model table.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Embedding`] for an unknown model variant.
    pub fn dimension_of(model: &fastembed::EmbeddingModel) -> Result<usize> {
        fastembed::TextEmbedding::get_model_info(model)
            .map(|info| info.dim)
            .map_err(|e| MnemosError::Embedding(format!("unknown fastembed model: {e}")))
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    /// Embed one text via local inference.
    ///
    /// Wraps the synchronous batch-oriented `fastembed::TextEmbedding::embed`
    /// (`&mut self`, hence the mutex) for a single input.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Embedding`] when local inference fails.
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();
        let out: Result<Vec<f32>> = async {
            // Never held across `.await`, so a blocking mutex is safe here.
            let mut guard = self
                .model
                .lock()
                .map_err(|_| MnemosError::Embedding("local model lock poisoned".to_owned()))?;
            // `&mut self` + sync + batch slice API → adapt a one-element batch.
            let mut out = guard
                .embed([text], None)
                .map_err(|e| MnemosError::Embedding(format!("fastembed inference failed: {e}")))?;
            debug_assert_eq!(out.len(), 1, "single-input embed must return one vector");
            out.pop()
                .ok_or_else(|| MnemosError::Embedding("fastembed returned no embedding".to_owned()))
        }
        .await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        mnemos_telemetry::global().record_with_latency(
            "mnemos-embedding-local",
            "embed",
            out.is_ok(),
            &out.as_ref().map_or_else(|e| e.to_string(), |v| format!("{} dim", v.len())),
            ms,
        );
        out
    }

    /// Actual dimension of the loaded local model (e.g. 384), **not**
    /// [`mnemos_core::EMBEDDING_DIM`]. See the module-level dimension warning.
    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_dim_constant_is_384_not_1536() {
        // Guards the documented contract: local dim ≠ EMBEDDING_DIM.
        assert_eq!(LOCAL_EMBEDDING_DIM, 384);
        assert_ne!(LOCAL_EMBEDDING_DIM, mnemos_core::EMBEDDING_DIM);
    }

    #[test]
    fn dimension_lookup_needs_no_download() {
        // Static metadata only — runs offline.
        assert_eq!(
            LocalEmbeddingProvider::dimension_of(&fastembed::EmbeddingModel::AllMiniLML6V2)
                .expect("known model"),
            384
        );
        assert_eq!(
            LocalEmbeddingProvider::dimension_of(&fastembed::EmbeddingModel::BGESmallENV15)
                .expect("known model"),
            384
        );
    }

    #[test]
    fn all_supported_models_have_nonzero_dim() {
        // Static metadata only — runs offline.
        for info in fastembed::TextEmbedding::list_supported_models() {
            assert!(info.dim > 0, "model {:?} has zero dim", info.model);
        }
    }

    #[tokio::test]
    #[ignore = "downloads model from Hugging Face Hub on first run (network)"]
    async fn init_default_model_and_embed() {
        let provider = LocalEmbeddingProvider::new().expect("model init");
        assert_eq!(provider.dimension(), LOCAL_EMBEDDING_DIM);
        let vec = provider.embed("hello world").await.expect("embed");
        assert_eq!(vec.len(), provider.dimension());
    }
}
