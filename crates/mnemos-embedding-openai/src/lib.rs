//! OpenAI-compatible embeddings provider.
//!
//! POSTs to `{base_url}/embeddings` (Embeddings API) with `{ model, input }`.
//! Works against `OpenAI` or any OpenAI-compatible endpoint (local servers,
//! gateways) — bearer auth is attached only when the API key is non-empty.

use async_trait::async_trait;
use mnemos_core::{LlmConfig, MnemosError, Result};

/// Embedding client for any OpenAI-compatible `/embeddings` endpoint.
pub struct OpenAiEmbeddingProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiEmbeddingProvider {
    /// Build from explicit parts; trailing `/`s are stripped from `base_url`.
    pub fn new(
        embedding_model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: embedding_model.into(),
        }
    }

    /// Build from unified [`LlmConfig`].
    ///
    /// Uses the [`LlmConfig::embedding_model`] field (not [`LlmConfig::model`]).
    #[must_use]
    pub fn from_config(config: &LlmConfig) -> Self {
        Self::new(
            config.embedding_model.clone(),
            config.base_url.clone(),
            config.api_key.clone(),
        )
    }

    /// Model sent in the `model` field.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Normalized base URL (no trailing slash).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Full embeddings endpoint URL.
    fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.base_url)
    }

    /// POST an `input` payload and return the raw JSON body with its status.
    async fn post_raw(&self, input: serde_json::Value) -> Result<serde_json::Value> {
        let body = serde_json::json!({
            "model": self.model,
            "input": input,
        });
        let mut req = self.client.post(self.embeddings_url()).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.map_err(|e| MnemosError::Http(e.to_string()))?;
        let status = resp.status();
        let json: serde_json::Value =
            resp.json().await.map_err(|e| MnemosError::Http(e.to_string()))?;
        if !status.is_success() {
            return Err(MnemosError::Embedding(format!("HTTP {status}: {json}")));
        }
        Ok(json)
    }

    /// Native `OpenAI` batch: one POST with `input` as an array of texts.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Http`] on transport/JSON failures and
    /// [`MnemosError::Embedding`] on non-2xx status or an unparsable body.
    async fn embed_batch_native(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let json = self.post_raw(serde_json::json!(texts)).await?;
        extract_embeddings(&json)
    }
}

/// Pull `data[0].embedding` out of an Embeddings response body.
///
/// # Errors
///
/// Returns [`MnemosError::Embedding`] when `data[0].embedding` is absent
/// or empty.
fn extract_embedding(json: &serde_json::Value) -> Result<Vec<f32>> {
    let all = extract_embeddings(json)?;
    all.into_iter().next().ok_or_else(|| {
        MnemosError::Embedding(format!("missing data[0].embedding: {json}"))
    })
}

/// Pull every `data[i].embedding` out of an Embeddings response body.
///
/// # Errors
///
/// Returns [`MnemosError::Embedding`] when `data` is absent, empty, or any
/// entry lacks a parseable `embedding` array.
fn extract_embeddings(json: &serde_json::Value) -> Result<Vec<Vec<f32>>> {
    let data = json
        .pointer("/data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| MnemosError::Embedding(format!("missing data[]: {json}")))?;
    if data.is_empty() {
        return Err(MnemosError::Embedding(format!("empty data[]: {json}")));
    }
    data.iter()
        .map(|entry| {
            entry
                .pointer("/embedding")
                .and_then(|v| serde_json::from_value::<Vec<f32>>(v.clone()).ok())
                .ok_or_else(|| {
                    MnemosError::Embedding(format!("missing data[].embedding: {json}"))
                })
        })
        .collect()
}

#[async_trait]
impl mnemos_embedding_trait::EmbeddingProvider for OpenAiEmbeddingProvider {
    /// Embed one text via `POST {base}/embeddings`.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Http`] on transport/JSON failures and
    /// [`MnemosError::Embedding`] on non-2xx status or an unparsable body.
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();
        let out: Result<Vec<f32>> = async {
            let json = self.post_raw(serde_json::json!(text)).await?;
            extract_embedding(&json)
        }
        .await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        mnemos_telemetry::global().record_with_latency(
            "mnemos-embedding-openai",
            "embed",
            out.is_ok(),
            &out.as_ref().map_or_else(|e| e.to_string(), |v| format!("{} dim", v.len())),
            ms,
        );
        out
    }

    /// Embed a batch with one native POST (`input` as array of texts).
    ///
    /// Falls back to sequential [`Self::embed`] calls when the native
    /// batch request fails for any reason.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Http`] on transport/JSON failures and
    /// [`MnemosError::Embedding`] when the model (or fallback) fails.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start = std::time::Instant::now();
        let out = if let Ok(out) = self.embed_batch_native(texts).await {
            Ok(out)
        } else {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed(t).await?);
            }
            Ok(out)
        };
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        mnemos_telemetry::global().record_with_latency(
            "mnemos-embedding-openai",
            "embed_batch",
            out.is_ok(),
            &format!("batch {}", texts.len()),
            ms,
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_embedding_trait::EmbeddingProvider;

    #[test]
    fn url_join_has_no_trailing_slash_duplication() {
        for base in [
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/",
            "http://x///",
        ] {
            let p = OpenAiEmbeddingProvider::new("emb", base, "k");
            assert!(!p.base_url().ends_with('/'), "{base}");
            assert!(p.embeddings_url().ends_with("/embeddings"));
            assert!(!p.embeddings_url().contains("//embeddings"));
        }
    }

    #[test]
    fn from_config_maps_embedding_model_field() {
        let cfg = LlmConfig {
            api_key: "k".into(),
            base_url: "http://localhost:11434/v1/".into(),
            model: "llama".into(),
            embedding_model: "emb".into(),
        };
        let p = OpenAiEmbeddingProvider::from_config(&cfg);
        assert_eq!(p.model(), "emb");
        assert_eq!(p.base_url(), "http://localhost:11434/v1");
    }

    #[test]
    fn data_shape_parses_to_vec() {
        let json = serde_json::json!({
            "object": "list",
            "data": [
                { "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] },
                { "object": "embedding", "index": 1, "embedding": [0.4, 0.5] },
            ],
            "model": "text-embedding-3-small",
        });
        let single = extract_embedding(&json).unwrap();
        assert_eq!(single, vec![0.1_f32, 0.2, 0.3]);
        let batch = extract_embeddings(&json).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[1], vec![0.4_f32, 0.5]);
    }

    #[test]
    fn empty_data_yields_err() {
        let json = serde_json::json!({ "object": "list", "data": [] });
        assert!(extract_embedding(&json).is_err());
        assert!(extract_embeddings(&json).is_err());
        let missing = serde_json::json!({ "id": "x" });
        assert!(extract_embedding(&missing).is_err());
        assert!(extract_embeddings(&missing).is_err());
    }

    #[tokio::test]
    #[ignore = "needs a live OpenAI-compatible endpoint"]
    async fn live_embed_smoke() {
        let p = OpenAiEmbeddingProvider::new(
            "text-embedding-3-small",
            "https://api.openai.com/v1",
            "",
        );
        let _ = p.embed("ping").await;
    }
}
