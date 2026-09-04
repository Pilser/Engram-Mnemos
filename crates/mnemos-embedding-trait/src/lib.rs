//! Embedding provider trait (text → vector).
//!
//! Concrete crates: `mnemos-embedding-openai` (API), `mnemos-embedding-local`.

use async_trait::async_trait;
use mnemos_core::Result;

/// Maps text to fixed-dimension embedding vectors.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed one text.
    ///
    /// # Errors
    ///
    /// Returns [`mnemos_core::MnemosError`] when the model fails.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch (default: sequential).
    ///
    /// # Errors
    ///
    /// Returns [`mnemos_core::MnemosError`] when the model fails.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }

    /// Expected vector dimension.
    fn dimension(&self) -> usize {
        mnemos_core::EMBEDDING_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEmbed;
    #[async_trait]
    impl EmbeddingProvider for MockEmbed {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0; 8])
        }
    }

    #[tokio::test]
    async fn batch_embeds_sequentially() {
        let out = MockEmbed
            .embed_batch(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
    }
}
