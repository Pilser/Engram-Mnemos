# MNEMOS Project Structure Plan

> Complete project architecture for a highly modular, multi-interface AI memory system

---

## Goal

Create an extremely modular Rust project that:
- Supports multiple interfaces (CLI, MCP server, dedicated MCP tools)
- Has pluggable LLM providers (OpenAI, Anthropic, local)
- Maintains high code quality across 20+ crates
- Enables fast incremental builds
- Is easy to extend with new features

---

## Folder Structure

```
mnemos/
├── Cargo.toml                         # Workspace root
├── Cargo.lock                         # Shared lock file
├── .cargo/config.toml                 # Linker config (mold)
├── rust-toolchain.toml                # Pin Rust version
├── .github/                           # CI/CD workflows
│
├── crates/
│   │
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── # CORE (pure logic, minimal deps)
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── mnemos-core/                   # Types, schema, config, errors
│   ├── mnemos-storage/                # HelixDB embedded integration
│   │
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── # PIPELINE (orchestration)
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── mnemos-ingestion/              # Ingest memories (full pipeline)
│   ├── mnemos-retrieval/              # Retrieve memories (CRR + stimulation)
│   ├── mnemos-consolidation/          # Decay, compress, promote, mitosis
│   │
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── # ML MODELS (pluggable, trait-based)
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── mnemos-ml-trait/               # Trait: MlModel, EmotionalTagger, etc.
│   ├── mnemos-emotional-tagger/       # Emotional valence scoring (LLM-based)
│   ├── mnemos-importance-scorer/      # Importance scoring (LLM-based)
│   ├── mnemos-concept-extractor/      # Concept extraction (LLM-based)
│   │
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── # STIMULATION (spreading activation)
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── mnemos-stimulation/            # Spreading activation engine
│   ├── mnemos-edge-weights/           # Learnable edge weights (Adam optimizer)
│   │
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── # LLM PROVIDER LAYER (pluggable, unified)
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── mnemos-llm-trait/              # Trait: LlmProvider, ChatProvider
│   ├── mnemos-llm-openai/             # OpenAI compatible (OpenAI, Groq, Together, etc.)
│   ├── mnemos-llm-anthropic/          # Anthropic (uses unified openai-compatible lib)
│   ├── mnemos-llm-local/              # Local LLM via Ollama (OpenAI compatible)
│   ├── mnemos-embedding-trait/        # Trait: EmbeddingProvider
│   ├── mnemos-embedding-openai/       # OpenAI embeddings
│   ├── mnemos-embedding-local/        # Local embeddings (fastembed, etc.)
│   │
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── # INTERFACES (user-facing)
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── mnemos-cli/                    # CLI library (query, reward, consolidate)
│   ├── mnemos-mcp-server/             # MCP server (wraps CLI as one tool: mnemos_cli)
│   ├── mnemos-mcp-tools/              # Dedicated MCP tools (one per function)
│   │
│   ├── # ═══════════════════════════════════════════════════════════
│   ├── # BINARY (entry point)
│   ├── # ═══════════════════════════════════════════════════════════
│   └── mnemos-app/                    # Main binary (combines everything)
│
├── tests/                             # Integration tests
│   ├── ingestion_tests.rs
│   ├── retrieval_tests.rs
│   └── consolidation_tests.rs
│
└── examples/                          # Usage examples
    ├── basic_ingest.rs
    ├── basic_recall.rs
    └── mcp_server.rs
```

---

## Crate Details

### Core Crates

#### `mnemos-core`
**Purpose:** Types, schema definitions, configuration, error types

```rust
// Types
pub struct Engram { /* ... */ }
pub struct Concept { /* ... */ }
pub struct Identity { /* ... */ }

// Schema
pub fn create_engram_schema() -> WriteBatch { /* ... */ }
pub fn create_concept_schema() -> WriteBatch { /* ... */ }

// Config
pub struct MnemosConfig {
    pub storage: StorageConfig,
    pub llm: LlmConfig,
    pub stimulation: StimulationConfig,
}

// Errors
pub enum MnemosError { /* ... */ }
```

**Dependencies:** None (pure types)

#### `mnemos-storage`
**Purpose:** HelixDB embedded integration

```rust
pub struct Storage {
    client: Client,
}

impl Storage {
    pub async fn open(config: StorageConfig) -> Result<Self>;
    pub async fn query(&self, request: QueryRequest) -> Result<Value>;
    pub async fn close(self) -> Result<()>;
}
```

**Dependencies:** `helix-db`

---

### Pipeline Crates

#### `mnemos-ingestion`
**Purpose:** Full ingestion pipeline (emotional tagger → importance scorer → concept extractor → store)

```rust
pub struct IngestionPipeline {
    emotional_tagger: Box<dyn EmotionalTagger>,
    importance_scorer: Box<dyn ImportanceScorer>,
    concept_extractor: Box<dyn ConceptExtractor>,
    storage: Storage,
}

impl IngestionPipeline {
    pub async fn ingest(&self, text: &str) -> Result<EngramId>;
}
```

**Dependencies:** `mnemos-core`, `mnemos-storage`, `mnemos-ml-trait`, `mnemos-emotional-tagger`, `mnemos-importance-scorer`, `mnemos-concept-extractor`

#### `mnemos-retrieval`
**Purpose:** Retrieve memories using CRR + Stimulation Layer

```rust
pub struct RetrievalPipeline {
    storage: Storage,
    stimulation: StimulationEngine,
    edge_weights: EdgeWeights,
}

impl RetrievalPipeline {
    pub async fn recall(&mut self, query: &str, limit: usize) -> Result<Vec<ResonanceResult>>;
    pub async fn reward(&self, engram_id: u64, reward: f64) -> Result<()>;
}
```

**Dependencies:** `mnemos-core`, `mnemos-storage`, `mnemos-stimulation`, `mnemos-edge-weights`

#### `mnemos-consolidation`
**Purpose:** Background maintenance (decay, compress, promote, mitosis)

```rust
pub struct ConsolidationPipeline {
    storage: Storage,
    config: ConsolidationConfig,
}

impl ConsolidationPipeline {
    pub async fn consolidate(&self) -> Result<ConsolidationReport>;
    pub async fn run_decay(&self) -> Result<()>;
    pub async fn run_mitosis(&self) -> Result<()>;
}
```

**Dependencies:** `mnemos-core`, `mnemos-storage`

---

### ML Model Crates

#### `mnemos-ml-trait`
**Purpose:** Trait definitions for all ML models

```rust
#[async_trait]
pub trait EmotionalTagger: Send + Sync {
    async fn tag(&self, text: &str) -> Result<f64>;
}

#[async_trait]
pub trait ImportanceScorer: Send + Sync {
    async fn score(&self, text: &str) -> Result<f64>;
}

#[async_trait]
pub trait ConceptExtractor: Send + Sync {
    async fn extract(&self, text: &str) -> Result<Vec<ExtractedConcept>>;
}
```

**Dependencies:** `mnemos-core`

#### `mnemos-emotional-tagger`
**Purpose:** Emotional valence scoring using LLM

```rust
pub struct LlmEmotionalTagger {
    llm: Box<dyn LlmProvider>,
}

#[async_trait]
impl EmotionalTagger for LlmEmotionalTagger {
    async fn tag(&self, text: &str) -> Result<f64> {
        let prompt = format!(
            "Score the emotional valence of this text from -1.0 (very negative) \
             to +1.0 (very positive). Return only the number.\n\nText: {}",
            text
        );
        let response = self.llm.chat(&prompt).await?;
        Ok(response.trim().parse()?)
    }
}
```

**Dependencies:** `mnemos-core`, `mnemos-ml-trait`, `mnemos-llm-trait`

#### `mnemos-importance-scorer`
**Purpose:** Importance scoring using LLM

```rust
pub struct LlmImportanceScorer {
    llm: Box<dyn LlmProvider>,
}

#[async_trait]
impl ImportanceScorer for LlmImportanceScorer {
    async fn score(&self, text: &str) -> Result<f64> {
        let prompt = format!(
            "Rate how important this memory is for a personal AI to retain long-term. \
             Return ONLY a float from 0.0 to 1.0.\n\nMemory: {}",
            text
        );
        let response = self.llm.chat(&prompt).await?;
        Ok(response.trim().parse()?)
    }
}
```

**Dependencies:** `mnemos-core`, `mnemos-ml-trait`, `mnemos-llm-trait`

#### `mnemos-concept-extractor`
**Purpose:** Concept extraction using LLM

```rust
pub struct LlmConceptExtractor {
    llm: Box<dyn LlmProvider>,
}

#[async_trait]
impl ConceptExtractor for LlmConceptExtractor {
    async fn extract(&self, text: &str) -> Result<Vec<ExtractedConcept>> {
        let prompt = format!(
            "Extract key concepts from this memory. Return JSON array of \
             {{\"name\": \"...\", \"confidence\": 0.0}}.\n\nMemory: {}",
            text
        );
        let response = self.llm.chat(&prompt).await?;
        Ok(parse_concepts(&response)?)
    }
}
```

**Dependencies:** `mnemos-core`, `mnemos-ml-trait`, `mnemos-llm-trait`

---

### Stimulation Crates

#### `mnemos-stimulation`
**Purpose:** Spreading activation engine

```rust
pub struct StimulationEngine {
    storage: Storage,
    config: StimulationConfig,
}

impl StimulationEngine {
    pub async fn stimulate(
        &self,
        query_embedding: Vec<f32>,
        edge_weights: &EdgeWeights,
    ) -> Result<StimulationResult>;
}
```

**Dependencies:** `mnemos-core`, `mnemos-storage`

#### `mnemos-edge-weights`
**Purpose:** Learnable edge weights with Adam optimizer

```rust
pub struct EdgeWeights {
    pub recalls: f64,
    pub abstracts_to: f64,
    pub reinforces: f64,
    pub temporal_seq: f64,
    pub contradicts: f64,
    pub defines: f64,
    pub spawned_from: f64,
    pub recurrent: f64,
    // Adam state
    m: [f64; 8],
    v: [f64; 8],
    t: u64,
}

impl EdgeWeights {
    pub fn defaults() -> Self;
    pub fn adam_update(&mut self, attributions: &[f64], reward: f64);
}
```

**Dependencies:** `mnemos-core`

---

### LLM Provider Crates

#### `mnemos-llm-trait`
**Purpose:** Unified LLM provider trait

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn chat(&self, prompt: &str) -> Result<String>;
    async fn chat_with_system(&self, system: &str, user: &str) -> Result<String>;
}

#[async_trait]
pub trait StructuredOutput: LlmProvider {
    async fn chat_structured<T: DeserializeOwned>(&self, prompt: &str) -> Result<T>;
}
```

**Dependencies:** `mnemos-core`

#### `mnemos-llm-openai`
**Purpose:** OpenAI compatible provider (OpenAI, Groq, Together, Ollama, etc.)

```rust
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,  // https://api.openai.com/v1 or http://localhost:11434/v1
    model: String,
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    async fn chat(&self, prompt: &str) -> Result<String> {
        // OpenAI Chat Completions API
    }
}
```

**Dependencies:** `mnemos-core`, `mnemos-llm-trait`, `reqwest`, `serde`

#### `mnemos-llm-anthropic`
**Purpose:** Anthropic provider (uses unified openai-compatible lib internally)

```rust
pub struct AnthropicProvider {
    inner: OpenAiCompatibleProvider,  // Anthropic has OpenAI-compatible API
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.inner.chat(prompt).await
    }
}
```

**Dependencies:** `mnemos-core`, `mnemos-llm-trait`, `mnemos-llm-openai`

#### `mnemos-llm-local`
**Purpose:** Local LLM via Ollama (OpenAI compatible)

```rust
pub struct OllamaProvider {
    inner: OpenAiCompatibleProvider,  // Ollama has OpenAI-compatible API
}

impl OllamaProvider {
    pub fn new(model: &str) -> Self {
        Self {
            inner: OpenAiCompatibleProvider::new(
                model,
                "http://localhost:11434/v1",  // Ollama default
            ),
        }
    }
}
```

**Dependencies:** `mnemos-core`, `mnemos-llm-trait`, `mnemos-llm-openai`

#### `mnemos-embedding-trait`
**Purpose:** Embedding provider trait

```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
```

**Dependencies:** `mnemos-core`

#### `mnemos-embedding-openai`
**Purpose:** OpenAI embeddings

```rust
pub struct OpenAiEmbeddingProvider {
    inner: OpenAiCompatibleProvider,
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // OpenAI Embeddings API
    }
}
```

**Dependencies:** `mnemos-core`, `mnemos-embedding-trait`, `mnemos-llm-openai`

#### `mnemos-embedding-local`
**Purpose:** Local embeddings (fastembed or similar)

```rust
pub struct LocalEmbeddingProvider {
    model: fastembed::TextEmbedding,
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Local inference, no API call
    }
}
```

**Dependencies:** `mnemos-core`, `mnemos-embedding-trait`, `fastembed`

---

### Interface Crates

#### `mnemos-cli`
**Purpose:** CLI library with all commands

```rust
pub struct Cli {
    ingestion: IngestionPipeline,
    retrieval: RetrievalPipeline,
    consolidation: ConsolidationPipeline,
}

impl Cli {
    pub async fn ingest(&self, text: &str) -> Result<()>;
    pub async fn recall(&self, query: &str, limit: usize) -> Result<Vec<ResonanceResult>>;
    pub async fn reward(&self, engram_id: u64, score: f64) -> Result<()>;
    pub async fn consolidate(&self) -> Result<()>;
    pub async fn stats(&self) -> Result<MemoryStats>;
}
```

**Dependencies:** All pipeline crates

#### `mnemos-mcp-server`
**Purpose:** MCP server that wraps CLI as one tool (`mnemos_cli`)

```rust
// Single MCP tool that exposes all CLI commands
#[mcp_tool]
pub struct MnemosCliTool {
    cli: Cli,
}

impl MnemosCliTool {
    #[mcp_method("mnemos_cli")]
    pub async fn execute(&self, command: String, args: Vec<String>) -> Result<String> {
        match command.as_str() {
            "ingest" => self.cli.ingest(&args[0]).await,
            "recall" => self.cli.recall(&args[0], args[1].parse()?).await,
            "reward" => self.cli.reward(args[0].parse()?, args[1].parse()?).await,
            "consolidate" => self.cli.consolidate().await,
            _ => Err("Unknown command"),
        }
    }
}
```

**Dependencies:** `mnemos-cli`

#### `mnemos-mcp-tools`
**Purpose:** Dedicated MCP tools (one per function)

```rust
// One MCP tool per function
#[mcp_tool]
pub struct MnemosIngestTool {
    cli: Cli,
}

#[mcp_tool]
pub struct MnemosRecallTool {
    cli: Cli,
}

#[mcp_tool]
pub struct MnemosRewardTool {
    cli: Cli,
}

#[mcp_tool]
pub struct MnemosConsolidateTool {
    cli: Cli,
}
```

**Dependencies:** `mnemos-cli`

---

### Binary Crate

#### `mnemos-app`
**Purpose:** Main binary entry point

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let config = MnemosConfig::from_env()?;
    let cli = Cli::new(config).await?;

    // Run CLI
    cli.run().await
}
```

**Dependencies:** All crates

---

## Dependency Graph (No Circular Deps)

```
mnemos-app
├── mnemos-cli
│   ├── mnemos-ingestion
│   │   ├── mnemos-core
│   │   ├── mnemos-storage
│   │   ├── mnemos-emotional-tagger
│   │   │   ├── mnemos-core
│   │   │   ├── mnemos-ml-trait
│   │   │   └── mnemos-llm-trait
│   │   ├── mnemos-importance-scorer
│   │   │   ├── mnemos-core
│   │   │   ├── mnemos-ml-trait
│   │   │   └── mnemos-llm-trait
│   │   └── mnemos-concept-extractor
│   │       ├── mnemos-core
│   │       ├── mnemos-ml-trait
│   │       └── mnemos-llm-trait
│   ├── mnemos-retrieval
│   │   ├── mnemos-core
│   │   ├── mnemos-storage
│   │   ├── mnemos-stimulation
│   │   │   ├── mnemos-core
│   │   │   └── mnemos-storage
│   │   └── mnemos-edge-weights
│   │       └── mnemos-core
│   └── mnemos-consolidation
│       ├── mnemos-core
│       └── mnemos-storage
├── mnemos-mcp-server
│   └── mnemos-cli
└── mnemos-mcp-tools
    └── mnemos-cli

mnemos-llm-openai
├── mnemos-core
└── mnemos-llm-trait

mnemos-llm-anthropic
├── mnemos-core
├── mnemos-llm-trait
└── mnemos-llm-openai  # Uses OpenAI-compatible API

mnemos-llm-local
├── mnemos-core
├── mnemos-llm-trait
└── mnemos-llm-openai  # Ollama is OpenAI-compatible

mnemos-embedding-openai
├── mnemos-core
├── mnemos-embedding-trait
└── mnemos-llm-openai

mnemos-embedding-local
├── mnemos-core
└── mnemos-embedding-trait
```

---

## Build Optimization

### Linker Configuration

```toml
# .cargo/config.toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

**mold linker:** 4.9x faster than lld, pure drop-in replacement.

### Release Profile

```toml
# Cargo.toml (workspace root)
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
strip = true
debug = false
```

### Incremental Builds

Each crate is an independent compilation target. Only changed crates recompile:
- Change `mnemos-core` → only dependent crates recompile
- Change `mnemos-emotional-tagger` → only `mnemos-ingestion` recompiles
- Change `mnemos-cli` → only `mnemos-app` recompiles

### Tracing Optimization

Use `tracing` crate with compile-time level filtering:

```toml
# Cargo.toml
tracing = { version = "0.1", features = ["max_level_debug", "release_max_level_info"] }
```

This eliminates trace calls at compile time in release builds — zero runtime cost.

---

## Quality Gates

### Per-Crate Standards

| Crate Type | Tests | Docs | Clippy |
|------------|-------|------|--------|
| Core | Unit tests required | Full docs | Deny warnings |
| Pipeline | Unit + integration | Full docs | Deny warnings |
| ML | Unit tests | Full docs | Deny warnings |
| Interfaces | Integration tests | Examples | Deny warnings |

### Workspace-Level

```toml
# Cargo.toml (workspace root)
[workspace.metadata.clippy]
all-targets = true

[workspace.lints.clippy]
all = "warn"
pedantic = "warn"
```

---

## Implementation Order

### Phase 1: Core Foundation
1. `mnemos-core` — types, schema, config
2. `mnemos-storage` — HelixDB integration
3. `mnemos-llm-trait` — LLM provider trait
4. `mnemos-llm-openai` — OpenAI compatible provider

### Phase 2: ML Models
5. `mnemos-ml-trait` — ML model traits
6. `mnemos-emotional-tagger` — emotional scoring
7. `mnemos-importance-scorer` — importance scoring
8. `mnemos-concept-extractor` — concept extraction

### Phase 3: Pipeline
9. `mnemos-ingestion` — full ingestion pipeline
10. `mnemos-edge-weights` — learnable weights
11. `mnemos-stimulation` — spreading activation
12. `mnemos-retrieval` — retrieval pipeline
13. `mnemos-consolidation` — consolidation loop

### Phase 4: Interfaces
14. `mnemos-cli` — CLI library
15. `mnemos-mcp-server` — MCP server (wraps CLI)
16. `mnemos-mcp-tools` — dedicated MCP tools
17. `mnemos-app` — main binary

### Phase 5: Additional Providers
18. `mnemos-llm-anthropic` — Anthropic provider
19. `mnemos-llm-local` — Ollama provider
20. `mnemos-embedding-trait` — embedding trait
21. `mnemos-embedding-openai` — OpenAI embeddings
22. `mnemos-embedding-local` — local embeddings

---

## Verification

After implementation:
1. `cargo build` — all crates compile
2. `cargo test` — all tests pass
3. `cargo clippy` — no warnings
4. `cargo doc` — all docs build
5. `cargo build --release` — optimized binary
6. Incremental build test: change one crate, verify only dependents recompile

---

## Summary

| Aspect | Decision |
|--------|----------|
| **Structure** | Workspace with 22 crates |
| **Interfaces** | CLI + MCP server (wraps CLI) + dedicated MCP tools |
| **LLM providers** | OpenAI compatible (unified), Anthropic, Ollama |
| **Embeddings** | OpenAI + local |
| **Linker** | mold (4.9x faster) |
| **Build** | Incremental (per-crate compilation) |
| **Tracing** | Compile-time level filtering |
| **Quality** | Clippy deny warnings, full docs, unit + integration tests |
