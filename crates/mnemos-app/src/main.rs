#![recursion_limit = "256"]
//! `mnemos` binary: env-configured memory CLI plus MCP servers.
//!
//! Wiring only — all behaviour lives in the pipeline crates. The binary:
//! loads [`MnemosConfig::from_env`](mnemos_core::MnemosConfig::from_env),
//! picks chat/embedding providers from `LLM_PROVIDER` / `EMBEDDING_PROVIDER`,
//! builds one [`Storage`](mnemos_storage::Storage) per pipeline plus the
//! [`Cli`](mnemos_cli::Cli) facade, then dispatches `argv` manually (no clap).

use std::sync::Arc;

use mnemos_cli::Cli;
use mnemos_concept_extractor::LlmConceptExtractor;
use mnemos_consolidation::ConsolidationPipeline;
use mnemos_contradiction::ContradictionDetector;
use mnemos_core::{LlmConfig, MnemosConfig};
use mnemos_edge_weights::EdgeWeights;
use mnemos_embedding_local::LocalEmbeddingProvider;
use mnemos_embedding_openai::OpenAiEmbeddingProvider;
use mnemos_embedding_trait::EmbeddingProvider;
use mnemos_emotional_tagger::LlmEmotionalTagger;
use mnemos_importance_scorer::LlmImportanceScorer;
use mnemos_ingestion::IngestionPipeline;
use mnemos_llm_anthropic::AnthropicProvider;
use mnemos_llm_local::OllamaProvider;
use mnemos_llm_openai::OpenAiCompatibleProvider;
use mnemos_llm_trait::LlmProvider;
use mnemos_mitosis::MitosisSplitter;
use mnemos_retrieval::RetrievalPipeline;
use mnemos_stimulation::StimulationEngine;
use mnemos_storage::Storage;

/// Load env files if present: `.env`, `deploy/.env`, `local/.env` (all optional, gitignored deploy/local take precedence).
fn load_env_files() {
    let _ = dotenvy::dotenv(); // .env at repo root
    // deploy/.env and local/.env override .env when present
    let _ = dotenvy::from_filename_override("deploy/.env");
    let _ = dotenvy::from_filename_override("local/.env");
    let _ = dotenvy::from_filename_override(".env.local");
}

/// Default result count for `engram recall` when `--limit` is omitted.
pub const DEFAULT_RECALL_LIMIT: usize = 5;

/// Parsed subcommand. [`parse_args`] is the only pure, unit-testable piece
/// of the binary; everything else performs I/O.
#[derive(Debug, PartialEq)]
pub enum Command {
    /// `ingest <text...>` — store one episodic memory.
    Ingest {
        text: String,
    },
    /// `recall <query...> [--limit N]` — recall top-`limit` memories.
    Recall {
        query: String,
        limit: usize,
    },
    /// `reward <score> [attributions csv]` — Adam-update edge weights.
    Reward {
        score: f64,
        attributions: Vec<f64>,
    },
    /// `consolidate` — run one consolidation ("sleep") cycle.
    Consolidate,
    /// `stats` — print aggregate memory statistics as JSON.
    Stats,
    /// `mcp-server` — serve the full MCP server over stdio.
    McpServer,
    /// `mcp-tools` — serve the MCP tool subset over stdio.
    McpTools,
    /// Explicit `help` / `--help` / `-h`.
    Help,
    /// Anything unparseable; `message` is shown alongside the usage text.
    Invalid {
        message: String,
    },
}

/// Parse `argv` (including `argv[0]`, the program name) into a [`Command`].
///
/// Pure: no env reads, no I/O.
#[must_use]
pub fn parse_args(argv: &[String]) -> Command {
    let mut words = argv.iter().skip(1);
    let Some(command) = words.next() else {
        return Command::Invalid {
            message: "no command given".to_string(),
        };
    };
    let rest: Vec<&str> = words.map(String::as_str).collect();
    match command.as_str() {
        "ingest" => {
            let text = rest.join(" ");
            if text.trim().is_empty() {
                Command::Invalid {
                    message: "ingest needs text: engram ingest <text...>".to_string(),
                }
            } else {
                Command::Ingest {
                    text,
                }
            }
        }
        "recall" => parse_recall(&rest),
        "reward" => parse_reward(&rest),
        "consolidate" => Command::Consolidate,
        "stats" => Command::Stats,
        "mcp-server" => Command::McpServer,
        "mcp-tools" => Command::McpTools,
        "help" | "--help" | "-h" => Command::Help,
        other => Command::Invalid {
            message: format!("unknown command {other:?}"),
        },
    }
}

/// Parse the tail of `recall <query...> [--limit N]`.
fn parse_recall(rest: &[&str]) -> Command {
    let mut limit = DEFAULT_RECALL_LIMIT;
    let mut query_parts: Vec<&str> = Vec::new();
    let mut index = 0;
    while index < rest.len() {
        let word = rest[index];
        if word == "--limit" {
            index += 1;
            let value = rest.get(index).and_then(|s| s.parse::<usize>().ok());
            let Some(value) = value else {
                return Command::Invalid {
                    message: "--limit needs a number: engram recall <query...> [--limit N]"
                        .to_string(),
                };
            };
            limit = value;
        } else if let Some(value) = word.strip_prefix("--limit=") {
            let Ok(value) = value.parse::<usize>() else {
                return Command::Invalid {
                    message: "--limit needs a number: engram recall <query...> [--limit N]"
                        .to_string(),
                };
            };
            limit = value;
        } else {
            query_parts.push(word);
        }
        index += 1;
    }
    let query = query_parts.join(" ");
    if query.trim().is_empty() {
        Command::Invalid {
            message: "recall needs a query: engram recall <query...> [--limit N]".to_string(),
        }
    } else {
        Command::Recall {
            query,
            limit,
        }
    }
}

/// Parse the tail of `reward <score> [attributions csv]`.
fn parse_reward(rest: &[&str]) -> Command {
    let Some(score_word) = rest.first() else {
        return Command::Invalid {
            message: "reward needs a score: engram reward <score> [attributions csv]"
                .to_string(),
        };
    };
    let Ok(score) = score_word.parse::<f64>() else {
        return Command::Invalid {
            message: format!("reward score is not a number: {score_word:?}"),
        };
    };
    let mut attributions = Vec::new();
    if let Some(csv) = rest.get(1) {
        for part in csv.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let Ok(value) = part.parse::<f64>() else {
                return Command::Invalid {
                    message: format!("reward attribution is not a number: {part:?}"),
                };
            };
            attributions.push(value);
        }
    }
    Command::Reward {
        score,
        attributions,
    }
}

/// Usage text, printed for `Help` and [`Command::Invalid`].
fn usage() -> &'static str {
    "usage: engram <command> [args]\n\
     \n\
     commands:\n\
     \x20 ingest <text...>                    store one episodic memory\n\
     \x20 recall <query...> [--limit N]       recall top-N memories as JSON (default 5)\n\
     \x20 reward <score> [attributions csv]   apply scalar reward to edge weights\n\
     \x20 consolidate                         run one consolidation cycle\n\
     \x20 stats                               print memory stats as JSON\n\
     \x20 mcp-server                           serve the full MCP server over stdio\n\
     \x20 mcp-tools                            serve the MCP tool subset over stdio\n\
     \n\
     env:\n\
     \x20 LLM_PROVIDER=openai|xai|anthropic|ollama (default openai)\n\
     \x20 EMBEDDING_PROVIDER=openai|local      (default openai)"
}

/// Build one chat provider from `LLM_PROVIDER` (default `openai`).
///
/// # Errors
///
/// Returns a message when `LLM_PROVIDER` names an unknown provider.
fn build_chat_provider(config: &LlmConfig) -> Result<Box<dyn LlmProvider>, String> {
    let name = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());
    match name.trim().to_lowercase().as_str() {
        "" | "openai" | "xai" | "grok" => {
            // xAI Grok is OpenAI-compatible at https://api.x.ai/v1 — use the same path.
            // Allow `XAI_API_KEY` to override `OPENAI_API_KEY` when set.
            let mut cfg = config.clone();
            if name.trim().to_lowercase().as_str() == "xai" || name.trim().to_lowercase().as_str() == "grok" {
                if let Ok(key) = std::env::var("XAI_API_KEY") {
                    if !key.trim().is_empty() {
                        cfg.api_key = key;
                    }
                }
            }
            Ok(Box::new(OpenAiCompatibleProvider::from_config(&cfg)))
        }
        "anthropic" => {
            // `LlmConfig::from_env` only reads `OPENAI_API_KEY`; prefer a
            // dedicated Anthropic key when one is set.
            let mut config = config.clone();
            if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
                if !key.trim().is_empty() {
                    config.api_key = key;
                }
            }
            Ok(Box::new(AnthropicProvider::from_config(&config)))
        }
        "ollama" => Ok(Box::new(OllamaProvider::from_config(config))),
        other => Err(format!(
            "unknown LLM_PROVIDER={other:?} (expected openai|xai|anthropic|ollama)"
        )),
    }
}

/// Build one embedding provider from `EMBEDDING_PROVIDER` (default `openai`).
///
/// # Errors
///
/// Returns a message when `EMBEDDING_PROVIDER` is unknown or the local
/// model fails to download/initialise.
fn build_embedding_provider(config: &LlmConfig) -> Result<Box<dyn EmbeddingProvider>, String> {
    let name = std::env::var("EMBEDDING_PROVIDER").unwrap_or_else(|_| "openai".to_string());
    match name.trim().to_lowercase().as_str() {
        "" | "openai" => Ok(Box::new(OpenAiEmbeddingProvider::from_config(config))),
        "local" => {
            eprintln!(
                "engram: warning: local embeddings are 384-dim but the store \
                 expects 1536-dim OpenAI vectors — do NOT mix stores built with \
                 different embedding providers"
            );
            Ok(Box::new(
                LocalEmbeddingProvider::new().map_err(|error| error.to_string())?,
            ))
        }
        other => Err(format!(
            "unknown EMBEDDING_PROVIDER={other:?} (expected openai|local)"
        )),
    }
}

/// Assemble all pipelines behind one [`Cli`] facade.
///
/// One handle is built (embedded disk/memory openers perform real I/O here)
/// and cloned for the three pipelines plus the facade's own stats handle.
async fn build_cli(config: &MnemosConfig) -> Result<Arc<Cli>, String> {
    let storage = Storage::from_config(&config.storage)
        .await
        .map_err(|error| error.to_string())?;

    // Each ML model needs its own `Box<dyn LlmProvider>`; the providers are
    // cheap HTTP clients, so one is built per model.
    // Detector and splitter also get their own providers (auto-wired, not opt-in).
    let detector_for_ingestion = ContradictionDetector::new(build_chat_provider(&config.llm)?);
    let splitter_for_cli = MitosisSplitter::new(build_chat_provider(&config.llm)?);
    let detector_for_cli = ContradictionDetector::new(build_chat_provider(&config.llm)?);

    let ingestion = IngestionPipeline::new(
        Box::new(LlmEmotionalTagger::new(build_chat_provider(&config.llm)?)),
        Box::new(LlmImportanceScorer::new(build_chat_provider(
            &config.llm,
        )?)),
        Box::new(LlmConceptExtractor::new(build_chat_provider(&config.llm)?)),
        build_embedding_provider(&config.llm)?,
        storage.clone(),
    )
    .with_contradiction_detector(detector_for_ingestion);
    let retrieval = RetrievalPipeline::new(
        storage.clone(),
        StimulationEngine::new(config.stimulation.clone(), EdgeWeights::defaults()),
        EdgeWeights::defaults(),
        // Second embedder instance: pipelines each own theirs. For `local`
        // this loads the on-device model twice (the Hub cache is reused, so
        // the weights download only once).
        build_embedding_provider(&config.llm)?,
    );
    let consolidation =
        ConsolidationPipeline::new(storage.clone(), config.consolidation.clone());
    Ok(Arc::new(
        Cli::new(ingestion, retrieval, consolidation, storage)
            .with_contradiction_detector(detector_for_cli)
            .with_mitosis_splitter(splitter_for_cli),
    ))
}

/// Run the binary; returns the process exit code.
async fn run(argv: Vec<String>) -> i32 {
    let command = parse_args(&argv);
    match command {
        Command::Help => {
            println!("{}", usage());
            0
        }
        Command::Invalid {
            message,
        } => {
            eprintln!("engram: error: {message}\n{}", usage());
            2
        }
        Command::McpServer | Command::McpTools | Command::Ingest {
            ..
        } | Command::Recall {
            ..
        } | Command::Reward {
            ..
        } | Command::Consolidate | Command::Stats => {
            dispatch(command).await
        }
    }
}

/// Build config + pipelines, then run one non-help [`Command`].
async fn dispatch(command: Command) -> i32 {
    let config = match MnemosConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("engram: error: failed to load config: {error}");
            return 1;
        }
    };
    eprintln!(
        "engram: storage backend=requested:{:?} effective:{:?} database={:?}",
        config.storage.backend,
        config.storage.effective_backend(),
        config.storage.database
    );
    let cli = match build_cli(&config).await {
        Ok(cli) => cli,
        Err(message) => {
            eprintln!("engram: error: {message}");
            return 1;
        }
    };
    match command {
        Command::Ingest {
            text,
        } => match cli.ingest(&text).await {
            Ok(id) => {
                println!("{id}");
                0
            }
            Err(error) => {
                eprintln!("engram: error: ingest failed: {error}");
                1
            }
        },
        Command::Recall {
            query,
            limit,
        } => match cli.recall(&query, limit).await {
            Ok(results) => {
                match serde_json::to_string(&results) {
                    Ok(json) => {
                        println!("{json}");
                        0
                    }
                    Err(error) => {
                        eprintln!("engram: error: failed to encode results: {error}");
                        1
                    }
                }
            }
            Err(error) => {
                eprintln!("engram: error: recall failed: {error}");
                1
            }
        },
        Command::Reward {
            score,
            attributions,
        } => match cli.reward(&attributions, score).await {
            Ok(()) => {
                println!("reward applied");
                0
            }
            Err(error) => {
                eprintln!("engram: error: reward failed: {error}");
                1
            }
        },
        Command::Consolidate => match cli.consolidate().await {
            Ok(report) => match serde_json::to_string(&report) {
                Ok(json) => {
                    println!("{json}");
                    0
                }
                Err(error) => {
                    eprintln!("engram: error: failed to encode report: {error}");
                    1
                }
            },
            Err(error) => {
                eprintln!("engram: error: consolidate failed: {error}");
                1
            }
        },
        Command::Stats => match cli.stats().await {
            Ok(stats) => match serde_json::to_string(&stats) {
                Ok(json) => {
                    println!("{json}");
                    0
                }
                Err(error) => {
                    eprintln!("engram: error: failed to encode stats: {error}");
                    1
                }
            },
            Err(error) => {
                eprintln!("engram: error: stats failed: {error}");
                1
            }
        },
        // `mnemos-mcp-server` returns `mnemos_core::Result`; `mnemos-mcp-tools`
        // returns `Result<_, rmcp::RmcpError>` — both map to a message + exit 1.
        Command::McpServer => match mnemos_mcp_server::run(cli).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("engram: error: mcp-server failed: {error}");
                1
            }
        },
        Command::McpTools => match mnemos_mcp_tools::run(cli).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("engram: error: mcp-tools failed: {error}");
                1
            }
        },
        Command::Help | Command::Invalid {
            ..
        } => {
            eprintln!("engram: error: unexpected command\n{}", usage());
            2
        }
    }
}

#[tokio::main]
async fn main() {
    load_env_files();
    let code = run(std::env::args().collect()).await;
    if code != 0 {
        std::process::exit(code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(words: &[&str]) -> Vec<String> {
        std::iter::once("mnemos")
            .chain(words.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn ingest_joins_words() {
        assert_eq!(
            parse_args(&argv(&["ingest", "the", "sky", "is", "blue"])),
            Command::Ingest {
                text: "the sky is blue".to_string()
            }
        );
    }

    #[test]
    fn ingest_without_text_is_invalid() {
        assert!(matches!(
            parse_args(&argv(&["ingest"])),
            Command::Invalid {
                ..
            }
        ));
    }

    #[test]
    fn recall_defaults_to_default_limit() {
        assert_eq!(
            parse_args(&argv(&["recall", "blue", "sky"])),
            Command::Recall {
                query: "blue sky".to_string(),
                limit: DEFAULT_RECALL_LIMIT
            }
        );
    }

    #[test]
    fn recall_parses_limit_before_and_after_query() {
        assert_eq!(
            parse_args(&argv(&["recall", "--limit", "3", "blue", "sky"])),
            Command::Recall {
                query: "blue sky".to_string(),
                limit: 3
            }
        );
        assert_eq!(
            parse_args(&argv(&["recall", "blue", "sky", "--limit=7"])),
            Command::Recall {
                query: "blue sky".to_string(),
                limit: 7
            }
        );
    }

    #[test]
    fn recall_without_query_or_bad_limit_is_invalid() {
        assert!(matches!(
            parse_args(&argv(&["recall"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["recall", "--limit", "3"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["recall", "sky", "--limit", "many"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["recall", "sky", "--limit"])),
            Command::Invalid {
                ..
            }
        ));
    }

    #[test]
    fn reward_parses_score_and_optional_csv() {
        assert_eq!(
            parse_args(&argv(&["reward", "0.8"])),
            Command::Reward {
                score: 0.8,
                attributions: Vec::new()
            }
        );
        assert_eq!(
            parse_args(&argv(&["reward", "-1", "0.5, 0.25,0.125"])),
            Command::Reward {
                score: -1.0,
                attributions: vec![0.5, 0.25, 0.125]
            }
        );
    }

    #[test]
    fn reward_rejects_missing_or_bad_numbers() {
        assert!(matches!(
            parse_args(&argv(&["reward"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["reward", "high"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["reward", "1.0", "0.5,nope"])),
            Command::Invalid {
                ..
            }
        ));
    }

    #[test]
    fn bare_commands_parse() {
        assert_eq!(parse_args(&argv(&["consolidate"])), Command::Consolidate);
        assert_eq!(parse_args(&argv(&["stats"])), Command::Stats);
        assert_eq!(parse_args(&argv(&["mcp-server"])), Command::McpServer);
        assert_eq!(parse_args(&argv(&["mcp-tools"])), Command::McpTools);
        assert_eq!(parse_args(&argv(&["help"])), Command::Help);
        assert_eq!(parse_args(&argv(&["--help"])), Command::Help);
        assert_eq!(parse_args(&argv(&["-h"])), Command::Help);
    }

    #[test]
    fn missing_and_unknown_commands_are_invalid() {
        assert!(matches!(
            parse_args(&argv(&[])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&[]),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["frobnicate"])),
            Command::Invalid {
                ..
            }
        ));
    }
}
