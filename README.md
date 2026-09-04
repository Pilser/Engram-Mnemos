# Engram Nmose — Cognitive Memory Operating System

> Embedded HelixDB + Rust memory for AI — episodic memory that learns.

Engram Nmose (MNEMOS) is a modular Rust workspace that gives an AI persistent, learnable memory via embedded HelixDB 3.0.0: engrams with embeddings on-node, concepts via LLM extraction, CRR resonance scoring, stimulation waves, learnable edge weights (Adam), contradiction detection, mitosis splitting, and identity crystallization — all behind a unified CLI + MCP (stdio + HTTP).

**Binary:** `mnemos` — `ingest` / `recall` / `reward` / `consolidate` / `stats` / `mcp-server` / `mcp-tools` / `mcp-http` (`:4545/mcp`, `:4545/mcp/tools`, `:4545/mcp/cli` + `GET /telemetry`).

**Get the binary from CI:** every push to `main` builds `mnemos` (`x86_64-unknown-linux-gnu`) via `wild` linker — download from **Actions → ci → Artifacts → mnemos-linux-x86_64** (no login needed for public repo). Or build locally: `cargo build -p mnemos-app --release`.

```sh
# storage is embedded by default (no server) — disk at ./data/helix
cp .env.example .env  # then set OPENAI_API_KEY / ANTHROPIC_API_KEY etc.
cargo run -p mnemos-app -- ingest "The Uganda ICT Hub is attracting VC"
cargo run -p mnemos-app -- recall "infrastructure" --limit 5
cargo run -p mnemos-app -- mcp-http  # serves MCP + /telemetry on :4545
```

See `.env.example`, `PROJECT-PLAN.md`, and `docs/parallel-recall-risks.md`.
