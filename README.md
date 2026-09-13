# Engram Mnemos — Cognitive Memory Operating System

> Embedded HelixDB + Rust memory for AI — episodic memory that learns.

Engram Mnemos (MNEMOS) is a modular Rust workspace that gives an AI persistent, learnable memory via embedded HelixDB 3.0.0: engrams with embeddings on-node, concepts via LLM extraction, CRR resonance scoring, stimulation waves, learnable edge weights (Adam), contradiction detection, mitosis splitting, and identity crystallization — all behind a unified CLI + MCP (stdio + HTTP).

**Binary:** `engram` — `ingest` / `recall` / `reward` / `consolidate` / `stats` / `mcp-server` / `mcp-tools` / `serve` (`:4545/mcp`, `:4545/mcp/tools`, `:4545/mcp/cli` + `GET /telemetry`, `/tools`, `/health`).

**Get the binary from CI:** every push to `master` builds `engram` (`x86_64-unknown-linux-gnu`) via `wild` linker — download from **Actions → ci → Artifacts → engram-linux-x86_64** (no login needed for public repo). Or build locally: `cargo build -p mnemos-app --release` → `target/release/engram`.

```sh
# storage is embedded by default (no server) — disk at ./data/helix
cp .env.example .env  # then set OPENAI_API_KEY / ANTHROPIC_API_KEY etc.
cargo run -p mnemos-app -- ingest "The Uganda ICT Hub is attracting VC"
cargo run -p mnemos-app -- recall "infrastructure" --limit 5
./target/release/engram serve  # persistent daemon: MCP + /telemetry on :4545
```

## MCP endpoints

One daemon serves three MCP surfaces (same pipelines, same learning state):

| Endpoint | MCP tools | Notes |
|---|---|---|
| `POST /mcp` | `recall`, `store`, `contradiction_check`, `consolidate` | Tool-Call Protocol (4 tools) |
| `POST /mcp/tools` | `engram_ingest`, `engram_recall`, `engram_reward`, `engram_consolidate`, `engram_stats`, `help` | one tool per function + 2-layer help |
| `POST /mcp/cli` | `engram_cli` (single tool) | whole CLI behind one tool: `{"command":"help"}` lists, `{"command":"help","args":["recall"]}` details per command |
| `GET /tools` | — | machine-readable catalog of every tool on every endpoint above |
| `GET /health`, `POST /cli`, `GET /telemetry*` | — | daemon RPC, liveness, telemetry |

## MCP client config (env style)

Local clients (Claude Desktop, Cursor, opencode, zeroclaw on the same machine) use plain HTTP — paste into your MCP config and restart the client:

```json
{
  "mcp": {
    "engram": {
      "type": "remote",
      "url": "http://127.0.0.1:4545/mcp/tools",
      "enabled": true,
      "timeout": 60000
    }
  }
}
```

Use `http://127.0.0.1:4545/mcp` for the 4 protocol tools, or `http://127.0.0.1:4545/mcp/cli` for the single `engram_cli` tool. If `MNEMOS_MCP_TOKEN` is set, add `"headers": {"Authorization": "Bearer <token>"}`.

See `.env.example`, `PROJECT-PLAN.md`, and `docs/parallel-recall-risks.md`.
