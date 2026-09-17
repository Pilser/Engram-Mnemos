# Engram Mnemos (MNEMOS)

> Persistent, **learnable** memory for AI agents — embedded HelixDB + Rust, no server required.

[![ci](https://github.com/Pilser/Engram-Mnemos/actions/workflows/ci.yml/badge.svg)](https://github.com/Pilser/Engram-Mnemos/actions/workflows/ci.yml)

Engram Mnemos is a modular Rust workspace (28 crates) that gives an AI agent episodic
memory that **improves from feedback**. Memories are stored in an embedded HelixDB graph
(no external database), recalled by resonance, and shaped over time by rewards — so
relevant memories surface higher and irrelevant ones stop surfacing.

One binary (`engram`) exposes the same engine over a **shell CLI**, a **one-to-one HTTP
API**, and **three MCP surfaces** (for Claude Desktop, Cursor, opencode, …).

---

## Table of contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Shell CLI](#shell-cli)
- [HTTP API](#http-api)
- [MCP surfaces](#mcp-surfaces)
- [Learning model](#learning-model)
- [Recall semantics](#recall-semantics)
- [Sequential memory](#sequential-memory)
- [Configuration](#configuration)
- [Storage backends](#storage-backends)
- [Telemetry](#telemetry)
- [Development](#development)
- [Docs](#docs)

---

## What it does

| Capability | Detail |
|---|---|
| **Episodic memory** | `ingest` stores a text episode as an `Engram` node (embedding on the node, timestamp, emotional charge, importance, decay rate). |
| **Semantic recall** | `recall` embeds the query and scores memories with **CRR** (Cognitive Resonance Retrieval), then spreads activation across the graph. |
| **Learns from feedback** | `reward` (score `-1.0 … 1.0`) updates a per-memory reward signal and the learnable edge weights, so ranking changes with use. |
| **Honest answers** | When nothing clears the relevance floor, `recall` answers **`I don't know`** instead of returning an off-topic match. |
| **Sequential / story memory** | `ingest --seq` chains memories with `TemporalSequence` edges; `recall --follow-seq` walks the chain up/down/both. |
| **Concepts & identities** | An LLM extracts concepts per episode (`Recalls`/`AbstractsTo` edges); consolidation can split overloaded concepts (mitosis) and crystallize identities. |
| **Contradiction detection** | Embedding pre-filter + LLM verification link contradicting memories and down-weight them. |
| **Consolidation ("sleep")** | Decay, prune, compress, promote, plus optional aggressive mitosis + identity + contradiction passes. |
| **Self-tuning reranker** | A tiny learned reranker (Option B) trains online from every reward and blends in automatically as data accumulates. |
| **Three interfaces** | Shell CLI, mirrored HTTP API, and MCP (stdio + HTTP) — all sharing one in-process engine. |
| **Embedded storage** | Disk (default), in-memory, S3-compatible object storage, or a HelixDB HTTP server. |
| **Pluggable providers** | Chat: OpenAI, xAI, DeepSeek, Anthropic, Ollama. Embeddings: OpenAI-compatible or local (fastembed). |

---

## How it works

```
                 ┌────────────── engram serve (one process, one HelixDB handle) ──────────────┐
   shell CLI ──► │  /cli RPC        HTTP API (/ingest /recall /reward …)     MCP (/mcp*)      │
   HTTP API  ──► │                                    │                                      │
   MCP       ──► │                                    ▼                                      │
                 │   ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐               │
                 │   │  Ingestion  │   │  Retrieval   │   │  Consolidation  │               │
                 │   │ tag→score→  │   │ CRR + wave + │   │ decay/prune/    │               │
                 │   │ embed→extract│  │ reranker     │   │ promote/mitosis │               │
                 │   └──────┬──────┘   └──────┬───────┘   └────────┬────────┘               │
                 │          └──────────────┬──┴────────────────────┘                        │
                 │                  ┌──────▼───────┐   ┌──────────────────┐                 │
                 │                  │   Storage    │   │ EdgeWeights      │                 │
                 │                  │  (HelixDB)   │   │ (Adam) + reranker│                 │
                 │                  └──────────────┘   └──────────────────┘                 │
                 └───────────────────────────────────────────────────────────────────────────┘
```

- **Graph model** — `Engram` nodes (embedding on-node), `Concept`, `Identity`; edges
  `Recalls`, `AbstractsTo`, `Defines`, `Reinforces`, `TemporalSequence`, `Contradicts`,
  `SpawnedFrom`.
- **Ingestion** (`mnemos-ingestion`) runs emotional tagging, importance scoring,
  embedding, and concept extraction concurrently, writes the `Engram`, links concepts,
  and bumps `source_count`.
- **Retrieval** (`mnemos-retrieval`) scores candidates with CRR, applies the learned
  reranker (shadow until trusted), applies the relevance floor, then spreads activation
  across `Reinforces` / `TemporalSequence` / `Contradicts` using learned edge weights.
- **Consolidation** (`mnemos-consolidation` + `mnemos-mitosis` + `mnemos-contradiction`)
  performs the maintenance ("sleep") passes.
- **Everything shares one storage handle** — embedded HelixDB has a single writer owner,
  so the daemon never opens the DB twice.

---

## Quick start

**Get the binary.** Every push to `master` builds `engram` for
`x86_64-unknown-linux-gnu` (via the `wild` linker):
**Actions → ci → Artifacts → `engram-linux-x86_64`**. Or build it yourself:
`cargo build -p mnemos-app --release` → `target/release/engram`.

```sh
# 1. configure providers (chat + embeddings). Storage is embedded by default.
cp .env.example .env
$EDITOR .env          # set LLM_PROVIDER / OPENAI_API_KEY (or DEEPSEEK_API_KEY, …)

# 2. start the daemon (MCP + HTTP API + telemetry on :4545).
#    On first boot it creates the data root and the vector index automatically.
./engram serve

# 3. in another shell — the CLI is a thin client to the running daemon:
./engram ingest "The Uganda ICT Hub is attracting VC"
./engram recall "infrastructure" --limit 5
```

> CLI commands require the daemon: they only hit `POST /cli`. If it is not running they
> print `engram daemon is not running … start it with engram serve`.

---

## Shell CLI

```
usage: engram <command> [args]
capability: episodic memory with optional sequential story chains (TemporalSequence);
            every recall must be rewarded via reward (use recall_id from recall);
            recall answers "I don't know" when nothing is relevant;
            learning is automatic (just recall+reward) — run status to see its state

  ingest <text...>  store one episodic memory (no sequence needed)
    optionally sequential: --seq <prev_id> links this engram after <prev_id> via
      TemporalSequence (use previous ingest's returned id); --seq-pos N groups many
      engrams at the same position N before the next (fan-out at a level)
  recall <query...> [--limit N]  recall top-N memories as JSON (default 5); sequential
      results show [sequential: pos=N head=...]; prints "I don't know" when no hit
      clears the relevance floor (MNEMOS_RECALL_MIN_SIM, default 0.70)
    optionally walk chain: --follow-seq <id> start from engram <id> (ignore query),
      --depth N steps (default 10), --dir up|down|both (default down)
  reward <score -1.0 to 1.0> [--recall-id N | attributions csv]  reward a recall based
      on relevancy so memory learns (edge weights via Adam) — must reward each recall
      (1.0 relevant positive, -1.0 irrelevant negative, 0 no-op)
  consolidate   run one consolidation cycle
  stats         print memory stats as JSON
  status        check embedding + LLM reachability, memory stats, and learning state
```

Operator commands (not shown by `engram --help`, see `engram --help-all`):

| Command | Purpose |
|---|---|
| `setup` | Create the `Engram.embedding` vector index (dimension from `EMBEDDING_DIM`). Also runs automatically on first boot. |
| `train-reranker` | Batch-train the learned reranker from the feedback log (online learning is automatic; this is optional). |
| `mcp-server` / `mcp-tools` | Serve MCP over **stdio** (for desktop clients). |
| `serve` (`daemon`, `up`) | Persistent daemon: HTTP API + MCP + telemetry + background tasks. |

**Typical agent loop:** `recall` → use the memories → `reward` the `recall_id` by
relevancy → (occasionally) `consolidate`.

---

## HTTP API

The daemon mirrors the **entire CLI** one-to-one as HTTP endpoints, served in-process
(same process, same DB handle). Every endpoint accepts **GET (query string)** and
**POST (JSON body)**.

| Endpoint | Mirrors |
|---|---|
| `GET /health` | daemon liveness (`{"ok":true,"data":{"status":"up"}}`, no DB touch) |
| `ANY /ingest` | `engram ingest <text...> [--seq <prev_id>] [--seq-pos N]` |
| `ANY /recall` | `engram recall <query...> [--limit N]` / `--follow-seq …` |
| `ANY /reward` | `engram reward <score> [--recall-id N | attributions csv]` |
| `ANY /consolidate` | `engram consolidate` |
| `ANY /setup` | `engram setup` |
| `ANY /stats` | `engram stats` |
| `ANY /status` | `engram status` |
| `ANY /train-reranker` | `engram train-reranker` |
| `ANY /help` | `engram --help` (machine-readable endpoint list) |

Response envelope (identical shape everywhere):

```json
{"ok": true,  "data": <cli-json>}
{"ok": false, "error": "<same message the CLI would print>"}
```

HTTP status: `200` ok · `400` bad input · `405` wrong method · `500` internal.
**`recall` returns `recall_id`** so a remote caller can `reward` exactly like the local
CLI (learning semantics are preserved; an unrewarded remote recall is identical to an
unrewarded local recall).

```sh
curl -s 'http://127.0.0.1:4545/recall?query=blue%20sky&limit=5'
curl -s -X POST http://127.0.0.1:4545/reward \
     -H 'content-type: application/json' -d '{"score":1.0,"recall_id":7}'
```

Full field-by-field reference: [`engramapi.txt`](engramapi.txt).

---

## MCP surfaces

One daemon serves three MCP surfaces (same pipelines, same learning state):

| Endpoint | Tools | Notes |
|---|---|---|
| `POST /mcp` | `recall`, `store`, `contradiction_check`, `consolidate` | Tool-Call Protocol (4 tools) |
| `POST /mcp/tools` | `engram_ingest`, `engram_recall`, `engram_reward`, `engram_consolidate`, `engram_stats`, `engram_status`, `help` | one tool per function + two-layer help |
| `POST /mcp/cli` | `engram_cli` | the whole CLI behind one tool: `{"command":"help"}` lists, `{"command":"help","args":["recall"]}` details |
| `GET /tools` | — | machine-readable catalog of every tool on every endpoint |
| `POST /cli` | — | daemon CLI RPC (`{"command": …}`) used by the shell CLI |
| `GET /telemetry*` | — | telemetry (see below) |

MCP client config (e.g. Claude Desktop, Cursor, opencode):

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

Use `…/mcp` for the 4 protocol tools or `…/mcp/cli` for the single `engram_cli` tool.
If `MNEMOS_MCP_TOKEN` is set, add `"headers": {"Authorization": "Bearer <token>"}`.

---

## Learning model

Learning is **automatic** — the agent only calls `recall` and `reward`.

1. **Per-memory reward signal** — each reward updates the recalled memories' `reward_score`
   as an exponential moving average (`α = 0.3`), clamped to `[-1, 1]`.
2. **Context gating** — the reward is stored with the query embedding it was given in and
   applied as `reward_score × cos(query, reward_context) × decay(exp(-0.01·days))`, so a
   negative on one topic does not bury the memory for a different topic, and stale
   feedback fades.
3. **Learnable edge weights** — `reward` also runs one Adam step over the edge-type weights
   (`Recalls`, `AbstractsTo`, `Reinforces`, `TemporalSequence`, `Contradicts`, `Defines`,
   `SpawnedFrom`, `Recurrent`) using the attributions of the recall that was rewarded.
   Weights persist to `MNEMOS_WEIGHTS_FILE`.
4. **Feedback log** — every recall logs the candidates it showed (with their features and
   position) and every reward logs its score to `MNEMOS_FEEDBACK_LOG` (JSONL).
5. **Learned reranker (Option B)** — a tiny linear pairwise model over 9 CRR features
   (seed shipped at `models/reranker-seed.json`). It fine-tunes **online** on every reward
   and is **hot-reloaded** (no restart) when retrained. It is blended into CRR as
   `crr × (1 − α + α·2·p)` where `p` is the model's relevance probability.

**Blend α is self-tuning.** With `MNEMOS_RERANKER_ALPHA=auto` (default), α ramps linearly
from `0.0` to `MNEMOS_RERANKER_MAX_ALPHA` (`0.5`) as reward events accumulate to
`MNEMOS_RERANKER_MIN_PAIRS` (`500`). So the system stays pure CRR (shadow) until enough
real feedback exists, then gradually trusts the learned model. `status` shows the current
state:

```json
"learning": {
  "model": "local",            // seed | local
  "reward_events": 8,
  "alpha": 0.008,
  "alpha_mode": "auto",
  "min_pairs_for_full_alpha": 500,
  "online_finetuning": true,
  "state": "active"            // shadow | active
}
```

Set `MNEMOS_RERANKER_ALPHA=0.0` for a permanent kill-switch, or a number to pin it.

---

## Recall semantics

Candidates from the vector search are scored by **CRR**:

```
resonance = semantic_sim
          × recency_weight              (exp(-decay_rate · days), floored)
          × (1 + |emotional_charge|)
          × identity_alignment          (Engram → Recalls → Concept → Defines → Identity)
          × contradiction_factor        (0.5 if flagged)
          × reward_factor               (learned, context-gated, decayed)
```

Then:

- the learned reranker is blended in (when α > 0),
- the **relevance floor** drops hits whose *effective* relevance
  (`semantic_sim × reward_factor`) is below `MNEMOS_RECALL_MIN_SIM` (`0.70`),
- the wave spreads activation across `Reinforces` / `TemporalSequence` / `Contradicts`
  using the learned edge weights and merges the discovered memories,
- results are sorted by resonance and truncated to `--limit`.

If nothing clears the floor, the answer is **`I don't know`**.

---

## Sequential memory

Memories can be chained into stories or ordered lists via `TemporalSequence` edges
(`earlier → later`):

```sh
id1=$(./engram ingest "Chapter 1: the hero leaves the village")
id2=$(./engram ingest "Chapter 2: the hero meets the wizard" --seq "$id1")
id3=$(./engram ingest "Chapter 3: the wizard gives a map"   --seq "$id2" --seq-pos 2)

./engram recall --follow-seq "$id1" --depth 5 --dir down
# {"sequential_chain":[id2,id3],"start_id":id1,"depth":5,"dir":"down"}
```

- `--seq <prev_id>` links a new engram after `prev_id`.
- `--seq-pos N` groups several engrams at the same position before the next (fan-out).
- Sequential recall results carry a `[sequential: pos=N head=… prev=… next=…]` annotation.

---

## Configuration

All configuration is environment variables (`.env` is auto-loaded next to the binary and
in the working directory). See [`.env.example`](.env.example) for the full annotated list.

**Chat LLM** — `LLM_PROVIDER` = `openai` | `xai` | `deepseek` | `anthropic` | `ollama`;
`OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL`, or `DEEPSEEK_API_KEY` +
`DEEPSEEK_BASE_URL`, or `XAI_API_KEY`, or `ANTHROPIC_API_KEY`.

**Embeddings** — `EMBEDDING_PROVIDER` = `openai` | `local`; `EMBEDDING_MODEL`,
`EMBEDDING_BASE_URL`, `EMBEDDING_DIM` (must match the store: `384` local MiniLM, `1536`
OpenAI).

**Key learning / recall knobs**

| Variable | Default | Meaning |
|---|---|---|
| `MNEMOS_RECALL_MIN_SIM` | `0.70` | Effective-relevance floor; below it → `I don't know` (`0.0` disables). |
| `MNEMOS_RECALL_MIN_SCORE` | `0.0` | Secondary resonance floor. |
| `MNEMOS_RECALL_WAVE` | on | Spreading-activation wave; `0` = single-pass CRR. |
| `MNEMOS_FEEDBACK_LOG` | `./data/helix/feedback.jsonl` | Feedback log for reranker training. |
| `MNEMOS_RERANKER_MODEL` | `./data/helix/reranker.json` | Local learned model. |
| `MNEMOS_RERANKER_SEED` | `./models/reranker-seed.json` | Shipped prior. |
| `MNEMOS_RERANKER_ALPHA` | `auto` | `auto` ramp · `0.0` shadow/kill-switch · number pins. |
| `MNEMOS_RERANKER_MIN_PAIRS` | `500` | Reward events for full α (auto). |
| `MNEMOS_RERANKER_MAX_ALPHA` | `0.5` | α ceiling (auto). |
| `MNEMOS_RERANKER_ONLINE` | `1` | Online fine-tuning on each reward. |
| `MNEMOS_RERANKER_LR` | `0.02` | Online learning rate. |
| `MNEMOS_WEIGHTS_FILE` | `./data/helix/mnemos-weights.json` | Persisted edge weights. |
| `MNEMOS_SETUP_ON_START` | `1` | First-boot data-root + vector-index bootstrap. |
| `MNEMOS_MCP_HOST` / `MNEMOS_MCP_PORT` | `127.0.0.1` / `4545` | Bind address. Use `0.0.0.0` to accept remote/bridge callers. |
| `MNEMOS_MCP_TOKEN` | — | Optional bearer token for all `/mcp/*` routes. |
| `MNEMOS_CONSOLIDATE_INTERVAL_SECS` | `0` | Background aggressive consolidation interval (`0` = off). |

---

## Storage backends

`MNEMOS_BACKEND` = `disk` (default) · `memory` · `object` · `http`.

- **disk** — embedded engine on `MNEMOS_DATA_ROOT` (default `./data/helix`).
- **memory** — embedded, ephemeral (tests).
- **object** — S3-compatible (`MNEMOS_OBJECT_BUCKET` / `_REGION` / `_ENDPOINT`).
- **http** — a HelixDB server at `HELIX_URL` (explicit server mode; no fallback).

Fallback chain: a disk request with an empty data root falls back to memory; an object
request without a bucket falls back to disk, then memory. `http` never falls back.

> Embedded HelixDB has a single writer owner — always run **one** `engram serve` per data
> root. (A second start now fails at the port bind *before* opening the DB, so it cannot
> corrupt a running daemon.)

---

## Telemetry

In-memory ring plus per-day JSONL files (`MNEMOS_TELEMETRY_DIR`), readable over HTTP:

`GET /telemetry` · `/telemetry/diagnose` · `/telemetry/counters` · `/telemetry/events` ·
`/telemetry/weights` · `/telemetry/system` · `/telemetry/files` ·
`/telemetry/file?date=&limit=&offset=&ok=` (and `DELETE /telemetry/file?date=`).
Set `MNEMOS_TELEMETRY=0` to disable.

---

## Development

- **Toolchain:** Rust `stable` (`rust-toolchain.toml`), workspace resolver 2.
- **Linker:** `wild` via `.cargo/config.toml` (`clang` + `-fuse-ld=wild`).
- **CI:** `.github/workflows/ci.yml` builds the release binary on every push to
  `master` and uploads the `engram-linux-x86_64` artifact. (check/clippy/test are
  currently commented out — the release build is the single gate.)
- **HelixDB:** git dependency on `helixdb/helix-db` (`branch = "main"`,
  `features = ["embedded"]`), because the published `3.0.0` crate lacks the embedded
  feature.

### Workspace (28 crates)

| Group | Crates |
|---|---|
| Core | `mnemos-core`, `mnemos-storage` |
| Pipelines | `mnemos-ingestion`, `mnemos-retrieval`, `mnemos-consolidation` |
| ML models | `mnemos-ml-trait`, `mnemos-emotional-tagger`, `mnemos-importance-scorer`, `mnemos-concept-extractor` |
| Stimulation | `mnemos-stimulation`, `mnemos-edge-weights` |
| LLM providers | `mnemos-llm-trait`, `mnemos-llm-openai`, `mnemos-llm-anthropic`, `mnemos-llm-local` |
| Embeddings | `mnemos-embedding-trait`, `mnemos-embedding-openai`, `mnemos-embedding-local` |
| Memory ops | `mnemos-contradiction`, `mnemos-mitosis`, `mnemos-reranker` |
| Interfaces | `mnemos-cli`, `mnemos-mcp-server`, `mnemos-mcp-tools`, `mnemos-mcp-protocol`, `mnemos-mcp-http` |
| Ops / binary | `mnemos-telemetry`, `mnemos-app` |

---

## Docs

- [`engramapi.txt`](engramapi.txt) — complete HTTP API reference (every endpoint, fields, examples).
- [`PROJECT-PLAN.md`](PROJECT-PLAN.md) — architecture and phase plan.
- [`SEQUENTIAL-PLAN.md`](SEQUENTIAL-PLAN.md) — sequential-memory design.
- [`docs/parallel-recall-risks.md`](docs/parallel-recall-risks.md) — concurrency notes.
- [`.env.example`](.env.example) — every configuration knob.
