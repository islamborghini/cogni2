<p align="center">
  <img src="landing/logo.svg" alt="Cogni" width="120" />
</p>

# Cogni

A token-efficient context layer for coding agents. Cogni cuts the tokens, and the cost, an agent spends per task without lowering how often it succeeds.

![Go](https://img.shields.io/badge/Go-1.26-00ADD8?logo=go&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Stages](https://img.shields.io/badge/parts-3%2F3%20shipped-success)

## Overview

A coding agent re-sends a growing context to the model on every step: retrieved code, tool outputs, and the transcript of its own reasoning. You pay for all of it, on every turn. Cogni sits between the agent and the model API and trims what gets sent, without changing the answers the agent produces.

The project is three components, each built and measured as an independent ablation. The value is the per-component number, not one combined figure, and the bar is fixed in advance: a reduction only counts if task success holds. Efficiency here means tokens or cost at a fixed success rate, never a quality trade.

Two constraints shape the design:

- It targets a closed, frontier-model API. There are no logits, hidden states, or prefill tricks. Everything is retrieval, static analysis, or a normal API call.
- Every claim is measured by holding the work fixed and varying only the component under test, so a saving can never come from quietly doing less.

For a plain-language walkthrough of the design, see [`docs/how-it-works.md`](docs/how-it-works.md).

### Requirements and build

Cogni is a Go module (Go 1.26 or newer). The offline test suite needs no API keys:

```sh
git clone https://github.com/islamborghini/cogni2
cd cogni2
go test ./...
```

The end-to-end benchmarks that call external services are gated behind a build tag and environment variables (see [The benchmark](#the-benchmark)), so they never run by accident.

## The three parts

| Component | What it does | Headline result |
|---|---|---|
| 1. cAST retrieval | chunk the repo at AST boundaries, embed, retrieve the top-k relevant chunks | recall@10 = 0.824, mean retrieved-code = 3554 tokens |
| 2. Skeleton-first compression | render lower-ranked chunks as signature + docstring + anchor | tokens down, retrieval and syntax intact (111 of 111 parse) |
| 3. History compression | summarize older observations between turns, keep the latest verbatim | about 20% lower net cost, success held by construction |

**1. cAST retrieval** (`internal/chunk`, `internal/embed`, `internal/index`). The repository is split along abstract syntax tree boundaries into functions, methods, and signature-only class headers, embedded, and served by exact cosine search over a local SQLite store. The agent gets a few relevant definitions instead of whole files. Full report: [`bench/results/stage1.md`](bench/results/stage1.md).

**2. Skeleton-first compression** (`internal/skeleton`). This changes only how retrieved chunks are rendered, not what is retrieved. The top chunks are sent in full; the rest are sent as a signature plus a first-paragraph docstring, with the body replaced by a placeholder and a `path:start-end` anchor for re-reading on demand. Bodies are omitted, never pruned mid-statement, so a skeleton is always valid code. Because retrieval is untouched, recall is unchanged by construction. Full report: [`bench/results/stage2.md`](bench/results/stage2.md).

**3. History compression** (`internal/compress`, `internal/agent`). Between steps, older tool observations are summarized under an editable guideline while the most recent action and observation are kept verbatim. Compression runs only when the running history exceeds a budget, and each observation is summarized at most once. The summarizer runs on a cheaper model than the agent, so net cost drops about 20% even though the raw token count stays close to flat. It is measured by replaying recorded trajectories, so the final answer, and success, are identical by construction. Full report: [`bench/results/stage3.md`](bench/results/stage3.md).

## The benchmark

Every number comes from a frozen benchmark, committed once so results stay comparable across changes.

- **Target**: `django/django` pinned at `1651351386ab31d8ae492c8a4922797714ca97d1` (release 4.2.3).
- **Task set**: 20 natural-language queries with hand-labeled ground-truth code spans, in `bench/tasks.yaml`. The set is frozen; only the loader changes if the corpus is swapped.
- **Instrument**: a tiktoken-compatible token meter that buckets every counted string by source (retrieved code, history, system, output), so each component's effect is attributable. Cost combines those tokens with per-model prices.
- **Discipline**: success is held fixed while only the component under test varies, so retrieval recall and end-to-end answers are unchanged by construction. Results are reported in more than one framing, including the least flattering one. Committed reports live in `bench/results/`.

The evals are gated behind the `eval` build tag and `COGNI_EVAL=1`, and most need an embedding key. Pin the target repo once:

```sh
git clone https://github.com/django/django
git -C django checkout 1651351386ab31d8ae492c8a4922797714ca97d1
```

Then reproduce each component:

```sh
# Part 1: embed the corpus and score retrieval (cached after the first run)
VOYAGE_API_KEY=your-key COGNI_EVAL=1 COGNI_BENCH_REPO="$PWD/django" \
  go test -tags eval ./bench/ -run Retrieval -v -timeout 30m

# Part 2: reuse the cached index and sweep the skeleton budget
VOYAGE_API_KEY=your-key COGNI_EVAL=1 COGNI_BENCH_REPO="$PWD/django" \
  go test -tags eval ./bench/ -run Skeleton -v -timeout 30m

# Part 3: replay recorded trajectories (no key or index needed)
COGNI_EVAL=1 go test -tags eval ./bench/ -run Replay -v
```

The index is cached under `$COGNI_INDEX_DIR` (default `$TMPDIR/cogni2-index`), keyed by repo, embedder, and budget, so the corpus is embedded only once. The Part 3 replay reads recorded trajectories under `bench/runs/` (produced by a live run and gitignored), so for a fresh clone the committed `bench/results/stage3.md` is the reference figure.

## Contributing

Contributions are welcome through issues and pull requests.

**Setup.** Build and test with `go test ./...`. The offline suite needs no keys and must stay green. Format with `gofmt` and keep `golangci-lint` clean; both run in CI on every pull request.

**Conventions.**

- Keep pull requests small and focused, one logical change each. Commit messages are a single line prefixed with `feat:`, `fix:`, `docs:`, `test:`, or `chore:`.
- Some shapes are load-bearing and must not change: the `RetrievedChunk` contract in `internal/retrieve`, the meter buckets in `internal/meter`, and the frozen task set in `bench/tasks.yaml`. Carry any extra data out of band rather than mutating these.
- A new component or capability ships with its own measured benchmark result before it is merged. Efficiency means tokens or cost at a fixed success rate, so a reduction that lowers success is not a result.

**Project layout.**

```
internal/retrieve/   locked data contract: RetrievedChunk + Retriever
internal/meter/      tiktoken-compatible, bucketed token accounting
internal/parse/      tree-sitter parsing (ported, Apache-2.0)
internal/chunk/      cAST AST chunker (part 1)
internal/embed/      Embedder: Voyage, OpenAI-compatible, or Fake (part 1)
internal/index/      file watcher, SQLite vector store, retriever (part 1)
internal/skeleton/   skeleton-first compression (part 2)
internal/compress/   history compression (part 3)
internal/agent/      multi-turn loop and OpenAI-compatible model client (part 3)
bench/               frozen task set, gated evals, committed results
docs/                design and usage guides
```

Maintained by [@islamborghini](https://github.com/islamborghini).

## License

Apache-2.0. Portions are ported from the original Cogni project under the same license.
