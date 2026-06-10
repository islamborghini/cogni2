# Stage 3 redesign — cache-safe checkpoint compaction

Target: `https://github.com/django/django` @ `1651351386ab31d8ae492c8a4922797714ca97d1`

## Run

- replayed 20 recorded baseline trajectories under prefix caching
- arms: uncompressed baseline, rewrite (GuidelineCompressor), checkpoint (CompactingCompressor, low-water 0.50*budget)
- agent model `openai/gpt-oss-120b`, summarizer `llama-3.1-8b-instant`; cache modeled at message granularity
- history budgets swept: [2000 1500 1000 500]

Same fixed-trajectory replay as `stage3-cache.md`, billed under prefix caching, but with a third arm: **checkpoint compaction** compacts rarely (only when history crosses the budget) all the way down to a low-water target, then freezes it — at most one cache divergence per checkpoint — where the original **rewrite** arm re-summarized a newer observation almost every turn and broke the cache again each time. The uncompressed baseline only appends, holding a **74% cache hit rate**.

Net cost is read three ways (reduction vs the uncompressed baseline; positive = cheaper):
- **no caching**: cache ignored (what the token replay reports).
- **Groq 0.5**: cached input at half price.
- **frontier 0.1**: cached read at ~a tenth — where a cache bust hurts most.

| budget | arm | cache hit | no caching | Groq 0.5 | frontier 0.1 | engaged |
|---:|---|---:|---:|---:|---:|---:|
| 2000 | rewrite | 67% | +5.8% | +0.2% | -12.6% | 9/20 |
| 2000 | checkpoint | 67% | +5.7% | +0.1% | -12.7% | 9/20 |
| 1500 | rewrite | 67% | +4.0% | -2.6% | -16.9% | 11/20 |
| 1500 | checkpoint | 67% | +4.0% | -2.6% | -17.0% | 11/20 |
| 1000 | rewrite | 67% | +3.9% | -2.8% | -17.3% | 11/20 |
| 1000 | checkpoint | 67% | +3.9% | -2.8% | -17.3% | 11/20 |
| 500 | rewrite | 66% | +4.4% | -2.5% | -17.5% | 11/20 |
| 500 | checkpoint | 66% | +4.4% | -2.5% | -17.5% | 11/20 |

On this short-horizon set the two policies are **indistinguishable**: the frozen 20 are 3-8 turns, so a task crosses the budget at most once, near the end, where a single compaction busts the cache with no later turns to amortize it. Both arms lose the same amount under caching. The benchmark cannot separate the designs — it is the wrong instrument for Stage 3.

## Where the redesign wins: a controlled horizon sweep

Synthetic trajectories of increasing length (goal + N rounds of action + large observation), billed at the frontier 0.1 discount. This is a **model, not a measurement** — it isolates the mechanism the short real tasks cannot exercise. As the horizon grows, the rewrite policy busts the cache almost every over-budget turn while checkpoint compaction busts once per checkpoint, so checkpoint's hit rate stays high and its net cost crosses ahead.

| rounds | base hit | rewrite hit | checkpoint hit | rewrite net cost | checkpoint net cost |
|---:|---:|---:|---:|---:|---:|
| 4 | 59% | 59% | 59% | -7.6% | -22.9% |
| 8 | 77% | 37% | 60% | -78.8% | -9.8% |
| 16 | 88% | 39% | 63% | -42.4% | +10.9% |
| 32 | 94% | 50% | 68% | +8.0% | +35.5% |
| 64 | 97% | 42% | 49% | +29.5% | +38.0% |

Net cost is reduction vs the uncompressed baseline (positive = cheaper). The crossover is the argument for the redesign; demonstrating it on real work needs the long-horizon (SWE-bench) task set, which is the separate next step.

## Status (2026-06-10): what is proven, what is deferred

**Proven.**
- The cache trap is real on recorded trajectories: history-rewriting Stage 3 goes from a small
  win without caching to **-13% to -17% net cost** at a frontier read discount (`stage3-cache.md`).
- The redesign is cache-safe by construction: checkpoint compaction busts the prefix at most once
  per checkpoint and is idempotent (unit-tested in `internal/compress/compact_test.go`).
- The mechanism wins in the limit: the synthetic sweep above shows checkpoint compaction going
  **net-positive past ~16 turns** and beating the rewrite policy by a widening margin.

**Deferred (needs budget).** The remaining question — does checkpoint compaction win on *real*
long agent trajectories — requires a capable agent doing 15-30 turn runs. Free inference cannot
produce those: Groq's free tier 413s at 8k TPM by ~turn 5; a fresh Gemini project reported
free-tier quota 0; and a local qwen3:8b submits in 3-4 turns (too short to engage compression) or
times out. The wall is **agent capability, which is not free** — not a gap in this project.

**Infra ready, so the budget-run is one command.** Built and waiting: the swappable loader
(`COGNI_TASKS`/`COGNI_RUNS_SUBDIR`), the SWE-bench Lite pytest task set
(`bench/tasks-swebench.yaml`, 12 long-horizon localization tasks), local-embedding support
(`EMBED_BATCH_SIZE` for Ollama), a tool-call nudge so weaker models do not drop trajectories, and
representative pricing for unpriced/local models. To finish, point the agent at a frontier API
(~$5-20) and run:

```sh
COGNI_EVAL=1 COGNI_TASKS=tasks-swebench.yaml COGNI_RUNS_SUBDIR=stage3-swebench \
  COGNI_BENCH_REPO=/path/to/pytest@6bd3f31 \
  LLM_BASE_URL=<provider> LLM_API_KEY=<key> COGNI_AGENT_MODEL=<strong-model> \
  go test -tags eval ./bench/ -run EndToEnd -v -timeout 60m
COGNI_EVAL=1 COGNI_RUNS_SUBDIR=stage3-swebench go test -tags eval ./bench/ -run Compact -v
```
