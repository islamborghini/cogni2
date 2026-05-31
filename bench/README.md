# bench

The frozen evaluation set and per-stage results.

## Frozen task set (Stage 1)

A FROZEN set of 20 tasks, committed once. Each task is a natural-language query
plus a hand-labeled list of ground-truth relevant file paths / line ranges. All
three stages run against this identical set so the numbers are comparable. Swap
in a SWE-bench Lite subset for published numbers later — only the task loader
changes, not the components.

## Results

- `results/stage1.md` — recall@10 + mean retrieved_code tokens (the baseline).
- `results/stage2.md` — recall@10 (must not drop) + token reduction vs stage1 +
  syntactic-validity check.
- `results/stage3.md` — success rate + mean total tokens (per bucket) +
  pct_total_reduction vs an uncompressed-history baseline at the same success.

## Runner note

Stage 3 needs a multi-turn agent harness that runs the frozen set under two
conditions (uncompressed-history baseline vs context-layer) and meters tokens by
bucket. The original Cogni project shipped a Claude Code A/B harness whose
transcript-capture, resume, rate-limit detection, and scoring are reusable; its
condition conductor was MCP-specific and will be rebuilt here for the
baseline-vs-context-layer comparison when Stage 3 lands.

Per-task token records are written to `runs/stage<N>/<task_id>.json` (gitignored;
only `results/` is committed).
