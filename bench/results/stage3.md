# Stage 3 — ACON-style history compression (trajectory replay)

Target: `https://github.com/django/django` @ `1651351386ab31d8ae492c8a4922797714ca97d1`

## Run

- replayed 20 recorded baseline trajectories
- compressed-observation size modeled as: anchor + first 30 words (deterministic, no LLM)
- history budgets swept: [2000 1500 1000 500]

## Result: ~20% lower net cost, success held by construction

Measured by **replaying fixed recorded agent trajectories** and changing only whether accumulated history is compressed — the Stage 1/2 discipline (fixed inputs, vary the rendering). The action sequence is identical in both arms, so the final answer and success are unchanged **by construction**; the token delta is pure compression effect, free of the run-to-run nondeterminism that dominates a live A/B. Whether compressed context would change the agent's *decisions* is the separate end-to-end question (deferred).

Read three ways, because they say different things:
- **net cost** (headline): overhead priced on the cheap compressor model, agent tokens on the agent model — the bill you actually pay.
- **gross tokens**: how much smaller the (expensive) agent context gets, ignoring overhead.
- **net tokens**: raw count with the summarizer's tokens charged at face value — near break-even on this short-horizon set, because the one-time summarization overhead barely amortizes.

- **uncompressed mean total**: 10997 tokens/trajectory

## History-budget sweep

| history budget | net cost | gross tokens | net tokens | mean overhead | trajectories compressed |
|---:|---:|---:|---:|---:|---:|
| 2000 | **19.4%** | 12.5% | -7.5% | 2608 | 9/20 |
| 1500 | **19.1%** | 13.1% | -14.3% | 3082 | 11/20 |
| 1000 | **18.9%** | 13.1% | -14.6% | 3181 | 11/20 |
| 500 | **20.0%** | 13.7% | -14.4% | 3248 | 11/20 |

Percentages are macro-averages of per-trajectory reductions vs. the uncompressed replay. The win is in **cost**, not raw token volume: compression shrinks the expensive agent context ~14% and offloads summarization to a model several times cheaper. Compressed observation size is modeled deterministically (anchor + first words); real-LLM summary validation is a separate step.
