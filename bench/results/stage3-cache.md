# Stage 3 under prompt caching — does the cost win survive?

Target: `https://github.com/django/django` @ `1651351386ab31d8ae492c8a4922797714ca97d1`

## Run

- replayed 20 recorded baseline trajectories under prefix caching
- agent model `openai/gpt-oss-120b`, summarizer `llama-3.1-8b-instant`; cache modeled at message granularity
- history budgets swept: [2000 1500 1000 500]

Same fixed-trajectory replay as the Stage 3 result, but input is billed under **prefix caching**: each turn reuses the cache up to the first message that changed, and pays full price after it. The uncompressed baseline only appends, so it keeps a **74% cache hit rate**; compression rewrites an older observation, which moves the first change earlier and re-bills the tail. The question is whether the net-cost reduction is real on a caching API.

Net-cost reduction is read three ways:
- **no caching**: what the token replay reports (cache ignored).
- **Groq 0.5**: cached input billed at half price (the free-tier eval's own discount).
- **frontier 0.1**: cached read at ~a tenth — where the cache is worth most and a bust hurts most.

| history budget | cache hit (base→treat) | net cost: no caching | Groq 0.5 | frontier 0.1 | engaged |
|---:|---:|---:|---:|---:|---:|
| 2000 | 74% → 67% | +5.8% | +0.2% | -12.6% | 9/20 |
| 1500 | 74% → 67% | +4.0% | -2.6% | -16.9% | 11/20 |
| 1000 | 74% → 67% | +3.9% | -2.8% | -17.3% | 11/20 |
| 500 | 74% → 66% | +4.4% | -2.5% | -17.5% | 11/20 |

Positive = cheaper than the uncompressed baseline; negative = more expensive. A column that flips from positive (no caching) to negative (with caching) is the prompt-cache trap: compression saves cheap, already-cached tokens and pays full price to recompute the tail it moved.
