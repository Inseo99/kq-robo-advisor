# Operating Thresholds After Costs

This document defines how signal-quality Alpha Decay results may be used after
transaction costs and slippage are included. It is deliberately conservative:
smoke checks are enough to prove the pipeline works, but not enough to change
production recommendations by themselves.

## Current Cost Assumption

Default validation cost model:

- Transaction cost: 10 bps
- Slippage: 5 bps
- Trade sides: 2
- Per-event haircut: 0.30%

Formula:

```text
net edge = gross edge - ((transaction cost bps + slippage bps) / 10000) * trade sides
```

This is applied to both actual signal events and random placebo events.

## Minimum Adoption Gates

A signal-quality threshold can be promoted from research candidate to operating
candidate only if all gates below pass.

| Gate | Requirement |
|---|---|
| Sample size | At least 5,000 OOS events for broad equity tests, or all available ETF assets for ETF tests |
| OOS edge after costs | Quality-filtered actual edge remains positive after costs |
| Placebo comparison | Quality edge remains above random/placebo edge after costs |
| Raw comparison | Quality edge is not meaningfully worse than raw after costs |
| Segmentation | Result does not rely only on sell-side or illiquid mid/small segments |
| Korean market filter | Limit-move excluded result remains acceptable |
| Stability | q>=0.60 and q>=0.70 results are directionally consistent |

## Current Evidence

### Broad Equity Runs Before Costs

Top 100 / n 500 showed:

- Raw: 25,500 events, +1.669% edge, +0.855% random
- q>=0.60: 9,344 events, +1.868% edge, +0.445% random
- q>=0.70: 6,595 events, +1.867% edge, +0.402% random

Interpretation:

- q>=0.60 and q>=0.70 are both research candidates.
- q>=0.70 is cleaner on edge-minus-random.
- q>=0.60 keeps more events and may be operationally less sparse.

### Segmentation Smoke Checks

Smoke checks indicate:

- Buy-side large-cap quality filtering improved over raw.
- Sell-side mid/small quality filtering deteriorated and should not be adopted.
- ETF quality filtering runs successfully, but current smoke p-value is not enough for adoption.
- Limit-move inclusion/exclusion can now be tested explicitly.

### Costed Smoke Check

ETF top 3 / n 1 / q>=0.70 with 10bps cost + 5bps slippage x 2 sides:

- Raw edge: +0.754% gross to +0.454% net
- Quality edge: +0.779% gross to +0.479% net
- Per-event haircut: 0.300%

Interpretation:

- Cost plumbing is working.
- The smoke test is not large enough to prove an operating threshold.

## Current Operating Decision

Do not use signal quality as a hard production filter for all signals yet.

Allowed use now:

- Research/reporting label: show signal quality score as confidence context.
- Candidate filter in validation: continue testing q>=0.60 and q>=0.70.
- Conservative Alpha Decay review: prefer buy-side large-cap signals when quality is high.

Not allowed yet:

- Do not suppress all raw sell signals solely because quality is low.
- Do not apply q>=0.70 universally to mid/small-cap or sell-side signals.
- Do not claim cost-adjusted alpha until top 100 / n 500 or larger costed runs pass.

## Recommended Next Full Runs

```powershell
python tests\validation_signal_quality_alpha_decay.py --top 100 --n 500 --sweep --qualities 0.6,0.7 --signal-side buy --ticker-segment large --cost-bps 10 --slippage-bps 5 --output tests\signal_quality_buy_large_top100_n500_costed.npz

python tests\validation_signal_quality_alpha_decay.py --top 100 --n 500 --sweep --qualities 0.6,0.7 --signal-side sell --ticker-segment mid_small --cost-bps 10 --slippage-bps 5 --output tests\signal_quality_sell_midsmall_top100_n500_costed.npz

python tests\validation_signal_quality_alpha_decay.py --universe-source etf --top 7 --n 500 --sweep --qualities 0.6,0.7 --cost-bps 10 --slippage-bps 5 --output tests\signal_quality_etf_top7_n500_costed.npz
```

## Rule of Thumb for UI/Report Wording

Use:

```text
이 신호는 검증상 유망한 품질 구간에 속합니다. 비용 반영 대규모 검증 전까지는 보조 신뢰도 지표로만 사용합니다.
```

Avoid:

```text
이 신호는 비용 반영 후에도 확실한 초과수익을 냅니다.
```
