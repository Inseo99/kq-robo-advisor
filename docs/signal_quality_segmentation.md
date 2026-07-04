# Signal Quality Segmentation Smoke Checks

These checks verify that the Week 6 segmentation controls run end-to-end.
They are smoke tests, not final adoption evidence. Larger runs such as
top 100 / n 500 or top 200 / n 1000 are still required before operating
thresholds are changed.

## Implemented Axes

- Universe source: equity or ETF cache
- Ticker segment: all, large, mid_small
- Signal side: all, buy, sell
- Limit-move handling: excluded by default, optionally included
- Trading friction: transaction cost bps, slippage bps, and trade sides

## Smoke Results

| run | cost haircut | raw events | raw edge % | raw random % | quality events | quality edge % | quality random % | delta vs raw % | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| equity buy large, q>=0.70, top 12, n 10 | 0.000% | 1564 | 1.381 | 1.302 | 456 | 2.048 | 0.903 | 0.668 | 0.0333 |
| equity sell mid_small, q>=0.70, top 12, n 10 | 0.000% | 321 | -0.115 | -0.070 | 360 | -0.417 | -0.045 | -0.301 | 0.8000 |
| equity buy large with limit moves included, q>=0.70, top 8, n 5 | 0.000% | 990 | 2.499 | 2.044 | 309 | 3.610 | 1.396 | 1.111 | 0.0000 |
| ETF all signals, q>=0.70, n 5 | 0.000% | 1872 | 0.686 | 0.492 | 816 | 1.070 | 0.335 | 0.384 | 0.2500 |
| ETF all signals, q>=0.70, top 3, n 1, cost 10bps + slippage 5bps x 2 sides | 0.300% | 1022 | 0.454 | 0.363 | 801 | 0.479 | 0.109 | 0.025 | 0.5000 |

## Reading

The smoke checks support the next full validation plan:

- Buy-side large-cap signals are the strongest early candidate.
- Sell-side mid/small signals should not be assumed useful; the smoke result
  deteriorated after quality filtering.
- ETF-level quality filtering works technically and may be useful, but the
  small placebo count is not enough to make a claim.
- Limit-move inclusion/exclusion can now be compared directly with the same
  script, which is important for Korean-market validation.
- Cost-adjusted runs now subtract the same per-event haircut from actual and
  placebo returns, so future adoption thresholds can be evaluated after
  trading friction.

## Example Commands

```powershell
python tests\validation_signal_quality_alpha_decay.py --top 100 --n 500 --sweep --qualities 0.6,0.7 --signal-side buy --ticker-segment large --output tests\signal_quality_buy_large_top100_n500.npz
python tests\validation_signal_quality_alpha_decay.py --top 100 --n 500 --sweep --qualities 0.6,0.7 --signal-side sell --ticker-segment mid_small --output tests\signal_quality_sell_midsmall_top100_n500.npz
python tests\validation_signal_quality_alpha_decay.py --universe-source etf --top 7 --n 500 --sweep --qualities 0.6,0.7 --output tests\signal_quality_etf_top7_n500.npz
python tests\validation_signal_quality_alpha_decay.py --top 100 --n 500 --sweep --qualities 0.6,0.7 --signal-side buy --ticker-segment large --cost-bps 10 --slippage-bps 5 --output tests\signal_quality_buy_large_top100_n500_costed.npz
```
