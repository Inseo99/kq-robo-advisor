# KQ Quant Tool Upgrade Timeline

This file is the execution checklist for turning the current working app into
a portfolio-ready quant project. The rule for every week is simple: keep the
existing `server.py` app runnable while extracting tested modules into
`src/kq_tool`.

## Current Baseline

- Legacy app: `server.py` + `index.html` still run at `http://127.0.0.1:8888`.
- Validation scripts exist under `tests/validation_*.py`.
- Alpha Decay v3 is applied as a validity/recheck layer, not as a guaranteed
  alpha engine.
- New package scaffold exists under `src/kq_tool`.
- Screener and strategy validation now separate recent momentum from S2
  12-1 momentum, and the strategy tab can compare quant momentum, quant S2,
  and KOSPI on one chart through the backend `quant_compare` response.
- Backtest strategy keys, labels, colors, aliases, and KOSPI benchmark display
  metadata now live in `src/kq_tool/backtest/strategy_meta.py`.
- Frontend contract tests now verify the screener button order, strategy
  backtest button order, and screener table header order against backend
  definitions.
- API smoke strategy checks are now factored into unit-tested helpers for
  `quant`, `quant_s2`, and `quant_compare`.
- Regression checks now fail if documented `tests/unit` counts drift from
  actual pytest collection.
- Release artifact contract tests verify launcher/regression batch files and
  handoff docs keep pointing to the expected commands.
- Windows launcher contract tests now lock Python discovery order, Chrome-first
  browser launch, server browser autostart disabling, and the Python 3.14
  `hmmlearn` dependency guard.
- API error JSON shape is now centralized in
  `src/kq_tool/api/http_response.py` and reused by the legacy handler.
- `server.py` now exposes `main()`, while `src/kq_tool/__main__.py` and the
  `kq-tool` console script provide the next package-level execution entrypoint.
- `RUN_WINDOWS.ps1` now sets `PYTHONPATH` to the local `src` directory and
  launches the app through `python -m kq_tool`, while `python server.py`
  remains documented as a compatibility path.
- Local server URL creation and browser-open scheduling now live in
  `src/kq_tool/api/runtime.py`, with `server.py` reusing those runtime helpers
  from its compatibility `main()`.
- Startup banner, Yahoo Finance connectivity status, server-ready guidance,
  and browser-autostart disabled text are now runtime helpers with unit tests.
- `serve_forever()` and Ctrl+C shutdown handling are now isolated in
  `src/kq_tool/api/runtime.py`, further thinning the compatibility
  `server.py main()` path.
- Browser startup policy and server serving are now orchestrated by
  `run_server_with_browser_policy()`, leaving `server.py main()` with a single
  package helper call for that runtime phase.
- Local HTTP server construction now goes through `create_http_server()`, so
  host/port/handler wiring is tested in `src/kq_tool/api/runtime.py`.
- `Handler._api_services()` now delegates server service registry assembly to
  `build_server_api_services()`, keeping dispatcher service wiring tested in
  `src/kq_tool/api/services.py`.
- Dispatcher responses are now applied through `apply_api_response()`, moving
  the `file`/`json`/empty-response branching out of `Handler.do_GET()`.
- Empty HTTP responses now go through `Handler._empty()`, giving OPTIONS,
  dispatcher fallbacks, and 404 paths the same CORS/header behavior.
- Legacy JSON endpoint methods now route through `send_json_action()` and
  `Handler._json_action()`, centralizing success/error JSON behavior.
- Normal GET handling can now use `handle_dispatched_get()`, combining route
  dispatch and response application before falling back to legacy handler
  routing.
- Dispatched GET error handling now goes through `handle_dispatched_get_safely()`,
  so normal `Handler.do_GET()` routing has a tested API-layer success/error
  wrapper before the legacy compatibility path is removed.
- HTTP body/JSON/empty response writing now has tested helpers in
  `src/kq_tool/api/http_response.py`, letting `server.py Handler` delegate
  response header/body emission while retaining legacy fallbacks.
- GET route payload selection now lives in `src/kq_tool/api/routes/get.py`,
  leaving `dispatcher.py` to adapt route responses into the legacy local
  server response callbacks.
- Legacy GET fallback routing now goes through `handle_legacy_get()` in
  `src/kq_tool/api/dispatcher.py`, reducing `Handler.do_GET()` to a helper
  call when the compatibility path is needed.
- The threaded/reusable local HTTP server base class now lives in
  `src/kq_tool/api/runtime.py`, with `server.py` inheriting it when available.
- `src/kq_tool/api/__init__.py` now exports the current dispatcher,
  response, services, and runtime helper surface with package-level contract
  coverage.

## Package Skeleton and Pure Functions

- [x] Create `src/kq_tool` package folders.
- [x] Add `pyproject.toml` with package, pytest, black, isort, and ruff config.
- [x] Extract stable constants into `src/kq_tool/config.py`.
- [x] Extract ticker normalization into `src/kq_tool/utils/tickers.py`.
- [x] Extract external-service retry/backoff helper into `src/kq_tool/utils/retry.py`.
- [x] Extract pure indicators into `src/kq_tool/analyzer/indicators.py`.
- [x] Add server reuse regression coverage for the extracted indicator helpers.
- [x] Extract Alpha Decay utility into `src/kq_tool/analyzer/alpha_decay.py`.
- [x] Add adaptive signal-quality helpers in
      `src/kq_tool/analyzer/signal_quality.py`.
      - Point-in-time RSI percentile thresholds.
      - Trend-aware RSI/BB signal quality scores.
      - MACD zero-line/slope quality scores.
      - Volume confirmation and OBV helpers.
- [x] Add optional signal-quality filtering/weighting to Alpha Decay so
      validation scripts can compare raw events versus high-quality events.
- [x] Add `tests/validation_signal_quality_alpha_decay.py` to compare raw
      Alpha Decay events against quality-filtered events on IS/OOS splits.
- [x] Add threshold sweep support to the signal-quality Alpha Decay validation
      script (`--sweep --qualities 0.4,0.5,0.6,0.7,0.8`) with cached
      per-ticker signal-quality scores.
- [x] Run the first broader signal-quality sweep on the top 50 market-cap
      universe with 200 placebo draws. Quality `>= 0.70` was the strongest
      candidate in this pass: 2,944 OOS events, +2.376% edge versus +1.714%
      raw, placebo +0.846%, p=0.0.
- [x] Recheck signal-quality sweep on the top 100 market-cap universe with
      500 placebo draws. Quality `>= 0.60` and `>= 0.70` were both better
      than raw: `>= 0.60` had 9,344 OOS events, +1.868% edge, placebo
      +0.445%, p=0.0; `>= 0.70` had 6,595 OOS events, +1.867% edge, placebo
      +0.402%, p=0.0. Raw was 25,500 events, +1.669% edge, placebo +0.855%,
      p=0.0.
- [x] Save signal-quality validation summaries as a human-readable CSV next
      to the compressed `.npz` result file.
- [x] Add unit tests for tickers, indicators, and Alpha Decay metadata.
- [x] Replace selected legacy `server.py` helper calls with imports from
      `src/kq_tool` after the unit tests are green.
      - Connected ticker normalization, technical indicators, and Alpha Decay.
      - Kept legacy fallback blocks inside `server.py` for safe rollback.

## Data Layer

- [x] Move cache helper into `src/kq_tool/data/cache.py`.
- [x] Move price normalization and period filtering into `src/kq_tool/data/price.py`.
- [x] Move yfinance live-price extraction priority into `src/kq_tool/data/price.py`
      and verify legacy `server.py` reuses it.
- [x] Move Yahoo Finance warmup/history-validity check into
      `src/kq_tool/data/price.py` and verify legacy `server.py` reuses it.
- [x] Move period-specific minimum price-row checks into
      `src/kq_tool/data/price.py` and verify yfinance fallback uses it.
- [x] Move yfinance download-frame preparation into
      `src/kq_tool/data/price.py` so normalization and row-count validation
      are handled together.
- [x] Move live current-price/source/date context resolution into
      `src/kq_tool/data/price.py` and reuse it from stock analysis paths.
- [x] Move fundamental calculation helpers into `src/kq_tool/data/fundamental.py`.
- [x] Move yfinance fundamental fallback usability check into
      `src/kq_tool/data/fundamental.py` and verify legacy `server.py`
      reuses it.
- [x] Move Point-in-Time market-cap logic into `src/kq_tool/data/marketcap.py`.
- [x] Add unit tests for look-ahead safe market-cap selection.
- [x] Connect legacy `server.py` to the new cache and market-cap helpers.
      - `server.py` still keeps fallback code for safe rollback.
      - Verified `get_top_marketcap_tickers` uses the new module.
- [x] Connect legacy `server.py` to price period filtering and fundamental helpers.
      - Verified Samsung Electronics price/fundamental lookup still uses real Excel data.
- [x] Move universe construction into `src/kq_tool/data/universe.py`.
- [x] Add repository interfaces in `src/kq_tool/data/repositories.py`.
- [x] Connect legacy `server.py` universe construction to the new data module.

## Analyzer and Portfolio Layer

- [x] Move robo signal scoring into `src/kq_tool/analyzer/robo.py`.
- [x] Move Reverse DCF helper into `src/kq_tool/analyzer/dcf.py`.
- [x] Connect legacy `server.py` to new robo and DCF helpers.
- [x] Move stock-analysis orchestration into `src/kq_tool/analyzer/stock_analyzer.py`.
- [x] Move risk-based allocation into `src/kq_tool/portfolio/risk_based.py`.
- [x] Move portfolio weight helpers into `src/kq_tool/portfolio/weights.py`.
- [x] Connect legacy `server.py` to risk-based and weight helper modules.
- [x] Move recommendation rule helpers into `src/kq_tool/portfolio/recommender.py`.
      - Portfolio validity, auto regime tilt, signal multipliers, signal tilt.
- [x] Move portfolio validity/review-window helper into
      `src/kq_tool/portfolio/validity.py`.
- [x] Connect legacy `server.py` to recommendation rule helpers.
- [x] Move full recommendation report orchestration into `src/kq_tool/portfolio/recommender.py`.
- [x] Add tests for weights summing to 1.0, fallback behavior, and Alpha Decay
      review windows.

## Backtest, Screener, Regime, and API

- [x] Move backtest metrics into `src/kq_tool/backtest/metrics.py`.
- [x] Connect legacy `server.py` to the new metrics helper.
- [x] Move backtest selector into `src/kq_tool/backtest/selector.py`.
- [x] Add selector tests for PiT-filtered label alignment.
- [x] Connect legacy `server.py` to the new selector helper.
- [x] Move reusable backtest engine helpers into `src/kq_tool/backtest/engine.py`.
- [x] Connect legacy `server.py` to period return, benchmark normalization, and underwater helpers.
- [x] Move full backtest engine orchestration into `src/kq_tool/backtest`.
- [x] Move strategy backtest period policy and price-frame preparation into
      `src/kq_tool/backtest/preparation.py` and verify legacy `server.py`
      reuses it.
- [x] Move robo-strategy indicator precomputation into
      `src/kq_tool/backtest/selector.py` and verify legacy `server.py`
      reuses it.
- [x] Move screener rules into `src/kq_tool/screener`.
- [x] Connect legacy `server.py` to screener helper modules.
- [x] Move screener price-date metadata calculation into
      `src/kq_tool/screener/engine.py` and verify legacy `server.py`
      reuses it.
- [x] Move screener cache prewarm policy into `src/kq_tool/screener/engine.py`
      to avoid duplicate parquet loading during parallel runs.
- [x] Move regime snapshot builder into `src/kq_tool/regime/classifier.py`.
- [x] Connect legacy `server.py` to the new regime snapshot helper.
- [x] Align Alpha Decay operating active window to 10 trading days in both
      legacy `server.py` and `src/kq_tool/analyzer/alpha_decay.py`.
- [x] Add recommendation helper tests for Robo confidence scale handling.
- [x] Fix chart controls so candle frame (daily/monthly/yearly) is separate
      from lookback period and monthly/yearly OHLCV is aggregated correctly.
- [x] Remove the server-side 252-row chart cap, add full-period chart loading,
      browser-side zoom/pan/range navigation, and PNG/CSV/result export buttons.
- [x] Move stock chart payload construction into `src/kq_tool/analyzer/chart.py`
      and add regression tests that preserve full-period chart history.
- [x] Centralize UI/API period lookback mapping in `src/kq_tool/data/price.py`
      and reuse it from legacy `server.py` chart/backtest paths.
- [x] Move ETF dynamic strategy generation and allocation result evaluation into
      `src/kq_tool/portfolio/allocation.py`, then connect legacy `server.py`
      to the shared helpers.
- [x] Move recommendation asset-signal summary conversion into
      `src/kq_tool/portfolio/recommender.py` and verify legacy `server.py`
      reuses it.
- [x] Move API JSON-safe serialization into `src/kq_tool/api/serialization.py`
      and verify legacy `server.py` reuses it.
- [x] Move HTTP response headers and JSON body encoding into
      `src/kq_tool/api/http_response.py` and verify legacy `server.py`
      reuses the shared helpers.
- [x] Move stock and strategy-backtest query parsing into
      `src/kq_tool/api/params.py`, including transaction-cost/slippage
      parameters for the legacy fallback path.
- [x] Move health endpoint status-section assembly into
      `src/kq_tool/api/health.py` and keep the legacy server response shape.
- [x] Move `/api/regime_ai` response payload construction into
      `src/kq_tool/regime/response.py` and verify legacy `server.py` reuses it.
- [x] Move macro regime payload construction into
      `src/kq_tool/regime/macro_builder.py` and verify legacy `server.py`
      reuses it.
- [x] Split HTTP endpoints into `src/kq_tool/api`.
- [x] Move startup browser policy into `src/kq_tool/api/runtime.py` and
      verify Chrome preference/fallback behavior with unit tests.
- [x] Move API service registry validation into `src/kq_tool/api/services.py`
      so dispatcher dependencies are explicit and tested.
- [x] Move static file reading/path validation into
      `src/kq_tool/api/static_files.py` while preserving the legacy index route.
- [x] Keep the old UI compatible during migration.

## Validation Reporting and Experiment Hygiene

- [x] Add reusable validation result loaders for `.npz` and CSV summaries.
- [x] Add comparison helpers for raw vs filtered model experiments.
- [x] Add tests for signal-quality summary ranking and guardrails.
- [x] Generate stable CSV/Markdown summaries for important validation runs.

## Signal Quality Segmentation

- [x] Split signal-quality Alpha Decay validation by large-cap vs mid/small-cap.
      - Added `--ticker-segment all|large|mid_small` and `--segment-split`.
- [x] Split validation by buy-side vs sell-side signals.
      - Added `--signal-side all|buy|sell`; smoke results show buy/large
        improved while sell/mid_small deteriorated, so side-specific adoption is required.
- [x] Split validation by ETF vs individual stocks where data allows.
      - Added `--universe-source equity|etf`; ETF path uses
        `data/cache/etf_assets_v2.parquet`.
- [x] Compare limit-move excluded vs included samples.
      - Added `--include-limit-moves`; default keeps Korean-market limit-move
        exclusion enabled.
- [x] Document Week 6 smoke checks in `docs/signal_quality_segmentation.md`.

## Cost, Slippage, and Trading Friction

- [x] Add reusable transaction-cost and slippage helpers.
      - Added `src/kq_tool/backtest/costs.py` with equal weights, turnover,
        bps conversion, transaction cost rate, and net-return adjustment.
- [x] Add cost sensitivity to strategy backtests.
      - `run_rebalanced_strategy_backtest()` now accepts transaction cost and
        slippage bps, reports turnover/cost summary, and `/api/stratbt` accepts
        `tc` and `slip` query parameters.
      - Strategy verification UI exposes 거래비용/슬리피지 bps inputs.
- [x] Add cost sensitivity to signal-quality validation summaries.
      - Added `src/kq_tool/validation/costs.py`.
      - `tests/validation_signal_quality_alpha_decay.py` now accepts
        `--cost-bps`, `--slippage-bps`, and `--trade-sides`; CSV/NPZ outputs
        preserve the cost assumptions.
- [x] Document operating thresholds after costs.
      - Added `docs/operating-thresholds-after-costs.md` with adoption gates,
        current evidence, prohibited claims, and next full costed runs.

## Production Readiness

- [x] Add Windows team launcher and run guide.
      - Added `RUN_KQ_TOOL.bat`, `RUN_WINDOWS.ps1`, and
        `docs/TEAM_RUN_GUIDE.md` so non-developer teammates can start the app
        without memorizing Python commands.
- [x] Add lightweight API smoke tests for core endpoints.
      - Added `tests/smoke_api.py`; default checks `/api/ping` and
        `/api/health`, with optional `--include-stock` for heavier stock API checks.
- [x] Add a startup health payload for data/module readiness.
      - Added `/api/health` and `src/kq_tool/api/health.py` with file,
        module, data, regime, universe, and ETF readiness.
- [x] Add a final regression command list for local release checks.
      - Added `docs/final-regression-checks.md`,
        `RUN_REGRESSION_CHECKS.bat`, and `RUN_REGRESSION_CHECKS.ps1`.
      - The regression runner compiles `server.py`, core validation scripts,
        and `src/kq_tool`, then runs the unit test suite.
- [x] Add release-readiness summary for team handoff and demo preparation.
      - Added `docs/release-readiness.md` with current status, commands,
        pass criteria, explanation points, and optional remaining work.
- [x] Update README with the current architecture and validation status.
      - README now documents the expanded `src/kq_tool` module layout,
        regression runner, API smoke check, health endpoint, chart/export
        support, cost sensitivity, and Alpha Decay segmentation status.
- [x] Clean non-runtime temporary files from the project root.
      - Removed one-off diagnosis scripts, stale `.bak_*` files, empty server
        logs, `.coverage`, `.pytest_cache`, and Python `__pycache__` folders.
      - Kept validation `.npz/.csv` outputs and data caches because they are
        research evidence and runtime inputs.
- [x] Align runtime dependencies with the regime model fallback path.
      - Added `lightgbm` and conditional `hmmlearn` to `requirements.txt` and
        `pyproject.toml`; TabPFN remains optional and disabled by default.
      - Python 3.14 skips `hmmlearn` because Windows wheels are not available
        yet; the app keeps running with the built-in transition fallback.

## Asset Allocation Strategy Coverage

- [x] Expand ETF allocation strategies from 9 to 10.
      - Added `정적 60/40` as a simple stock/bond baseline:
        KODEX 200 60% + KOSEF 국고채10년 40%.
      - Connected the strategy in `server.py`, `src/kq_tool/config.py`,
        `index.html`, and `tests/validation_asset_allocation_v2.py`.
      - Added a unit test that locks the strategy count at 10 and verifies
        the 60/40 weights sum to 1.0.
- [x] Make ETF/strategy configuration reusable from `src/kq_tool/config.py`.
      - `server.py` now imports ETF metadata, allocation strategies, risk-based
        keys, port, and horizons from the package config with legacy fallback.
      - Added a regression test that verifies `server.py` and
        `src/kq_tool/config.py` stay synchronized.
- [x] Extend server configuration reuse beyond strategy definitions.
      - `server.py` now reuses risk-free rate, equity risk premium, Alpha
        Decay signal direction, recommendation meta components, and regime
        target weights from the package modules with legacy fallback.
      - Added regression tests that keep `server.py`,
        `src/kq_tool/config.py`, and `src/kq_tool/portfolio/recommender.py`
        synchronized.
- [x] Remove duplicate Alpha Decay signal direction mapping from robo scoring.
      - `src/kq_tool/analyzer/robo.py` now reuses
        `src/kq_tool/config.py::SIGNAL_DIRECTION`, matching the Alpha Decay
        module and `server.py`.
      - Added a robo regression test so future signal names/directions stay
        centralized.
- [x] Remove duplicate signal-quality column definitions.
      - `src/kq_tool/analyzer/signal_quality.py` now derives
        `SIGNAL_COLUMNS` from `src/kq_tool/config.py::SIGNAL_DIRECTION`.
      - Added a signal-quality regression test so Alpha Decay, robo scoring,
        and signal-quality filters use the same signal family.
- [x] Move robo score weights and labels into shared config.
      - Added `ROBO_SIGNAL_WEIGHTS` and `ROBO_SIGNAL_LABELS` to
        `src/kq_tool/config.py`.
      - `src/kq_tool/analyzer/robo.py` now derives its legacy score weights
        and confidence-weighted detail labels from the shared config.
      - Added a regression test that preserves the original 0/50/100 score
        scale while keeping weights centralized.
- [x] Reuse shared robo score weights in `server.py` fallback paths.
      - Legacy `server.py` robo scoring and confidence-weighted fallback now
        read `ROBO_SIGNAL_WEIGHTS` and `ROBO_SIGNAL_LABELS` from config when
        available.
      - Expanded server/config synchronization tests to cover the robo score
        policy as well as strategy and signal direction settings.
- [x] Reuse shared screener universe limit in `server.py`.
      - `server.py` now derives `SCREENER_LIMIT` from
        `src/kq_tool/config.py` with a legacy fallback.
      - Existing synchronization tests now cover the screener limit alongside
        market assumptions and signal settings.
- [x] Centralize robo buy/sell score thresholds.
      - Added `ROBO_BUY_THRESHOLD` and `ROBO_SELL_THRESHOLD` to
        `src/kq_tool/config.py`.
      - `analyzer/robo.py`, `screener/engine.py`, and `server.py` fallback
        paths now use the same threshold policy.
      - Added regression coverage for threshold-driven robo and screener
        signal classification.

## Validation Rule

Do not treat a model improvement as real until it passes:

- Point-in-Time data handling
- IS/OOS split
- Placebo or permutation check
- Baseline comparison against simple portfolios
- Cost/slippage sensitivity when trading frequency matters

## Alpha Decay Next Step

The current evidence supports Alpha Decay mainly as a buy-side validity and
review-window tool. The next validation pass should separate:

- large-cap vs small-cap liquidity groups
- ETF vs individual stock signals
- buy-side vs sell-side signals
- limit-move excluded vs included samples




