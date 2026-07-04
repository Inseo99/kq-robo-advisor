

#!/usr/bin/env python3
"""
KQ Quant Tool  —  python server.py  →  http://127.0.0.1:8888
종목 검색·차트·로보신호·스크리너·전략백테스트(KOSPI비교)·ETF배분·Alpha Decay·매크로
"""
# ── 의존성 자동 설치 ────────────────────────────────────────────────────────
# curl_cffi: 최신 yfinance가 Yahoo Finance의 크럼(crumb) 인증을 통과하기 위해
#            내부적으로 요구하는 패키지. 누락/구버전이면 "Invalid Crumb" 401 오류가 난다.
import subprocess, sys
for _p in ['yfinance','curl_cffi','scipy','pandas','numpy']:
    try: __import__(_p)
    except ImportError:
        print(f'설치 중: {_p}...')
        subprocess.check_call([sys.executable,'-m','pip','install',_p,'-q','--upgrade'],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

try:
    import yfinance as _yfv
    _maj,_min = (int(x) for x in _yfv.__version__.split('.')[:2])
    if (_maj,_min) < (0,2):
        raise ImportError
except Exception:
    print('yfinance 버전이 오래되어 업그레이드합니다 (Invalid Crumb 오류 방지)...')
    subprocess.check_call([sys.executable,'-m','pip','install','-q','--upgrade','yfinance','curl_cffi'],
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

import http.server, os, threading, time, json
import traceback, warnings, logging, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
from scipy.optimize import curve_fit, minimize
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)  # 내부 재시도 과정의 노이즈 로그 숨김

PORT     = 8888
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, 'src')
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
RF       = 0.035
ERP      = 0.060
HORIZONS = [1,3,5,7,10,15,20]

try:
    from kq_tool.config import (
        EQUITY_RISK_PREMIUM as _KQ_CONFIG_EQUITY_RISK_PREMIUM,
        ETFS as _KQ_CONFIG_ETFS,
        HORIZONS as _KQ_CONFIG_HORIZONS,
        PORT as _KQ_CONFIG_PORT,
        RISK_FREE_RATE as _KQ_CONFIG_RISK_FREE_RATE,
        RISK_BASED_KEYS as _KQ_CONFIG_RISK_BASED_KEYS,
        ROBO_BUY_THRESHOLD as _KQ_CONFIG_ROBO_BUY_THRESHOLD,
        ROBO_SELL_THRESHOLD as _KQ_CONFIG_ROBO_SELL_THRESHOLD,
        ROBO_SIGNAL_LABELS as _KQ_CONFIG_ROBO_SIGNAL_LABELS,
        ROBO_SIGNAL_WEIGHTS as _KQ_CONFIG_ROBO_SIGNAL_WEIGHTS,
        SCREENER_LIMIT as _KQ_CONFIG_SCREENER_LIMIT,
        SIGNAL_DIRECTION as _KQ_CONFIG_SIGNAL_DIRECTION,
        STRATEGIES as _KQ_CONFIG_STRATEGIES,
    )
    PORT = _KQ_CONFIG_PORT
    HORIZONS = list(_KQ_CONFIG_HORIZONS)
    RF = _KQ_CONFIG_RISK_FREE_RATE
    ERP = _KQ_CONFIG_EQUITY_RISK_PREMIUM
except Exception as _config_mod_e:
    print(f'  [module] src/kq_tool config import 실패 - legacy 설정 사용: {_config_mod_e}')
    _KQ_CONFIG_ETFS = None
    _KQ_CONFIG_STRATEGIES = None
    _KQ_CONFIG_RISK_BASED_KEYS = None
    _KQ_CONFIG_ROBO_BUY_THRESHOLD = None
    _KQ_CONFIG_ROBO_SELL_THRESHOLD = None
    _KQ_CONFIG_ROBO_SIGNAL_LABELS = None
    _KQ_CONFIG_ROBO_SIGNAL_WEIGHTS = None
    _KQ_CONFIG_SCREENER_LIMIT = None
    _KQ_CONFIG_SIGNAL_DIRECTION = None

try:
    from kq_tool.api.actions import run_json_service_action as _kq_run_json_service_action
    from kq_tool.api.actions import run_stock_action as _kq_run_stock_action
    from kq_tool.api.actions import run_strategy_backtest_action as _kq_run_strategy_backtest_action
    from kq_tool.api.dispatcher import handle_dispatched_get_safely as _kq_handle_dispatched_get_safely
    from kq_tool.api.dispatcher import handle_legacy_get as _kq_handle_legacy_get
    from kq_tool.api.health import build_health_payload as _kq_build_health_payload
    from kq_tool.api.health import build_server_health_payload as _kq_build_server_health_payload
    from kq_tool.api.http_response import (
        cors_headers as _kq_cors_headers,
        error_payload as _kq_error_payload,
        make_response_writer as _kq_make_response_writer,
        no_cache_headers as _kq_no_cache_headers,
        send_empty_response as _kq_send_empty_response,
        send_json_action as _kq_send_json_action,
        send_json_response as _kq_send_json_response,
    )
    from kq_tool.api.runtime import (
        ThreadingReusableHTTPServer as _KQThreadingReusableHTTPServer,
        browser_autostart_disabled_message as _kq_browser_autostart_disabled_message,
        create_http_server as _kq_create_http_server,
        open_preferred_browser as _kq_open_preferred_browser,
        run_server_with_browser_policy as _kq_run_server_with_browser_policy,
        schedule_browser_open as _kq_schedule_browser_open,
        serve_until_interrupted as _kq_serve_until_interrupted,
        server_ready_messages as _kq_server_ready_messages,
        server_url as _kq_server_url,
        should_auto_open_browser as _kq_should_auto_open_browser,
        startup_intro_messages as _kq_startup_intro_messages,
        yfinance_status_messages as _kq_yfinance_status_messages,
    )
    from kq_tool.api.services import build_server_api_services as _kq_build_server_api_services
    from kq_tool.api.serialization import clean_json_value as _kq_clean_json_value
    from kq_tool.api.static_files import serve_static_file as _kq_serve_static_file
except Exception as _api_mod_e:
    print(f'  [module] src/kq_tool api helper import 실패 - legacy 라우팅 사용: {_api_mod_e}')
    _kq_run_json_service_action = None
    _kq_run_stock_action = None
    _kq_run_strategy_backtest_action = None
    _kq_handle_dispatched_get_safely = None
    _kq_handle_legacy_get = None
    _kq_build_health_payload = None
    _kq_build_server_health_payload = None
    _kq_cors_headers = None
    _kq_error_payload = None
    _kq_make_response_writer = None
    _kq_no_cache_headers = None
    _kq_send_empty_response = None
    _kq_send_json_action = None
    _kq_send_json_response = None
    _KQThreadingReusableHTTPServer = None
    _kq_browser_autostart_disabled_message = None
    _kq_create_http_server = None
    _kq_open_preferred_browser = None
    _kq_run_server_with_browser_policy = None
    _kq_schedule_browser_open = None
    _kq_serve_until_interrupted = None
    _kq_server_ready_messages = None
    _kq_server_url = None
    _kq_should_auto_open_browser = None
    _kq_startup_intro_messages = None
    _kq_yfinance_status_messages = None
    _kq_build_server_api_services = None
    _kq_clean_json_value = None
    _kq_serve_static_file = None

try:
    from kq_tool.data.cache import cached as _kq_cached
    from kq_tool.data.marketcap import (
        build_mcap_history as _kq_build_mcap_history,
        build_top_marketcap_tickers as _kq_build_top_marketcap_tickers,
        get_top_mcap_at as _kq_get_top_mcap_at,
    )
    from kq_tool.data.price import (
        PERIOD_DAYS as _KQ_PERIOD_DAYS,
        days_for_period as _kq_days_for_period,
        extract_live_price as _kq_extract_live_price,
        filter_price_period as _kq_filter_price_period,
        has_min_rows_for_period as _kq_has_min_rows_for_period,
        min_rows_for_period as _kq_min_rows_for_period,
        normalize_yfinance_columns as _kq_normalize_yfinance_columns,
        prepare_yfinance_price_frame as _kq_prepare_yfinance_price_frame,
        resolve_current_price_context as _kq_resolve_current_price_context,
        sample_price as _kq_sample_price,
        warm_yfinance_session as _kq_warm_yfinance_session,
    )
    from kq_tool.data.universe import (
        build_universe as _kq_build_universe,
        market_counts as _kq_market_counts,
    )
    from kq_tool.data.fundamental import (
        excel_fundamental_info as _kq_excel_fundamental_info,
        has_yfinance_fundamental_info as _kq_has_yfinance_fundamental_info,
        sample_fundamental_info as _kq_sample_fundamental_info,
    )
    _KQ_DATA_HELPERS_READY = True
except Exception as _data_mod_e:
    print(f'  [module] src/kq_tool data helper import 실패 - legacy 데이터 함수 사용: {_data_mod_e}')
    _KQ_DATA_HELPERS_READY = False
    _kq_cached = None
    _kq_build_mcap_history = None
    _kq_build_top_marketcap_tickers = None
    _kq_get_top_mcap_at = None
    _KQ_PERIOD_DAYS = None
    _kq_days_for_period = None
    _kq_extract_live_price = None
    _kq_filter_price_period = None
    _kq_has_min_rows_for_period = None
    _kq_min_rows_for_period = None
    _kq_normalize_yfinance_columns = None
    _kq_prepare_yfinance_price_frame = None
    _kq_resolve_current_price_context = None
    _kq_sample_price = None
    _kq_warm_yfinance_session = None
    _kq_build_universe = None
    _kq_market_counts = None
    _kq_excel_fundamental_info = None
    _kq_has_yfinance_fundamental_info = None
    _kq_sample_fundamental_info = None

_LEGACY_PERIOD_DAYS = {
    '1d': 1, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365,
    '2y': 730, '3y': 1095, '5y': 1825, '7y': 2555,
    '10y': 3650, '12y': 4380, '15y': 5475, '20y': 7300,
}

def _period_days(period, default_days=365):
    """공유 price helper가 없을 때도 기존 기간 필터 동작을 유지한다."""
    if _kq_days_for_period:
        return _kq_days_for_period(period, default_days)
    return _LEGACY_PERIOD_DAYS.get(period or '', default_days)

try:
    from kq_tool.utils.retry import retry_call as _kq_retry_call
    from kq_tool.utils.tickers import ticker_to_code as _kq_ticker_to_code
    from kq_tool.analyzer.indicators import (
        atr as _kq_atr,
        bollinger_bands as _kq_bb,
        close_series as _kq_close_series,
        macd as _kq_macd,
        rsi as _kq_rsi,
    )
    from kq_tool.analyzer.chart import build_stock_chart as _kq_build_stock_chart
    from kq_tool.analyzer.alpha_decay import (
        alpha_single as _kq_alpha_single,
        half_life_to_confidence as _kq_half_life_to_confidence,
    )
    from kq_tool.analyzer.dcf import reverse_dcf_growth as _kq_reverse_dcf_growth
    from kq_tool.analyzer.robo import (
        confidence_weighted_robo as _kq_confidence_weighted_robo,
        legacy_robo_score as _kq_legacy_robo_score,
        score_to_signal as _kq_score_to_signal,
    )
    from kq_tool.analyzer.stock_analyzer import analyze_stock_payload as _kq_analyze_stock_payload
    _KQ_MODULAR_HELPERS_READY = True
except Exception as _mod_e:
    print(f'  [module] src/kq_tool helper import 실패 - legacy 함수 사용: {_mod_e}')
    _KQ_MODULAR_HELPERS_READY = False
    _kq_retry_call = None
    _kq_ticker_to_code = None
    _kq_build_stock_chart = None
    _kq_close_series = None
    _kq_rsi = None
    _kq_macd = None
    _kq_bb = None
    _kq_atr = None
    _kq_alpha_single = None
    _kq_half_life_to_confidence = None
    _kq_reverse_dcf_growth = None
    _kq_confidence_weighted_robo = None
    _kq_legacy_robo_score = None
    _kq_score_to_signal = None
    _kq_analyze_stock_payload = None

try:
    from kq_tool.portfolio.allocation import (
        evaluate_asset_allocation_strategies as _kq_evaluate_asset_allocation_strategies,
        gtaa_weights as _kq_gtaa_weights,
        inverse_volatility_weights as _kq_inverse_volatility_weights,
        risk_based_strategy_weights as _kq_risk_based_strategy_weights,
    )
    from kq_tool.portfolio.risk_based import (
        diversification_ratio as _kq_diversification_ratio,
        erc_weights as _kq_erc_weights,
        gmv_weights as _kq_gmv_weights,
        mdp_weights as _kq_mdp_weights,
        risk_contributions as _kq_risk_contributions,
    )
    from kq_tool.portfolio.weights import (
        combine_weight_sets as _kq_combine_weight_sets,
        meta_base_weights as _kq_meta_base_weights,
        normalize_weights as _kq_normalize_weights,
    )
    from kq_tool.portfolio.recommender import (
        META_COMPONENTS as _KQ_META_COMPONENTS,
        REGIME_TARGETS as _KQ_REGIME_TARGETS,
        apply_signal_tilt as _kq_apply_signal_tilt,
        auto_regime_tilt as _kq_auto_regime_tilt,
        build_recommendation_report as _kq_build_recommendation_report,
        default_asset_signal as _kq_default_asset_signal,
        load_regime_alpha_summary as _kq_load_regime_alpha_summary,
        portfolio_validity as _kq_portfolio_validity,
        probability_weighted_regime_target as _kq_probability_weighted_regime_target,
        regime_probability_blend as _kq_regime_probability_blend,
        regime_alpha_signal_adjustment as _kq_regime_alpha_signal_adjustment,
        signal_weight_multiplier as _kq_signal_weight_multiplier,
        stock_analysis_to_asset_signal as _kq_stock_analysis_to_asset_signal,
    )
    _KQ_PORTFOLIO_HELPERS_READY = True
except Exception as _portfolio_mod_e:
    print(f'  [module] src/kq_tool portfolio helper import 실패 - legacy 포트폴리오 함수 사용: {_portfolio_mod_e}')
    _KQ_PORTFOLIO_HELPERS_READY = False
    _kq_evaluate_asset_allocation_strategies = None
    _kq_gtaa_weights = None
    _kq_inverse_volatility_weights = None
    _kq_risk_based_strategy_weights = None
    _kq_diversification_ratio = None
    _kq_erc_weights = None
    _kq_gmv_weights = None
    _kq_mdp_weights = None
    _kq_risk_contributions = None
    _kq_combine_weight_sets = None
    _kq_meta_base_weights = None
    _kq_normalize_weights = None
    _KQ_META_COMPONENTS = None
    _KQ_REGIME_TARGETS = None
    _kq_apply_signal_tilt = None
    _kq_auto_regime_tilt = None
    _kq_build_recommendation_report = None
    _kq_default_asset_signal = None
    _kq_load_regime_alpha_summary = None
    _kq_portfolio_validity = None
    _kq_probability_weighted_regime_target = None
    _kq_regime_probability_blend = None
    _kq_regime_alpha_signal_adjustment = None
    _kq_signal_weight_multiplier = None
    _kq_stock_analysis_to_asset_signal = None

try:
    from kq_tool.backtest.comparison import (
        build_quant_comparison_response as _kq_build_quant_comparison_response,
    )
    from kq_tool.backtest.engine import (
        equal_weight_period_return as _kq_equal_weight_period_return,
        normalize_benchmark_to_equity as _kq_normalize_benchmark_to_equity,
        underwater_curve as _kq_underwater_curve,
    )
    from kq_tool.backtest.metrics import perf_metrics as _kq_perf_metrics
    from kq_tool.backtest.orchestrator import (
        run_rebalanced_strategy_backtest as _kq_run_rebalanced_strategy_backtest,
    )
    from kq_tool.backtest.preparation import (
        filter_strategy_benchmark_period as _kq_filter_strategy_benchmark_period,
        prepare_strategy_price_frame as _kq_prepare_strategy_price_frame,
        use_fixed_start_for_period as _kq_use_fixed_start_for_period,
    )
    from kq_tool.backtest.selector import (
        build_robo_precomputed_indicators as _kq_build_robo_precomputed_indicators,
        select_for_backtest as _kq_select_for_backtest,
    )
    from kq_tool.backtest.strategy_meta import (
        QUANT as _KQ_QUANT,
        QUANT_COMPARE as _KQ_QUANT_COMPARE,
        QUANT_S2 as _KQ_QUANT_S2,
        strategy_descriptor as _kq_strategy_descriptor,
    )
    _KQ_BACKTEST_HELPERS_READY = True
except Exception as _backtest_mod_e:
    print(f'  [module] src/kq_tool backtest helper import 실패 - legacy 백테스트 함수 사용: {_backtest_mod_e}')
    _KQ_BACKTEST_HELPERS_READY = False
    _kq_equal_weight_period_return = None
    _kq_normalize_benchmark_to_equity = None
    _kq_underwater_curve = None
    _kq_perf_metrics = None
    _kq_run_rebalanced_strategy_backtest = None
    _kq_filter_strategy_benchmark_period = None
    _kq_prepare_strategy_price_frame = None
    _kq_use_fixed_start_for_period = None
    _kq_build_robo_precomputed_indicators = None
    _kq_select_for_backtest = None
    _kq_build_quant_comparison_response = None
    _KQ_QUANT = None
    _KQ_QUANT_COMPARE = None
    _KQ_QUANT_S2 = None
    _kq_strategy_descriptor = None

try:
    from kq_tool.regime.classifier import current_regime_snapshot as _kq_current_regime_snapshot
    from kq_tool.regime.macro_builder import (
        REGIME_DEFINITION as _KQ_REGIME_DEFINITION,
        build_macro_payload as _kq_build_macro_payload,
    )
    from kq_tool.regime.response import build_regime_ai_payload as _kq_build_regime_ai_payload
    from kq_tool.regime.market_report import build_market_report_context as _kq_build_market_report_context
    from kq_tool.regime.market_report import save_user_market_report as _kq_save_user_market_report
    _KQ_REGIME_HELPERS_READY = True
except Exception as _regime_mod_e:
    print(f'  [module] src/kq_tool regime helper import 실패 - legacy 국면 함수 사용: {_regime_mod_e}')
    _KQ_REGIME_HELPERS_READY = False
    _kq_current_regime_snapshot = None
    _KQ_REGIME_DEFINITION = None
    _kq_build_macro_payload = None
    _kq_build_regime_ai_payload = None
    _kq_build_market_report_context = None
    _kq_save_user_market_report = None

try:
    from kq_tool.screener.engine import build_screener_record as _kq_build_screener_record
    from kq_tool.screener.engine import latest_price_date_from_groups as _kq_latest_price_date_from_groups
    from kq_tool.screener.engine import prewarm_screener_cache as _kq_prewarm_screener_cache
    from kq_tool.screener.strategies import build_screeners as _kq_build_screeners
    _KQ_SCREENER_HELPERS_READY = True
except Exception as _screener_mod_e:
    print(f'  [module] src/kq_tool screener helper import 실패 - legacy 스크리너 함수 사용: {_screener_mod_e}')
    _KQ_SCREENER_HELPERS_READY = False
    _kq_build_screener_record = None
    _kq_latest_price_date_from_groups = None
    _kq_prewarm_screener_cache = None
    _kq_build_screeners = None

# ── 엑셀 데이터 로더 (선택적) ────────────────────────────────────────────
# data/ 디렉토리에 엑셀 파일이 있으면 자동으로 활용:
#   - 종목 유니버스: KOSPI+KOSDAQ 전체 (~3000)
#   - 재무 데이터: FnGuide 2014~2026 분기 31개 항목
#   - 매크로: 환율·금리·GDP·수출입
EXCEL_DATA = None
EXCEL_FIN  = None
EXCEL_MACRO = None
try:
    import data_loader as _dl_mod
    print('  [data] 엑셀 데이터 로드 중...')
    EXCEL_DATA  = _dl_mod.load_stocks(markets=('KOSPI','KOSDAQ'))
    EXCEL_FIN   = _dl_mod.load_financials()
    EXCEL_MACRO = _dl_mod.load_macro()
    _fin_n = len(_dl_mod._FIN_TICKERS_CACHE) if _dl_mod._FIN_TICKERS_CACHE else 0
    print(f'  [data] OK 종목 {len(EXCEL_DATA)}개, 재무 {_fin_n}개, 매크로 {len(EXCEL_MACRO)}시트 로드 완료')
except Exception as _e:
    print(f'  [data] 엑셀 데이터 없음 (yfinance 모드): {_e}')

# ── AI 시장 국면 모델 (TabPFN + HMM) ─────────────────────────────────────
# 서버 시작 시 매크로 데이터로 자동 학습. 실패해도 서버는 계속 동작.
REGIME_MODEL = None
try:
    if os.environ.get('KQ_ENABLE_TABPFN', '0').strip() == '1':
        os.environ['KQ_DISABLE_TABPFN'] = '0'
    else:
        os.environ['KQ_DISABLE_TABPFN'] = '1'
    import regime_model as _rm
    if EXCEL_MACRO:
        print('  [regime] 매크로 국면 모델 학습 중...')
        REGIME_MODEL = _rm.HierarchicalRegimeModel()
        REGIME_MODEL.fit(EXCEL_MACRO)
        if REGIME_MODEL.trained:
            _classifier = REGIME_MODEL.classifier.model_type or 'Rule-based'
            _n_samples = len(REGIME_MODEL.train_data) if REGIME_MODEL.train_data is not None else 0
            print(f'  [regime] OK 학습 완료 - 분류기: {_classifier}, 표본: {_n_samples}분기')
        else:
            print('  [regime] WARN 학습 실패 (데이터 부족) - 폴백 모드')
            REGIME_MODEL = None
    else:
        print('  [regime] 매크로 데이터 없음 - 모델 학습 건너뜀')
except Exception as _e:
    print(f'  [regime] 모델 로드 실패 (서버는 계속 동작): {_e}')
    REGIME_MODEL = None

# ── 종목 유니버스 ─────────────────────────────────────────────────────────
def _build_universe():
    """엑셀 데이터가 있으면 KOSPI+KOSDAQ 전체, 없으면 하드코딩 20종"""
    if _KQ_DATA_HELPERS_READY:
        return _kq_build_universe(EXCEL_DATA)
    if EXCEL_DATA:
        out = {}
        for ticker, info in EXCEL_DATA.items():
            # 야후 형식 티커
            suffix = '.KS' if info['market']=='KOSPI' else '.KQ'
            yt = f"{ticker}{suffix}"
            sector = info.get('sector') or '기타'
            out[yt] = (info['name'], sector)
        return out
    # fallback (엑셀 없을 때)
    return {
        '005930.KS':('삼성전자',    'IT/반도체'),
        '000660.KS':('SK하이닉스',  'IT/반도체'),
        '373220.KS':('LG에너지솔루션','2차전지'),
        '207940.KS':('삼성바이오로직스','바이오'),
        '005380.KS':('현대차',      '자동차'),
        '000270.KS':('기아',        '자동차'),
        '051910.KS':('LG화학',      '화학/2차전지'),
        '006400.KS':('삼성SDI',     '2차전지'),
        '035420.KS':('NAVER',       'IT/플랫폼'),
        '035720.KS':('카카오',      'IT/플랫폼'),
        '068270.KS':('셀트리온',    '바이오'),
        '105560.KS':('KB금융',      '금융'),
        '055550.KS':('신한지주',    '금융'),
        '086790.KS':('하나금융지주','금융'),
        '066570.KS':('LG전자',      '전자'),
        '005490.KS':('POSCO홀딩스', '철강'),
        '015760.KS':('한국전력',    '유틸리티'),
        '030200.KS':('KT&G',        '통신'),
        '096770.KS':('SK이노베이션','에너지/화학'),
        '003550.KS':('LG',          '지주'),
    }

UNIVERSE = _build_universe()
if _KQ_DATA_HELPERS_READY:
    _market_counts = _kq_market_counts(UNIVERSE)
    _kospi_n = _market_counts['KOSPI']
    _kosdaq_n = _market_counts['KOSDAQ']
else:
    _kospi_n = sum(1 for k in UNIVERSE if k.endswith(".KS"))
    _kosdaq_n = sum(1 for k in UNIVERSE if k.endswith(".KQ"))
print(f'  [data] UNIVERSE 종목 수: {len(UNIVERSE)} (KOSPI={_kospi_n}, KOSDAQ={_kosdaq_n})')

# 스크리너용 샘플링: 시가총액 상위 N개만 사용 (전체 다 돌리면 너무 느림)
SCREENER_LIMIT = int(_KQ_CONFIG_SCREENER_LIMIT or 200)  # 스크리너에서 다룰 종목 수 상한

_TOP_MARKETCAP_CACHE = None  # 시총 상위 종목 캐시 (서버 시작 후 1회 계산)

def get_top_marketcap_tickers(limit=None):
    """시가총액 상위 N개 종목 (yfinance 티커) 반환
    스크리너와 백테스트가 동일한 유니버스를 쓰도록 보장.
    캐시되어 두 번째 호출부터 즉시 반환.
    """
    global _TOP_MARKETCAP_CACHE
    if limit is None:
        limit = SCREENER_LIMIT
    if _TOP_MARKETCAP_CACHE is not None and len(_TOP_MARKETCAP_CACHE) >= limit:
        return _TOP_MARKETCAP_CACHE[:limit]

    if EXCEL_FIN is None:
        # 재무 데이터 없으면 UNIVERSE 전체 (백테스트가 느려짐)
        _TOP_MARKETCAP_CACHE = list(UNIVERSE.keys())
        return _TOP_MARKETCAP_CACHE[:limit]

    if _KQ_DATA_HELPERS_READY:
        fin_cache = getattr(_dl_mod, '_FIN_TICKERS_CACHE', None) if '_dl_mod' in globals() else None
        _TOP_MARKETCAP_CACHE = _kq_build_top_marketcap_tickers(
            UNIVERSE, _ticker_to_code, EXCEL_FIN, fin_cache, limit=limit
        )
        return _TOP_MARKETCAP_CACHE[:limit]

    mcap_pairs = []
    mcap_key = '시가총액(티커-상장예정주식수 포함)(백만원)'
    for yt in UNIVERSE.keys():
        code = _ticker_to_code(yt)
        try:
            if code in _dl_mod._FIN_TICKERS_CACHE:
                sub = EXCEL_FIN.loc[code]
                if mcap_key in sub.index:
                    rows = sub.loc[mcap_key]
                    if isinstance(rows, pd.Series):
                        rows = rows.to_frame().T
                    vals = rows['value'].dropna()
                    if not vals.empty:
                        mcap_pairs.append((yt, float(vals.iloc[-1])))
        except Exception:
            pass
    mcap_pairs.sort(key=lambda x: x[1], reverse=True)
    _TOP_MARKETCAP_CACHE = [t for t, _ in mcap_pairs]
    return _TOP_MARKETCAP_CACHE[:limit]

def _ticker_to_code(yt):
    """야후 티커(005930.KS) → 6자리 코드(005930)"""
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_ticker_to_code(yt)
    return yt.split('.')[0]


def build_mcap_history(universe_dict, ticker_to_code_fn, fin_data, fin_tickers_cache=None):
    """모든 종목의 분기별 시총을 날짜 x 종목 DataFrame으로 구성한다."""
    if _KQ_DATA_HELPERS_READY:
        return _kq_build_mcap_history(
            universe_dict, ticker_to_code_fn, fin_data, fin_tickers_cache
        )
    if fin_data is None:
        return pd.DataFrame()

    mcap_key = '시가총액(티커-상장예정주식수 포함)(백만원)'
    mcap_series_dict = {}

    for yt in universe_dict:
        code = ticker_to_code_fn(yt)
        try:
            if fin_tickers_cache is not None and code not in fin_tickers_cache:
                continue
            sub = fin_data.loc[code]
            if mcap_key not in sub.index:
                continue

            rows = sub.loc[mcap_key]
            if isinstance(rows, pd.Series):
                rows = rows.to_frame().T

            df = rows[['date', 'value']].dropna()
            if df.empty:
                continue

            values = df.set_index(pd.to_datetime(df['date']))['value']
            values = pd.to_numeric(values, errors='coerce').dropna()
            if values.empty:
                continue

            mcap_series_dict[yt] = values
        except Exception:
            continue

    if not mcap_series_dict:
        return pd.DataFrame()

    mcap_df = pd.DataFrame(mcap_series_dict)
    mcap_df = mcap_df.apply(pd.to_numeric, errors='coerce')
    return mcap_df.sort_index().ffill()


_MCAP_HISTORY_CACHE = None

def get_mcap_history():
    """server.py 공용 Point-in-Time 시총 이력 캐시."""
    global _MCAP_HISTORY_CACHE
    if _MCAP_HISTORY_CACHE is not None:
        return _MCAP_HISTORY_CACHE

    fin_cache = getattr(_dl_mod, '_FIN_TICKERS_CACHE', None) if '_dl_mod' in globals() else None
    _MCAP_HISTORY_CACHE = build_mcap_history(
        UNIVERSE, _ticker_to_code, EXCEL_FIN, fin_cache
    )
    return _MCAP_HISTORY_CACHE


def get_top_mcap_at(mcap_history, date, n=200):
    """date 시점 이전에 확인 가능한 시총 기준 상위 n개 종목 반환."""
    if _KQ_DATA_HELPERS_READY:
        return _kq_get_top_mcap_at(mcap_history, date, n)
    if mcap_history is None or mcap_history.empty:
        return []

    available = mcap_history.index[mcap_history.index <= date]
    if len(available) == 0:
        return []

    valid = mcap_history.loc[available[-1]].dropna()
    if valid.empty:
        return []

    valid = pd.to_numeric(valid, errors='coerce').dropna()
    if valid.empty:
        return []

    return valid.nlargest(min(n, len(valid))).index.tolist()
# ── 한국 ETF ─────────────────────────────────────────────────────────────
_LEGACY_ETFS = {
    '069500.KS':('KODEX 200',      '주식',  '공격'),
    '229200.KS':('KODEX 코스닥150','성장주','공격'),
    '130680.KS':('TIGER 원유선물', '원자재','공격'),
    '132030.KS':('KODEX 골드선물', '금',    '헤지'),
    '114260.KS':('KODEX 국고채3년','중기채','수비'),
    '148070.KS':('KOSEF 국고채10년','장기채','수비'),
    '153130.KS':('KODEX 단기채권', '초단기','수비'),
}
ETFs = dict(_KQ_CONFIG_ETFS or _LEGACY_ETFS)

_LEGACY_STRATEGIES = {
    '영구포트폴리오': {'069500.KS':0.25,'132030.KS':0.25,'148070.KS':0.25,'153130.KS':0.25},
    '황금나비':   {'069500.KS':0.20,'229200.KS':0.20,'132030.KS':0.20,'148070.KS':0.20,'153130.KS':0.20},
    '올웨더':     {'069500.KS':0.30,'130680.KS':0.075,'132030.KS':0.075,'148070.KS':0.40,'114260.KS':0.15},
    '정적 60/40': {'069500.KS':0.60,'148070.KS':0.40},
    '동일비중':   {t:round(1/7,4) for t in ETFs},
    '역변동성':   None,
    'GMV':        None,   # Global Minimum Variance (위험기반)
    'MDP':        None,   # Most Diversified Portfolio (위험기반)
    'ERC':        None,   # Equal Risk Contribution / Risk Parity (위험기반)
    'GTAA':       None,
}
STRATEGIES = {
    name: (dict(weights) if isinstance(weights, dict) else weights)
    for name, weights in (_KQ_CONFIG_STRATEGIES or _LEGACY_STRATEGIES).items()
}

# 위험기반 배분 3종 (이 셋만 별도로 표시 - 위험기여도 차트용)
RISK_BASED_KEYS = list(_KQ_CONFIG_RISK_BASED_KEYS or ['GMV', 'MDP', 'ERC'])

# ── 캐시 ────────────────────────────────────────────────────────────────
_cache: dict = {}

def _cached(key, ttl, fn, *args):
    if _KQ_DATA_HELPERS_READY:
        return _kq_cached(_cache, key, ttl, fn, *args)
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    result = fn(*args)
    _cache[key] = (now, result)
    return result

# ── Yahoo Finance 연결(쿠키/크럼) 워밍업 + 재시도 ────────────────────────
# "Invalid Crumb" 401 오류의 주된 원인:
#   1) 여러 스레드가 동시에 처음 요청을 보내면 각자 쿠키/크럼을 받으려 경합하며 깨짐
#   2) yfinance/curl_cffi 버전이 낮아 Yahoo의 최신 인증 방식을 처리 못함
#   3) 짧은 시간에 과도한 동시 요청으로 Yahoo가 일시적으로 차단(레이트리밋)
# → 서버 시작 시 단 한 번 동기적으로 워밍업하여 쿠키/크럼을 미리 확보해두고,
#   이후 모든 다운로드는 실패 시 지수 백오프로 재시도한다.
_yf_lock = threading.Lock()
_yf_ready = False
_yf_ok    = False

def _ensure_yf_session():
    global _yf_ready, _yf_ok
    if _yf_ready:
        return _yf_ok
    with _yf_lock:
        if _yf_ready:
            return _yf_ok
        try:
            if _kq_warm_yfinance_session is not None:
                _yf_ok = _kq_warm_yfinance_session()
            else:
                import yfinance as yf
                test = yf.Ticker('005930.KS').history(period='5d')
                _yf_ok = test is not None and not test.empty
        except Exception:
            _yf_ok = False
        _yf_ready = True
        return _yf_ok

def _yf_retry(fn, tries=3, base_delay=1.0):
    """fn()을 호출하고, 결과가 비어있거나 예외가 나면(크럼/레이트리밋 오류 등)
    지수 백오프 + 지터로 재시도한다. 마지막에도 실패하면 None 반환."""
    if _KQ_MODULAR_HELPERS_READY and _kq_retry_call is not None:
        return _kq_retry_call(fn, tries=tries, base_delay=base_delay)
    for i in range(tries):
        try:
            r = fn()
            if r is not None and not (hasattr(r,'empty') and r.empty):
                return r
        except Exception:
            pass
        if i < tries-1:
            time.sleep(base_delay*(2**i) + random.uniform(0,0.4))
    return None

# ── 샘플 데이터 ──────────────────────────────────────────────────────────
def _sprice(ticker, n=750, base=50000):
    if _KQ_DATA_HELPERS_READY:
        return _kq_sample_price(ticker, n=n, base=base)
    np.random.seed(abs(hash(ticker))%(2**31))
    p = base * np.exp(np.cumsum(np.random.normal(0.0002, 0.017, n)))
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='B')
    return pd.DataFrame({'Open':p*.99,'High':p*1.02,'Low':p*.98,
                         'Close':p,'Volume':np.random.randint(50000,3000000,n)}, index=idx)

def _sfund(ticker):
    if _KQ_DATA_HELPERS_READY:
        return _kq_sample_fundamental_info(ticker)
    np.random.seed(abs(hash(ticker))%(2**31))
    price = float(np.random.uniform(10000,120000))
    eps   = float(np.random.uniform(500,9000))
    return dict(trailingPE=round(price/eps,1),
                priceToBook=round(float(np.random.uniform(.5,4.)),2),
                returnOnEquity=round(float(np.random.uniform(.05,.28)),3),
                marketCap=round(price*float(np.random.uniform(3e8,6e9)),-8),
                currentPrice=round(price,-2),
                trailingEps=round(eps,0))

# ── 데이터 다운로드 ──────────────────────────────────────────────────────
_warned_tickers = set()  # 동일 종목에 대해 샘플데이터 경고를 중복 출력하지 않기 위한 집합

def _dl(ticker, period='1y'):
    """주가 OHLCV 조회 — 엑셀 데이터 우선, ETF·KOSPI는 yfinance
    실제 종가(raw)를 사용하여 현재가·차트가 일치하도록 함.
    """
    # 1) 엑셀 주가 데이터 우선 (개별 종목)
    if EXCEL_DATA and ticker in UNIVERSE:
        code = _ticker_to_code(ticker)
        try:
            # adjusted=False: 실제 종가(분할/배당 미반영) 사용
            df = _dl_mod.get_ohlcv_df(code, adjusted=False)
            if df is not None and len(df) > 20:
                # period에 따라 필터링 (엑셀 데이터 마지막 날짜 기준)
                if _KQ_DATA_HELPERS_READY:
                    return _kq_filter_price_period(df, period), False
                if period and period != 'max':
                    days = _period_days(period, 365)
                    # 엑셀 데이터 마지막 시점 기준 cutoff (현재 시점 아님!)
                    last_date = df.index[-1]
                    cutoff = last_date - pd.Timedelta(days=days)
                    df_f = df[df.index >= cutoff]
                    # 필터링 후 데이터 부족하면 전체 데이터 반환 (잘림 방지)
                    min_rows = 1 if period == '1d' else 20
                    if len(df_f) >= min_rows:
                        return df_f, False
                # 필터링 안 했거나 부족하면 전체 반환
                return df, False
        except Exception as e:
            pass

    # 2) yfinance fallback — UNIVERSE에 없는 티커만 시도 (ETF, KOSPI 지수 등)
    #    개별 종목(KOSPI/KOSDAQ)은 엑셀이 우선이므로 여기 도달 시 fallback 없이 종료
    if ticker not in UNIVERSE:
        _ensure_yf_session()
        try:
            import yfinance as yf
            df = _yf_retry(lambda: yf.download(ticker, period=period, progress=False, auto_adjust=False), tries=1)
            if df is not None and not df.empty:
                if _kq_prepare_yfinance_price_frame is not None:
                    prepared = _kq_prepare_yfinance_price_frame(df, period)
                    if prepared is not None:
                        return prepared, False
                else:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if len(df) >= (1 if period == '1d' else 20):
                        return df, False
        except Exception:
            pass

    # 3) 샘플 데이터
    if ticker not in _warned_tickers:
        _warned_tickers.add(ticker)
        print(f'WARN {ticker}: 실시간 데이터 수신 실패 - 샘플 데이터로 대체합니다')
    base = 10000 if ticker in ETFs else 50000
    return _sprice(ticker, base=base), True

def _c(df):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_close_series(df)
    c = df['Close'] if isinstance(df, pd.DataFrame) else df
    if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
    return c.squeeze()

def _fund_info(ticker):
    """재무 정보 조회 — 엑셀 데이터 우선, 없으면 yfinance"""
    # 1) 엑셀 재무 데이터 우선
    if EXCEL_FIN is not None:
        code = _ticker_to_code(ticker)
        if code in _dl_mod._FIN_TICKERS_CACHE:
            try:
                latest = _dl_mod.get_latest_fin(EXCEL_FIN, code)
                if latest:
                    if _KQ_DATA_HELPERS_READY:
                        info = _kq_excel_fundamental_info(
                            latest,
                            lambda metric: _dl_mod.get_metric_value(code, metric),
                        )
                        return info, False
                    # 엑셀 메트릭 시트에서 PER·PBR·EPS·BPS 직접 조회 (있으면 우선)
                    pe_metric  = _dl_mod.get_metric_value(code, 'per')
                    pbr_metric = _dl_mod.get_metric_value(code, 'pbr')
                    eps_metric = _dl_mod.get_metric_value(code, 'eps')
                    bps_metric = _dl_mod.get_metric_value(code, 'bps')
                    div_yield  = _dl_mod.get_metric_value(code, 'div_yield')

                    # 재무 데이터에서 백업 계산
                    mcap_mil = latest.get('시가총액(티커-상장예정주식수 포함)(백만원)', 0)
                    market_cap = mcap_mil * 1e6
                    net_income = latest.get('당기순이익(천원)', 0) * 1000
                    equity     = latest.get('자본총계(천원)', 0) * 1000
                    shares     = latest.get('기말발행주식수(보통주)(주)', 0)

                    pe_calc  = (market_cap / (net_income*4)) if net_income > 0 else None
                    pbr_calc = (market_cap / equity) if equity > 0 else None
                    eps_calc = (net_income / shares) if shares > 0 else None
                    roe      = latest.get('ROE(%)', 0) / 100

                    info = dict(
                        trailingPE      = pe_metric  if pe_metric  and 0<pe_metric<200  else (round(pe_calc,2) if pe_calc and 0<pe_calc<200 else None),
                        priceToBook     = pbr_metric if pbr_metric and 0<pbr_metric<30 else (round(pbr_calc,2) if pbr_calc and 0<pbr_calc<30 else None),
                        returnOnEquity  = round(roe,4),
                        marketCap       = market_cap if market_cap>0 else None,
                        beta            = 1.0,
                        currentPrice    = None,
                        trailingEps     = round(eps_metric) if eps_metric else (round(eps_calc) if eps_calc else None),
                        freeCashflow    = latest.get('영업활동으로인한현금흐름(천원)', 0) * 1000,
                        sharesOutstanding = shares,
                        bookValue       = bps_metric,
                        dividendYield   = (div_yield/100) if div_yield else None,
                    )
                    return info, False
            except Exception as e:
                pass

    # 2) yfinance fallback
    _ensure_yf_session()
    try:
        import yfinance as yf
        info = _yf_retry(lambda: yf.Ticker(ticker).info, tries=2)
        has_info = (
            _kq_has_yfinance_fundamental_info(info)
            if _kq_has_yfinance_fundamental_info is not None
            else bool(info and info.get('trailingPE'))
        )
        if has_info:
            return info, False
    except Exception:
        pass
    return _sfund(ticker), True

# ── 기술적 지표 ──────────────────────────────────────────────────────────
def _rsi(c, n=14):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_rsi(c, window=n)
    d = c.diff()
    u = d.clip(lower=0).rolling(n).mean()
    v = (-d.clip(upper=0)).rolling(n).mean().replace(0,1e-9)
    return 100 - 100/(1+u/v)

def _macd(c, f=12, s=26, sig=9):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_macd(c, fast=f, slow=s, signal=sig)
    ml = c.ewm(span=f,adjust=False).mean()-c.ewm(span=s,adjust=False).mean()
    sl = ml.ewm(span=sig,adjust=False).mean()
    return ml, sl

def _bb(c, n=20, k=2):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_bb(c, window=n, num_std=k)
    m = c.rolling(n).mean(); sd = c.rolling(n).std()
    return m+k*sd, m, m-k*sd

def _atr(df, n=14):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_atr(df, window=n)
    h,l,pc = df['High'].squeeze(),df['Low'].squeeze(),df['Close'].squeeze().shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ── 단일 종목 전체 분석 ──────────────────────────────────────────────────
def analyze_stock(ticker, period='1y'):
    key = f'stock:{ticker}:{period}'
    return _cached(key, 300, _analyze_stock, ticker, period)

def _analyze_stock(ticker, period='1y'):
    df, is_sample = _dl(ticker, period)
    info, info_sample = _fund_info(ticker)
    c = _c(df)

    if _KQ_MODULAR_HELPERS_READY and _kq_analyze_stock_payload is not None:
        cur_source = '엑셀'
        cur_date = c.index[-1].strftime('%Y-%m-%d')
        live_price = None
        try:
            _ensure_yf_session()
            import yfinance as yf
            t_obj = yf.Ticker(ticker)
            live_info = _yf_retry(lambda: t_obj.info, tries=2)
            if _kq_resolve_current_price_context is not None:
                context = _kq_resolve_current_price_context(c.index, live_info)
                live_price = context.get('price')
                cur_source = context.get('source') or cur_source
                cur_date = context.get('date') or cur_date
            else:
                if _kq_extract_live_price is not None:
                    live_price = _kq_extract_live_price(live_info)
                elif isinstance(live_info, dict):
                    live_price = (live_info.get('currentPrice')
                                  or live_info.get('regularMarketPrice')
                                  or live_info.get('previousClose'))
                if live_price and live_price > 0:
                    cur_source = '실시간'
                    cur_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        except Exception:
            live_price = None

        name = (UNIVERSE.get(ticker) or ETFs.get(ticker) or (ticker,'?'))[0]
        return _kq_analyze_stock_payload(
            ticker,
            df,
            period=period,
            info=info if isinstance(info, dict) else {},
            name=name,
            is_sample=is_sample,
            current_price=live_price,
            current_source=cur_source,
            current_date=cur_date,
        )

    # 지표 계산
    ma20  = c.rolling(20).mean()
    ma60  = c.rolling(60).mean()
    ma200 = c.rolling(200).mean()
    r14   = _rsi(c)
    ml, sl = _macd(c)
    up, mid, lo = _bb(c)
    atr_s = _atr(df)

    # 차트용 데이터: 서버에서 임의로 252일 제한하지 않고 요청 기간 전체를 내려준다.
    # UI에서 확대/이동/스크롤로 필요한 구간을 직접 탐색한다.
    if _KQ_MODULAR_HELPERS_READY and _kq_build_stock_chart is not None:
        chart = _kq_build_stock_chart(
            df,
            c,
            indicators=dict(
                ma20=ma20, ma60=ma60, ma200=ma200,
                bb_up=up, bb_mid=mid, bb_lo=lo,
                rsi=r14, macd=ml, signal=sl,
            ),
            period=period,
        )
    else:
        tail = len(c)
        def _series(s):
            return [round(float(v),2) if pd.notna(v) else None
                    for v in s.iloc[-tail:].tolist()]

        chart = dict(
            period = period,
            count  = int(tail),
            dates  = [d.strftime('%Y-%m-%d') for d in c.iloc[-tail:].index],
            open   = _series(df['Open'].squeeze()),
            high   = _series(df['High'].squeeze()),
            low    = _series(df['Low'].squeeze()),
            close  = _series(c),
            ma20   = _series(ma20),
            ma60   = _series(ma60),
            ma200  = _series(ma200),
            bb_up  = _series(up),
            bb_mid = _series(mid),
            bb_lo  = _series(lo),
            vol    = [int(v) if pd.notna(v) else 0 for v in df['Volume'].squeeze().iloc[-tail:].tolist()],
            rsi    = _series(r14),
            macd   = _series(ml),
            signal = _series(sl),
            hist   = [round(float(ml.iloc[i]-sl.iloc[i]),2) if pd.notna(ml.iloc[i]) else None
                      for i in range(-tail, 0)],
        )

    # 현재 지표값
    cur   = float(c.iloc[-1])
    prev  = float(c.iloc[-2]) if len(c)>1 else cur
    chg   = (cur-prev)/prev*100
    cur_source = '엑셀'   # 가격 출처 표기
    cur_date   = c.index[-1].strftime('%Y-%m-%d')

    # ── yfinance 실시간 현재가 시도 (실패 시 엑셀 폴백) ──────────────────
    try:
        _ensure_yf_session()
        import yfinance as yf
        t_obj = yf.Ticker(ticker)
        live_info = _yf_retry(lambda: t_obj.info, tries=2)
        live_price = None
        if _kq_resolve_current_price_context is not None:
            context = _kq_resolve_current_price_context(c.index, live_info)
            live_price = context.get('price')
        elif _kq_extract_live_price is not None:
            live_price = _kq_extract_live_price(live_info)
        elif isinstance(live_info, dict):
            live_price = (live_info.get('currentPrice')
                          or live_info.get('regularMarketPrice')
                          or live_info.get('previousClose'))
        if live_price and live_price > 0:
            cur = float(live_price)
            chg = (cur - prev) / prev * 100
            if _kq_resolve_current_price_context is not None:
                cur_source = context.get('source') or '실시간'
                cur_date = context.get('date') or cur_date
            else:
                cur_source = '실시간'
                cur_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    except Exception:
        pass

    rv    = float(r14.iloc[-1])  if pd.notna(r14.iloc[-1])  else 50.
    macd_v= float(ml.iloc[-1])   if pd.notna(ml.iloc[-1])   else 0.
    sig_v = float(sl.iloc[-1])   if pd.notna(sl.iloc[-1])   else 0.
    bb_u  = float(up.iloc[-1])   if pd.notna(up.iloc[-1])   else cur*1.05
    bb_l  = float(lo.iloc[-1])   if pd.notna(lo.iloc[-1])   else cur*0.95
    ma20_v= float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else cur
    ma60_v= float(ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else cur
    ma200v= float(ma200.iloc[-1])if pd.notna(ma200.iloc[-1])else cur
    atr_v = float(atr_s.iloc[-1])if pd.notna(atr_s.iloc[-1])else cur*0.02
    def _rpx(v):
        return round(float(v), 2) if abs(float(v)) < 1000 else round(float(v), -1)
    entry_v = cur
    buy_low_v = max(0, entry_v - 0.25 * atr_v)
    buy_high_v = entry_v + 0.25 * atr_v
    target_v = entry_v + 2.0 * atr_v
    stop_v = max(0, entry_v - 1.5 * atr_v)

    # Alpha Decay (신호별 반감기 측정) — 로보신호 신뢰도 가중에 먼저 사용
    ad = _alpha_single(c, active_window=10)

    # 로보어드바이저 신호 (기존 단순 임계 신호)
    s_rsi  = 1 if rv<30 else (-1 if rv>70 else 0)
    s_macd = 1 if macd_v>sig_v else -1
    s_bb   = 1 if cur<bb_l else (-1 if cur>bb_u else 0)
    s_ma20 = 1 if cur>ma20_v else -1
    s_ma60 = 1 if cur>ma60_v else -1

    if _KQ_MODULAR_HELPERS_READY:
        score = _kq_legacy_robo_score(s_rsi, s_macd, s_bb, s_ma20, s_ma60)
        sig = _kq_score_to_signal(score)
    else:
        sub_values = {'s_rsi':s_rsi, 's_macd':s_macd, 's_bb':s_bb, 's_ma20':s_ma20, 's_ma60':s_ma60}
        raw = sum(float(sub_values[key]) * weight for key, weight in ROBO_SIGNAL_WEIGHTS.items())
        score = round(max(0, min(100, (raw+80)/1.6)), 1)
        sig = '매수' if score >= ROBO_BUY_THRESHOLD else (
            '매도' if score <= ROBO_SELL_THRESHOLD else '관망'
        )

    # Alpha Decay 신뢰도 가중 종합 신호 (신규)
    cw = _confidence_weighted_robo(ad, s_rsi, s_macd, s_bb, s_ma20, s_ma60)

    # 재무
    pe   = info.get('trailingPE')  if isinstance(info,dict) else None
    pbr  = info.get('priceToBook') if isinstance(info,dict) else None
    roe  = info.get('returnOnEquity') if isinstance(info,dict) else None
    mcap = info.get('marketCap')   if isinstance(info,dict) else None

    name = (UNIVERSE.get(ticker) or ETFs.get(ticker) or (ticker,'?'))[0]
    return dict(
        ticker=ticker, name=name,
        is_sample=is_sample,
        cur=round(cur,0), chg=round(chg,2),
        cur_source=cur_source, cur_date=cur_date,
        chart=chart,
        indicators=dict(rsi=round(rv,1),macd=round(macd_v,2),signal=round(sig_v,2),
                        bb_upper=round(bb_u,0),bb_lower=round(bb_l,0),
                        ma20=round(ma20_v,0),ma60=round(ma60_v,0),ma200=round(ma200v,0),
                        atr=round(atr_v,0)),
        robo=dict(score=score,signal=sig,
                  s_rsi=s_rsi,s_macd=s_macd,s_bb=s_bb,s_ma20=s_ma20,s_ma60=s_ma60,
                  entry=_rpx(entry_v), buy_low=_rpx(buy_low_v), buy_high=_rpx(buy_high_v),
                  target=_rpx(target_v), stoploss=_rpx(stop_v),
                  rr=round(2/(1.5),2),
                  cw_score=cw['score'], cw_signal=cw['signal'],
                  confidence=cw['confidence'], exit_days=cw['exit_days'],
                  valid_days=cw.get('valid_days'), validity_basis=cw.get('validity_basis'),
                  validity_text=cw.get('validity_text'),
                  cw_detail=cw['detail']),
        alpha=ad,
        fund=dict(pe=pe,pbr=pbr,roe=roe,mcap=mcap),
    )

# ── Alpha Decay (단일 종목) ──────────────────────────────────────────────
# 신호명 → (방향, 해당 robo 서브신호 키). 방향: +1=매수성, -1=매도성
_LEGACY_SIGNAL_DIRECTION = {
    'RSI 과매도':      (+1, 's_rsi'),
    'RSI 과매수':      (-1, 's_rsi'),
    'MACD 골든크로스': (+1, 's_macd'),
    'MACD 데드크로스': (-1, 's_macd'),
    'BB 하단터치':     (+1, 's_bb'),
    'BB 상단터치':     (-1, 's_bb'),
}
SIGNAL_DIRECTION = dict(_KQ_CONFIG_SIGNAL_DIRECTION or _LEGACY_SIGNAL_DIRECTION)

_LEGACY_ROBO_SIGNAL_WEIGHTS = {'s_rsi': 20, 's_macd': 25, 's_bb': 20, 's_ma20': 15, 's_ma60': 20}
_LEGACY_ROBO_SIGNAL_LABELS = [
    ('s_rsi', 'RSI(14)'),
    ('s_macd', 'MACD'),
    ('s_bb', '볼린저밴드'),
    ('s_ma20', 'MA20'),
    ('s_ma60', 'MA60'),
]
ROBO_SIGNAL_WEIGHTS = dict(_KQ_CONFIG_ROBO_SIGNAL_WEIGHTS or _LEGACY_ROBO_SIGNAL_WEIGHTS)
ROBO_SIGNAL_LABELS = list(_KQ_CONFIG_ROBO_SIGNAL_LABELS or _LEGACY_ROBO_SIGNAL_LABELS)
ROBO_BUY_THRESHOLD = int(_KQ_CONFIG_ROBO_BUY_THRESHOLD or 65)
ROBO_SELL_THRESHOLD = int(_KQ_CONFIG_ROBO_SELL_THRESHOLD or 35)

def _alpha_single(
    c,
    active_window=10,
    limit_threshold=0.295,
    quality_scores=None,
    min_signal_quality=None,
    weight_by_quality=False,
):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_alpha_single(
            c,
            horizons=HORIZONS,
            active_window=active_window,
            limit_threshold=limit_threshold,
            quality_scores=quality_scores,
            min_signal_quality=min_signal_quality,
            weight_by_quality=weight_by_quality,
        )
    def _exp(t,a,lam,cc): return a*np.exp(-lam*t)+cc
    r14=_rsi(c); ml,sl=_macd(c); up,_,lo=_bb(c)
    signals={
        'RSI 과매도':(r14<30),
        'RSI 과매수':(r14>70),
        'MACD 골든크로스':((ml>sl)&(ml.shift(1)<=sl.shift(1))),
        'MACD 데드크로스':((ml<sl)&(ml.shift(1)>=sl.shift(1))),
        'BB 하단터치':(c<=lo),
        'BB 상단터치':(c>=up),
    }
    limit_hit = c.pct_change().abs() >= limit_threshold
    res={}
    for sname,mask in signals.items():
        try:
            raw_dates = mask[mask].index
            active_slice = mask.iloc[-active_window:] if len(mask) else mask
            is_active = bool(active_slice.any()) if len(active_slice) else False
            active_age = None
            if is_active:
                active_locs = np.flatnonzero(mask.to_numpy(dtype=bool))
                if len(active_locs):
                    active_age = int(len(mask) - 1 - active_locs[-1])
            dates = []
            excluded_limit = 0
            for dt in raw_dates:
                idx = c.index.get_loc(dt)
                if isinstance(idx, slice):
                    idx = idx.stop - 1
                bad_limit = bool(limit_hit.iloc[idx])
                if idx + 1 < len(c):
                    bad_limit = bad_limit or bool(limit_hit.iloc[idx + 1])
                if bad_limit:
                    excluded_limit += 1
                    continue
                dates.append(dt)
            direction = SIGNAL_DIRECTION.get(sname, (1, None))[0]
            if len(dates)<5:
                res[sname]=dict(count=int(len(dates)),half_life=None,
                    raw_count=int(len(raw_dates)), excluded_limit=int(excluded_limit),
                    horizon_rets={}, fit_y=None, is_active=is_active,
                    active_age=active_age, active_window=active_window,
                    direction=direction, status='표본 부족')
                continue
            h_rets={}
            for h in HORIZONS:
                rets=[]
                for dt in dates:
                    try:
                        idx=c.index.get_loc(dt)
                        if isinstance(idx, slice):
                            idx = idx.stop - 1
                        if idx+h<len(c):
                            raw_ret = float((c.iloc[idx+h]-c.iloc[idx])/c.iloc[idx])
                            rets.append(direction * raw_ret)
                    except: pass
                h_rets[h]=float(np.mean(rets)) if rets else None
            valid=[(h,r) for h,r in h_rets.items() if r is not None]
            hl=None; fit_y=None
            HL_CAP = max(HORIZONS) * 1.5   # 관측 구간 밖으로 발산하는 반감기는 신뢰 불가 → 상한 처리
            if len(valid)>=4:
                try:
                    ta=np.array([x[0] for x in valid]); ya=np.array([x[1] for x in valid])
                    popt,_=curve_fit(_exp,ta,ya,p0=[ya[0],0.1,0.],maxfev=3000)
                    if popt[1]>0:
                        hl_raw = float(np.log(2)/popt[1])
                        if 0 < hl_raw <= HL_CAP:
                            hl=round(hl_raw,1)
                        # hl_raw가 범위 밖이면(거의 감쇠하지 않는 신호) hl=None으로 둔다
                    fit_y=[round(float(_exp(t,*popt))*100,3) for t in range(1,31)]
                except: pass
            res[sname]=dict(count=int(len(dates)), raw_count=int(len(raw_dates)),
                excluded_limit=int(excluded_limit), half_life=hl,
                horizon_rets={str(h):round(r*100,3) if r is not None else None for h,r in h_rets.items()},
                fit_y=fit_y, is_active=is_active, active_age=active_age,
                active_window=active_window, direction=direction,
                status=('측정됨' if hl is not None else '감쇠 불안정'))
        except: pass
    return res

# ── 반감기 → 신뢰도(0~10) 매핑 ────────────────────────────────────────────
# 설계 의도: 반감기가 "있다"는 것 자체가 (그 신호 발생 후 평균 수익률이
# 지수감쇠 패턴을 보일 만큼) 통계적으로 일관된 신호라는 뜻이다.
# 반감기가 너무 짧으면(<3일) 효과가 너무 빨리 사라져 포착하기 어렵고,
# 반감기가 너무 길면(>20일) "감쇠"라기보다 추세에 가까워 신호 고유의 정보력이 약하다.
# 5~12일 구간을 가장 다루기 좋은 신뢰도로 보고, 그 바깥은 점수를 낮춘다.
def _half_life_to_confidence(hl, count):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_half_life_to_confidence(hl, count)
    if hl is None or hl <= 0:
        return 2.0  # 반감기 추정 불가 → 낮은 신뢰도(데이터 부족/불규칙)
    if hl < 3:        conf = 5.0 + (hl/3)*2.0      # 1~5 → 5~7
    elif hl <= 12:     conf = 7.0 + (12-hl)/9*3.0   # 3~12 → 7~10 (5일 근처가 최고점에 가까움)
    elif hl <= 20:      conf = 7.0 - (hl-12)/8*3.0   # 12~20 → 4~7
    else:               conf = max(2.0, 4.0 - (hl-20)/10*2.0)  # 20+ → 점차 하락
    # 신호 발생 횟수가 적으면(표본 부족) 신뢰도 약간 할인
    if count < 10:   conf *= 0.85
    elif count < 20:  conf *= 0.95
    return round(min(10.0, max(0.0, conf)), 1)

def _confidence_weighted_robo(alpha, s_rsi, s_macd, s_bb, s_ma20, s_ma60):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_confidence_weighted_robo(alpha, s_rsi, s_macd, s_bb, s_ma20, s_ma60)
    """
    Alpha Decay 반감기 기반 신뢰도로 각 신호를 가중하여
    '신뢰도 가중 종합 스코어'와 '권장 청산 D-day'를 계산한다.
    - alpha: _alpha_single() 결과 (신호명 → {half_life, count, is_active, ...})
    - s_*: 기존 ±1/0 방향 신호
    """
    # 신호명 → (서브신호 키, 기존 가중치) 매핑. MA20/MA60은 Alpha Decay 측정 대상이 아니므로
    # 반감기 데이터가 없는 만큼 기본 신뢰도(보통 신뢰도)로 처리한다.
    sub_values  = {'s_rsi':s_rsi, 's_macd':s_macd, 's_bb':s_bb, 's_ma20':s_ma20, 's_ma60':s_ma60}

    # 서브신호별 "현재 활성화된 Alpha Decay 신호명" 매핑 (방향이 일치하는 것만)
    active_by_sub = {}
    for sname, (direction, subkey) in SIGNAL_DIRECTION.items():
        d = alpha.get(sname)
        if not d or not d.get('is_active'):
            continue
        if sub_values.get(subkey) == direction:
            active_by_sub.setdefault(subkey, []).append((sname, d))

    detail = []
    weighted_sum = 0.0
    weight_total = 0.0
    decay_days = []   # 활성 신호들의 반감기 (가중 평균용)
    decay_weights = []

    for subkey, label in ROBO_SIGNAL_LABELS:
        direction = sub_values[subkey]
        w0 = ROBO_SIGNAL_WEIGHTS[subkey]
        matches = active_by_sub.get(subkey, [])

        if direction == 0:
            # 중립 신호는 가중치 0
            detail.append(dict(key=subkey, label=label, direction=0,
                               confidence=None, half_life=None, weight=0.0, signal_name=None))
            continue

        if matches:
            # 같은 서브신호에 여러 Alpha Decay 신호가 매칭되는 경우는 없지만 안전하게 첫 번째 사용
            sname, d = matches[0]
            conf = _half_life_to_confidence(d.get('half_life'), d.get('count', 0))
            hl = d.get('half_life')
            decay_basis = 'half_life' if hl else None
            if hl:
                decay_days.append(hl)
                decay_weights.append(conf)
            else:
                # v2: half_life 측정 실패 시 (한국 시장 모멘텀 특성)
                # horizon_rets에서 peak 수익 시점을 fallback으로 사용
                h_rets = d.get('horizon_rets', {})
                valid_rets = [(int(h), float(r)) for h, r in h_rets.items()
                              if r is not None and r > 0]
                if valid_rets:
                    peak_day, _ = max(valid_rets, key=lambda x: x[1])
                    decay_days.append(float(peak_day))
                    decay_weights.append(conf * 0.7)  # 반감기보다 약한 신뢰도
                    hl = peak_day  # detail 표시용
                    decay_basis = 'peak'
        else:
            # Alpha Decay 측정 대상이 아닌 신호(MA20/MA60) 또는 활성 신호 미매칭
            conf = 5.0  # 중간 신뢰도(데이터 없음 → 가중치 절반)
            sname, hl = None, None
            decay_basis = None

        eff_w = w0 * (conf/10.0)
        weighted_sum += direction * eff_w
        weight_total += w0  # 정규화는 원래 가중치 총합 기준 유지(점수 스케일 보존)

        detail.append(dict(key=subkey, label=label, direction=direction,
                           confidence=conf, half_life=hl, weight=round(eff_w,2),
                           signal_name=sname, decay_basis=decay_basis))

    # 종합 스코어: 신뢰도 가중합을 기존 스케일(0~100)에 맞춰 변환
    score_cw = round(max(0, min(100, (weighted_sum+80)/1.6)), 1)
    sig_cw = '매수' if score_cw >= ROBO_BUY_THRESHOLD else (
        '매도' if score_cw <= ROBO_SELL_THRESHOLD else '관망'
    )

    # 종합 신뢰도(0~10): 활성 신호들의 신뢰도를 가중치(eff_w)로 가중평균
    active_confs  = [d['confidence'] for d in detail if d['confidence'] is not None and d['direction']!=0]
    active_eff_w  = [d['weight'] for d in detail if d['direction']!=0]
    overall_conf  = round(float(np.average(active_confs, weights=active_eff_w)),1) \
                    if active_confs and sum(active_eff_w)>0 else None

    # 권장 청산 D-day: 활성 Alpha Decay 신호들의 반감기를 신뢰도로 가중평균
    exit_days = round(float(np.average(decay_days, weights=decay_weights)),1) \
                if decay_days and sum(decay_weights)>0 else None
    if exit_days is not None:
        valid_days = exit_days
        validity_basis = 'alpha_decay'
        validity_text = f'활성 신호의 Alpha Decay 기반 유효기간 (반감기/peak 종합, {exit_days:.0f}일)'
    else:
        valid_days = 5 if sig_cw in ('매수', '매도') else 10
        validity_basis = 'review_interval'
        validity_text = '활성 신호 없음, 보수적 재점검 기간 적용 (5~10일)'

    return dict(
        score=score_cw, signal=sig_cw,
        confidence=overall_conf,
        exit_days=exit_days,
        valid_days=valid_days,
        validity_basis=validity_basis,
        validity_text=validity_text,
        detail=detail,
    )

# ── 역방향 DCF (Reverse Gordon-Growth) ───────────────────────────────────
# 일반적인 DCF는 "성장률을 가정 → 적정가치 산출"이지만,
# 역방향 DCF는 반대로 "현재가가 적정가라고 가정 → 시장이 내재한 성장률(g)을 역산"한다.
# Gordon Growth: P = EPS(1+g) / (r-g)  (r = 요구수익률 = RF+ERP)
#   →  g = (r·P - EPS) / (P + EPS)
# g가 낮을수록(=시장이 낮은 성장만 기대하고 있는데도 그 가격) 저평가 후보로 해석한다.
def _reverse_dcf_growth(price, eps, r=RF+ERP):
    if _KQ_MODULAR_HELPERS_READY:
        return _kq_reverse_dcf_growth(price, eps, r)
    if price is None or eps is None or price <= 0 or eps <= 0:
        return None
    g = (r*price - eps) / (price + eps)
    if g < -0.5 or g > 0.5:   # 비현실적인 범위는 추정 실패로 간주
        return None
    return float(g)

# ── 전체 스크리닝 ────────────────────────────────────────────────────────
def run_screener():
    return _cached('screener', 1800, _run_screener)

def _prewarm_screener_cache():
    """스크리너 병렬 실행 전 필요한 parquet 캐시를 한 번만 로드한다."""
    if '_dl_mod' not in globals():
        return
    price_sheets = getattr(_dl_mod, 'PRICE_SHEETS', {})
    metric_sheets = getattr(_dl_mod, 'METRIC_SHEETS', {})
    if _kq_prewarm_screener_cache is not None:
        _kq_prewarm_screener_cache(_dl_mod._load_price_parquet, price_sheets, metric_sheets)
        return
    for kind in ('open', 'high', 'low', 'close', 'volume'):
        fname = price_sheets.get(kind)
        if fname:
            try:
                _dl_mod._load_price_parquet(fname)
            except Exception:
                pass
    for metric in ('per', 'pbr', 'eps', 'bps', 'div_yield'):
        fname = metric_sheets.get(metric)
        if fname:
            try:
                _dl_mod._load_price_parquet(fname)
            except Exception:
                pass

def _fetch_one(ticker):
    try:
        import yfinance as yf
        df, _ = _dl(ticker,'1y')
        info, _ = _fund_info(ticker)
        c = _c(df)
        name = UNIVERSE.get(ticker, (ticker,'?'))[0]
        if _KQ_SCREENER_HELPERS_READY:
            return ticker, _kq_build_screener_record(ticker, name, c, info, RF + ERP)
        mom = float(c.iloc[-1] / c.iloc[-60] - 1) if len(c)>=60 and float(c.iloc[-60]) > 0 else None
        s2_mom = float(c.iloc[-21] / c.iloc[-252] - 1) if len(c)>=252 and float(c.iloc[-252]) > 0 else None
        pe  = float(info.get('trailingPE',0) or 0) or None  if isinstance(info,dict) else None
        pbr = float(info.get('priceToBook',0) or 0) or None if isinstance(info,dict) else None
        roe = float(info.get('returnOnEquity',0) or 0) or None if isinstance(info,dict) else None
        eps = float(info.get('trailingEps',0) or 0) or None if isinstance(info,dict) else None
        cur = float(c.iloc[-1])
        chg = float((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100) if len(c)>1 else 0.
        dcf_g = _reverse_dcf_growth(cur, eps)   # 시장이 내재한 기대성장률(역산)
        # 로보 신호 (빠른 버전)
        r14=_rsi(c); ml,sl=_macd(c)
        rv=float(r14.iloc[-1]) if pd.notna(r14.iloc[-1]) else 50.
        macd_bull = float(ml.iloc[-1])>float(sl.iloc[-1]) if pd.notna(ml.iloc[-1]) else False
        raw = (1 if rv<30 else -1 if rv>70 else 0)*20 + (1 if macd_bull else -1)*25
        score = round(max(0,min(100,(raw+45)/0.9)),1)
        sig = '매수' if score >= ROBO_BUY_THRESHOLD else (
            '매도' if score <= ROBO_SELL_THRESHOLD else '관망'
        )
        # 종목 이름 가져오기 (UNIVERSE 딕셔너리에서)
        return ticker, dict(name=name,pe=pe,pbr=pbr,roe=roe,mom=mom,s2_mom=s2_mom,cur=round(cur,0),chg=round(chg,2),
                            dcf_g=round(dcf_g,4) if dcf_g is not None else None,
                            score=score,signal=sig)
    except:
        return ticker, None

def _run_screener():
    results = {}
    price_date = None
    try:
        _dl_mod._load_price_parquet('price_종가.parquet')
        close_path = os.path.join(_dl_mod.CACHE_DIR, 'price_종가.parquet')
        close_groups = _dl_mod._PRICE_GROUPS.get(close_path, {})
        if _kq_latest_price_date_from_groups is not None:
            price_date = _kq_latest_price_date_from_groups(close_groups)
        else:
            last_dates = [s.dropna().index[-1] for s in close_groups.values()
                          if s is not None and len(s.dropna()) > 0]
            if last_dates:
                price_date = max(last_dates).strftime('%Y-%m-%d')
    except Exception:
        price_date = None

    # 시가총액 상위 종목 사용 (스크리너 + 백테스트 일관성)
    candidates = get_top_marketcap_tickers()
    if len(candidates) < len(UNIVERSE):
        print(f'  [screener] 시가총액 상위 {len(candidates)}개로 제한')

    _prewarm_screener_cache()

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in candidates}
        for f in as_completed(futs):
            t, d = f.result()
            if d: results[t] = d

    def rank(key, rev=False, allow_nonpositive=False):
        pairs = [(t,d[key]) for t,d in results.items() if d.get(key) is not None
                 and not (isinstance(d[key],float) and np.isnan(d[key])) and
                 (allow_nonpositive or (d[key]>0 if not rev else True))]
        pairs.sort(key=lambda x:x[1], reverse=rev)
        return [t for t,_ in pairs[:8]]

    if _KQ_SCREENER_HELPERS_READY:
        screeners = _kq_build_screeners(results)
        # 기존 UI는 5개 스크리너 버튼을 기준으로 구성되어 있어 응답 표면을 유지한다.
        screeners.pop('로보매수', None)
    else:
        screeners = {
            '저PER':    rank('pe'),
            '저PBR':    rank('pbr'),
            '고ROE':    rank('roe',rev=True),
            '모멘텀':   rank('mom',rev=True),
            '역방향DCF': rank('dcf_g', allow_nonpositive=True),  # 낮은 내재성장률 = 저평가 후보
            'S2모멘텀': rank('s2_mom',rev=True),
        }

    return dict(
        data=results,
        price_date=price_date,
        screeners=screeners
    )

# ── 위험기반 자산배분 (GMV · MDP · ERC) ──────────────────────────────────
# 이론 출처: 위험기반배분 강의자료 1장(GMV/MDP), 2장(Risk Parity/ERC)
#   GMV: min w'Σw                         s.t. sum(w)=1, 0<=w_i<=cap
#   MDP: max DR(w) = w'σ / sqrt(w'Σw)     s.t. sum(w)=1, 0<=w_i<=cap
#   ERC: min Σ(RC_i - σ_p/N)^2            s.t. sum(w)=1, 0<=w_i<=cap
# (RC_i = w_i * (Σw)_i / σ_p,  σ_p = sqrt(w'Σw),  Σ RC_i = σ_p 가 항상 성립)

def risk_contributions(w, cov):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_risk_contributions(w, cov)
    """각 자산의 위험기여도(RC)와 포트폴리오 변동성(sigma_p)을 반환.
    RC_i = w_i * (Σw)_i / σ_p,  sum(RC_i) == σ_p (오일러 정리)"""
    w = np.asarray(w, dtype=float)
    sigma_p = float(np.sqrt(w @ cov @ w))
    if sigma_p <= 1e-12:
        return np.zeros_like(w), 0.0
    marginal = cov @ w           # (Σw)_i  = MRC_i * sigma_p
    rc = w * marginal / sigma_p  # RC_i
    return rc, sigma_p

def _gmv_weights(cov, cap=0.5):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_gmv_weights(cov, cap=cap)
    n = cov.shape[0]
    x0 = np.ones(n)/n
    bounds = [(0.0, cap)] * n
    cons = [{'type':'eq', 'fun': lambda w: w.sum()-1.0}]
    res = minimize(lambda w: w @ cov @ w, x0, method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter':500, 'ftol':1e-10})
    w = res.x if res.success else x0
    return np.clip(w, 0, None) / np.clip(w, 0, None).sum()

def _mdp_weights(cov, vols, cap=0.5):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_mdp_weights(cov, vols, cap=cap)
    n = cov.shape[0]
    x0 = np.ones(n)/n
    bounds = [(0.0, cap)] * n
    cons = [{'type':'eq', 'fun': lambda w: w.sum()-1.0}]
    def neg_dr(w):
        num = w @ vols
        den = np.sqrt(max(w @ cov @ w, 1e-12))
        return -num/den   # DR 최대화 = -DR 최소화
    res = minimize(neg_dr, x0, method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter':500, 'ftol':1e-10})
    w = res.x if res.success else x0
    return np.clip(w, 0, None) / np.clip(w, 0, None).sum()

def _erc_weights(cov, cap=0.5):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_erc_weights(cov, cap=cap)
    n = cov.shape[0]
    x0 = np.ones(n)/n
    bounds = [(1e-6, cap)] * n
    cons = [{'type':'eq', 'fun': lambda w: w.sum()-1.0}]
    def obj(w):
        rc, sigma_p = risk_contributions(w, cov)
        target = sigma_p/n
        return float(np.sum((rc-target)**2))
    res = minimize(obj, x0, method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter':1000, 'ftol':1e-12})
    w = res.x if res.success else x0
    return np.clip(w, 0, None) / np.clip(w, 0, None).sum()

def _diversification_ratio(w, cov, vols):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_diversification_ratio(w, cov, vols)
    w = np.asarray(w); vols = np.asarray(vols)
    num = w @ vols
    den = np.sqrt(max(w @ cov @ w, 1e-12))
    return float(num/den)

# ── ETF 백테스트 ─────────────────────────────────────────────────────────
def run_backtest():
    return _cached('backtest', 3600, _run_backtest)

def _run_backtest():
    # ETF 가격 데이터 병렬 다운로드
    etf_prices = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_dl, t, '3y'): t for t in ETFs}
        for f in as_completed(futs):
            t = futs[f]; df, _ = f.result()
            etf_prices[t] = df

    ret_m = {}
    for t, df in etf_prices.items():
        try:
            c = _c(df).resample('ME').last().pct_change().dropna()
            ret_m[t] = c
        except: pass

    ret_df = pd.DataFrame(ret_m).dropna()
    if len(ret_df) < 12: return {}

    if _kq_inverse_volatility_weights and _kq_gtaa_weights and _kq_risk_based_strategy_weights:
        etf_tickers = list(ETFs.keys())
        STRATEGIES['역변동성'] = _kq_inverse_volatility_weights(
            etf_prices,
            tickers=etf_tickers,
            close_fn=_c,
        )
        STRATEGIES['GTAA'] = _kq_gtaa_weights(
            etf_prices,
            tickers=etf_tickers,
            close_fn=_c,
        )
        STRATEGIES.update(
            _kq_risk_based_strategy_weights(
                ret_df,
                gmv_fn=_gmv_weights,
                mdp_fn=_mdp_weights,
                erc_fn=_erc_weights,
            )
        )
    else:
        # 역변동성
        vols = {}
        for t in ETFs:
            try:
                c = _c(etf_prices[t]).pct_change().dropna()
                vols[t] = max(float(c.tail(60).std()), 1e-5)
            except: vols[t] = 0.01
        inv = {t:1/v for t,v in vols.items()}; tot=sum(inv.values())
        STRATEGIES['역변동성'] = {t:round(w/tot,4) for t,w in inv.items()}

        # GTAA
        active=[]
        for t,df in etf_prices.items():
            try:
                c=_c(df)
                if len(c)>=200 and float(c.iloc[-1])>float(c.rolling(200).mean().iloc[-1]):
                    active.append(t)
            except: pass
        if not active: active=list(ETFs.keys())[:3]
        STRATEGIES['GTAA']={t:round(1/len(active),4) for t in active}

        # ── 위험기반 자산배분 (GMV·MDP·ERC) ──────────────────────────────────
        # 월간 수익률(ret_df)로 공분산행렬 추정 (연율화)
        rb_tickers = list(ret_df.columns)
        cov_m = ret_df[rb_tickers].cov().values * 12.0          # 연율화 공분산
        vol_v = np.sqrt(np.diag(cov_m))                          # 연율화 변동성 벡터

        try:
            w_gmv = _gmv_weights(cov_m)
            STRATEGIES['GMV'] = {t: round(float(w),4) for t,w in zip(rb_tickers, w_gmv)}
        except Exception:
            STRATEGIES['GMV'] = None
        try:
            w_mdp = _mdp_weights(cov_m, vol_v)
            STRATEGIES['MDP'] = {t: round(float(w),4) for t,w in zip(rb_tickers, w_mdp)}
        except Exception:
            STRATEGIES['MDP'] = None
        try:
            w_erc = _erc_weights(cov_m)
            STRATEGIES['ERC'] = {t: round(float(w),4) for t,w in zip(rb_tickers, w_erc)}
        except Exception:
            STRATEGIES['ERC'] = None

    if _kq_evaluate_asset_allocation_strategies:
        return _kq_evaluate_asset_allocation_strategies(
            ret_df,
            STRATEGIES,
            risk_contributions_fn=risk_contributions,
            diversification_ratio_fn=_diversification_ratio,
            risk_based_keys=RISK_BASED_KEYS,
            risk_free_rate=RF,
        )

    results = {}
    for name, weights in STRATEGIES.items():
        if not weights: continue
        try:
            avail=[t for t in weights if t in ret_df.columns and weights.get(t,0)>0]
            if not avail: continue
            w=np.array([weights[t] for t in avail]); w/=w.sum()
            pf=ret_df[avail].dot(w)
            n=len(pf); years=n/12
            cum=(1+pf).prod()
            cagr=cum**(1/years)-1
            vol=pf.std()*np.sqrt(12)
            sharpe=(cagr-RF)/vol if vol>0 else 0
            wealth=(1+pf).cumprod()
            mdd=float(((wealth-wealth.cummax())/wealth.cummax()).min())
            cum_s=(1+pf).cumprod()*100

            # ── 위험기여도(RC) 분해: "돈 비중 vs 위험 비중" ──
            sub_cov = ret_df[avail].cov().values * 12.0
            rc, sigma_p = risk_contributions(w, sub_cov)
            rc_pct = (rc / sigma_p * 100.0) if sigma_p > 1e-12 else np.zeros_like(rc)
            dr = _diversification_ratio(w, sub_cov, np.sqrt(np.diag(sub_cov)))

            results[name]=dict(
                weights={t:round(float(weights[t]),4) for t in avail},
                metrics=dict(cagr=round(float(cagr)*100,2),vol=round(float(vol)*100,2),
                             sharpe=round(float(sharpe),3),mdd=round(float(mdd)*100,2),
                             calmar=round(float(cagr/abs(mdd)),3) if abs(mdd)>1e-6 else 0),
                cum=[round(v,2) for v in cum_s.tolist()],
                dates=[d.strftime('%Y-%m') for d in cum_s.index],
                risk_contrib={t: round(float(p),2) for t,p in zip(avail, rc_pct)},
                capital_weight={t: round(float(x)*100,2) for t,x in zip(avail, w)},
                diversification_ratio=round(dr,3),
                is_risk_based = name in RISK_BASED_KEYS,
            )
        except: pass
    return results

# ── 매크로 국면 ──────────────────────────────────────────────────────────
_REGIME_DEF = {
    '골디락스':       dict(성장='↑',물가='↓',주식=8,국채=2,원자재=-1,금=-2,현금=1,label='성장↑·물가↓',color='#3fb950',
                           hint='주식·IT 강세, 채권 중립'),
    '리플레이션':     dict(성장='↑',물가='↑',주식=6,국채=-3,원자재=9,금=3,현금=1,label='성장↑·물가↑',color='#d29922',
                           hint='원자재·에너지 수혜, 채권 약세'),
    '스태그플레이션': dict(성장='↓',물가='↑',주식=-7,국채=-4,원자재=7,금=6,현금=1,label='성장↓·물가↑',color='#f85149',
                           hint='금·원자재 헤지, 현금 방어'),
    '디플레이션':     dict(성장='↓',물가='↓',주식=-6,국채=7,원자재=-5,금=1,현금=1,label='성장↓·물가↓',color='#388bfd',
                           hint='국채 강세, 방어주·현금 선호'),
}

def _build_macro():
    """엑셀 매크로 데이터에서 현재 지표 + 국면 추정"""
    if _kq_build_macro_payload is not None:
        return _kq_build_macro_payload(
            EXCEL_MACRO,
            classify_regime_fn=getattr(_dl_mod, 'classify_regime', None),
        )

    indicators = dict(PMI=52.1, CPI_YoY=2.8, GDP_QoQ=0.7, 기준금리=3.50,
                      장기금리10Y=3.85, 장단기스프레드=0.35, VIX=18.2)
    current = '리플레이션'
    hint = '데이터 부족으로 추정값 사용'

    if EXCEL_MACRO:
        try:
            # GDP 성장률
            if 'gdp' in EXCEL_MACRO:
                gdp_df = EXCEL_MACRO['gdp']
                gc = [c for c in gdp_df.columns if '성장률' in c]
                if gc:
                    indicators['GDP_QoQ'] = round(float(gdp_df[gc[0]].dropna().iloc[-1])*100, 2)

            # 금리 (국고 1년·3년·10년)
            if 'rate' in EXCEL_MACRO:
                r = EXCEL_MACRO['rate']
                r10 = [c for c in r.columns if '국고10년' in c]
                r1  = [c for c in r.columns if '국고1년' in c]
                if r10: indicators['장기금리10Y'] = round(float(r[r10[0]].dropna().iloc[-1]), 2)
                if r1 and r10:
                    spr = float(r[r10[0]].dropna().iloc[-1]) - float(r[r1[0]].dropna().iloc[-1])
                    indicators['장단기스프레드'] = round(spr, 2)

            # 환율
            if 'fx' in EXCEL_MACRO:
                fx = EXCEL_MACRO['fx']
                usd = [c for c in fx.columns if '미국' in c or '달러' in c]
                if usd:
                    indicators['환율USD'] = round(float(fx[usd[0]].dropna().iloc[-1]), 1)

            # 국면 자동 분류
            r = _dl_mod.classify_regime(EXCEL_MACRO)
            current = r.get('regime', '리플레이션')
            hint = f"GDP {r.get('gdp',0)}% + 장단기스프레드 {r.get('spread',0)}%p → {current} 국면"
        except Exception as e:
            print(f'  [macro] 매크로 계산 오류: {e}')

    return dict(regimes=_REGIME_DEF, indicators=indicators,
                current=current, current_hint=hint)

MACRO = _build_macro()

# ─────────────────────────────────────────────────────────────────────────
#  전략 백테스트 (개별 종목 전략 + KOSPI 벤치마크 비교)
# ─────────────────────────────────────────────────────────────────────────
KOSPI_TICKER = '^KS11'   # KOSPI 지수

def _strategy_key(name, fallback):
    return name.key if name is not None else fallback

def run_strategy_backtest(
    strategy='quant',
    top_n=5,
    rebalance='M',
    period='3y',
    transaction_cost_bps=0.0,
    slippage_bps=0.0,
):
    transaction_cost_bps = float(transaction_cost_bps)
    slippage_bps = float(slippage_bps)
    quant_compare_key = _strategy_key(_KQ_QUANT_COMPARE, 'quant_compare')
    if strategy == quant_compare_key:
        key = f'stratbt_compare:{top_n}:{rebalance}:{period}:tc{transaction_cost_bps}:slip{slippage_bps}'
        return _cached(
            key,
            1800,
            _run_quant_comparison_backtest,
            top_n,
            rebalance,
            period,
            transaction_cost_bps,
            slippage_bps,
        )
    key = f'stratbt:{strategy}:{top_n}:{rebalance}:{period}:tc{transaction_cost_bps}:slip{slippage_bps}'
    return _cached(
        key,
        1800,
        _run_strategy_backtest,
        strategy,
        top_n,
        rebalance,
        period,
        transaction_cost_bps,
        slippage_bps,
    )

def _run_quant_comparison_backtest(top_n, rebalance, period, transaction_cost_bps=0.0, slippage_bps=0.0):
    quant_key = _strategy_key(_KQ_QUANT, 'quant')
    quant_s2_key = _strategy_key(_KQ_QUANT_S2, 'quant_s2')
    quant_label = (
        _kq_strategy_descriptor(quant_key).label
        if _kq_strategy_descriptor is not None else '퀀트(모멘텀)'
    )
    quant_s2_label = (
        _kq_strategy_descriptor(quant_s2_key).label
        if _kq_strategy_descriptor is not None else '퀀트(S2모멘텀)'
    )
    momentum = run_strategy_backtest(
        quant_key, top_n, rebalance, period, transaction_cost_bps, slippage_bps
    )
    if momentum.get('error'):
        return {'error': f"{quant_label}: {momentum.get('error')}"}

    s2 = run_strategy_backtest(
        quant_s2_key, top_n, rebalance, period, transaction_cost_bps, slippage_bps
    )
    if s2.get('error'):
        return {'error': f"{quant_s2_label}: {s2.get('error')}"}

    if _kq_build_quant_comparison_response is not None:
        return _kq_build_quant_comparison_response(momentum, s2)

    payload = dict(momentum)
    payload['strategy'] = 'quant_compare'
    payload['comparison_runs'] = [
        {'label': '퀀트(모멘텀)', 'color': '#388bfd', 'data': momentum},
        {'label': '퀀트(S2모멘텀)', 'color': '#3fb950', 'data': s2},
    ]
    payload['comparison_benchmark'] = {
        'label': 'KOSPI',
        'color': '#8b949e',
        'dates': momentum.get('benchmark_dates') or [],
        'equity': momentum.get('benchmark') or [],
        'metrics': momentum.get('bench_metrics') or {},
    }
    return payload

def _perf_metrics(equity, freq_per_year=12):
    if _KQ_BACKTEST_HELPERS_READY:
        return _kq_perf_metrics(equity, freq_per_year=freq_per_year, risk_free_rate=RF)
    """누적 자산곡선(Series) → 성과지표"""
    rets = equity.pct_change().dropna()
    n = len(rets)
    if n < 3: return {}
    years = n / freq_per_year
    total = float(equity.iloc[-1] / equity.iloc[0])
    cagr  = total**(1/years) - 1 if years>0 else 0
    vol   = float(rets.std() * np.sqrt(freq_per_year))
    sharpe= (cagr - RF) / vol if vol>0 else 0
    peak  = equity.cummax()
    dd    = (equity - peak) / peak
    mdd   = float(dd.min())
    calmar= cagr/abs(mdd) if abs(mdd)>1e-6 else 0
    # 승률
    win   = float((rets>0).sum() / n) if n>0 else 0
    return dict(
        total_return=round((total-1)*100,2),
        cagr=round(float(cagr)*100,2),
        vol=round(vol*100,2),
        sharpe=round(float(sharpe),3),
        mdd=round(mdd*100,2),
        calmar=round(float(calmar),3),
        win_rate=round(win*100,1),
        years=round(years,2),
    )

def _run_strategy_backtest(strategy, top_n, rebalance, period, transaction_cost_bps=0.0, slippage_bps=0.0):
    # 1) 종가 데이터 일괄 로드 (parquet 1회만 — 병렬 호출 안 함)
    #    기존 ThreadPoolExecutor 방식은 3070종목 × parquet 캐시 락 경합으로 매우 느림
    #
    # 시작일 정책: 'max'/'12y' 등은 2014-01-01 고정, 짧은 기간은 데이터 마지막에서 역산.
    BACKTEST_START = pd.Timestamp('2014-01-01')
    use_fixed_start = (
        _kq_use_fixed_start_for_period(period)
        if _kq_use_fixed_start_for_period is not None
        else period in ('12y', '15y', '20y', 'max', None) or period not in ('1y', '2y', '3y', '5y', '7y')
    )

    # data_loader의 종가 그룹 직접 사용 (한 번만 로딩)
    _dl_mod._load_price_parquet('price_종가.parquet')
    close_groups_path = os.path.join(_dl_mod.CACHE_DIR, 'price_종가.parquet')
    close_groups = _dl_mod._PRICE_GROUPS.get(close_groups_path, {})

    if not close_groups:
        return {'error': '종가 데이터 로드 실패'}

    # UNIVERSE에서 시가총액 상위 200개만 사용 (스크리너와 일관성, 학술 표준)
    # - 유동성 확보 (실전 거래 가능 종목)
    # - 생존편향 완화 (소형주 상폐 위험 회피)
    # - Fama-French, AQR 등 학술 연구의 표준 방법론
    top_tickers = set(UNIVERSE.keys())
    mcap_history = get_mcap_history()
    px = {}
    for t in top_tickers:
        code = _ticker_to_code(t)
        s = close_groups.get(code)
        if s is not None and len(s) >= 20:
            px[t] = s

    if not px:
        return {'error': 'UNIVERSE와 데이터 매칭 실패'}

    # KOSPI 벤치마크 (yfinance, 1회만)
    kospi = None
    try:
        kdf, _ = _dl(KOSPI_TICKER, period if not use_fixed_start else '12y')
        if kdf is not None and not kdf.empty:
            kospi = _c(kdf)
            # yfinance가 오늘 미체결 데이터까지 가져오면 NaN — 끝부분 정리
            kospi = kospi.dropna()
    except Exception:
        kospi = None

    if kospi is None or len(kospi) < 30:
        # 종목 평균으로 합성 벤치마크 (KOSPI 다운로드 실패 시)
        kospi = pd.DataFrame(px).mean(axis=1).dropna()

    # 2) 월말 리밸런싱 날짜
    price_df = pd.DataFrame(px)
    if _kq_prepare_strategy_price_frame is not None:
        price_df, prep_error = _kq_prepare_strategy_price_frame(
            price_df,
            period,
            period_days_fn=_period_days,
            backtest_start=BACKTEST_START,
        )
        if prep_error:
            return {'error': prep_error}
        kospi = _kq_filter_strategy_benchmark_period(
            kospi,
            period,
            backtest_start=BACKTEST_START,
        )
    else:
        if price_df.empty: return {'error':'데이터 없음'}

        # 기간 필터
        if use_fixed_start:
            # 고정 시작일 (2014-01-01)
            price_df = price_df[price_df.index >= BACKTEST_START]
        else:
            # 데이터 마지막 날짜 기준 역산 (기존 동작)
            days = _period_days(period, 1095)
            last_date = price_df.index[-1]
            cutoff = last_date - pd.Timedelta(days=days)
            price_df = price_df[price_df.index >= cutoff]

        # KOSPI도 동일 기간으로 필터
        if use_fixed_start and kospi is not None:
            kospi = kospi[kospi.index >= BACKTEST_START]

        # 종목별 최소 데이터 길이 필터 (60일 이상 데이터 있는 종목만 사용)
        valid_cols = [c for c in price_df.columns if price_df[c].notna().sum() >= 60]
        if not valid_cols:
            return {'error':'유효 종목 부족'}
        price_df = price_df[valid_cols]
        
        # 전체 기간 유지: 각 종목은 자신의 시작일부터, 빠진 날짜는 ffill
        # 핵심: dropna(how='any') 절대 금지 — 늦게 상장된 종목 때문에 전체 기간이 잘림
        price_df = price_df.ffill()
        # 모든 종목이 다 NaN인 행만 제거 (보통 첫 1~2일)
        price_df = price_df.dropna(how='all')

    rule = {'M':'ME','Q':'QE','W':'W'}.get(rebalance,'ME')
    rebal_dates = price_df.resample(rule).last().index
    rebal_dates = [d for d in rebal_dates if d in price_df.index or
                   price_df.index[price_df.index<=d].size>0]

    # ── 사전 계산: 로보 전략 지표를 전체 기간에 대해 1회만 계산 ──
    # 매월 hist 슬라이스로 계산하면 144회 × 3000종목 = 매우 느림
    # 종목별 단독 계산 후 DataFrame으로 결합 (벡터화 다종목 처리는 NaN 문제 있음)
    precomputed = None
    if strategy == 'robo':
        if _kq_build_robo_precomputed_indicators is not None:
            precomputed = _kq_build_robo_precomputed_indicators(price_df)
        else:
            rsi_dict, macd_bull_dict, ma20_dict, ma60_dict = {}, {}, {}, {}
            for col in price_df.columns:
                s = price_df[col].dropna()
                if len(s) < 60:
                    continue
                # RSI (14일)
                delta = s.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                ag = gain.rolling(14, min_periods=14).mean()
                al = loss.rolling(14, min_periods=14).mean()
                rs = ag / al.replace(0, np.nan)
                rsi_dict[col] = 100 - (100 / (1 + rs))
                # MACD (12, 26, 9)
                ema12 = s.ewm(span=12, adjust=False).mean()
                ema26 = s.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd_bull_dict[col] = (macd_line > signal_line).astype(int)
                # MA20, MA60
                ma20_dict[col] = s.rolling(20).mean()
                ma60_dict[col] = s.rolling(60).mean()

            precomputed = dict(
                rsi=pd.DataFrame(rsi_dict),
                macd_bull=pd.DataFrame(macd_bull_dict),
                ma20=pd.DataFrame(ma20_dict),
                ma60=pd.DataFrame(ma60_dict),
            )

    if _KQ_BACKTEST_HELPERS_READY and _kq_run_rebalanced_strategy_backtest is not None:
        pit_selector = None
        if mcap_history is not None and not mcap_history.empty:
            pit_selector = lambda cur_date: get_top_mcap_at(mcap_history, cur_date, SCREENER_LIMIT)
        return _kq_run_rebalanced_strategy_backtest(
            price_df,
            strategy=strategy,
            top_n=top_n,
            rebalance=rebalance,
            period=period,
            benchmark=kospi,
            precomputed=precomputed,
            pit_selector=pit_selector,
            universe_names={ticker: meta[0] for ticker, meta in UNIVERSE.items()},
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )

    # 3) 매 리밸런싱 시점마다 종목 선정 → 다음 기간 수익률
    equity = [100.0]
    eq_dates = [price_df.index[0].strftime('%Y-%m-%d')]
    holdings_log = []

    daily_ret = price_df.pct_change().fillna(0)
    months = price_df.resample(rule).last()

    for i in range(len(months)-1):
        cur_date  = months.index[i]
        next_date = months.index[i+1]
        # 룩백 데이터 (cur_date 까지)
        hist = price_df.loc[:cur_date]
        if len(hist) < 60:
            continue

        # Point-in-Time 시총 필터:
        # 백테스트 시점에 이미 확인 가능한 시총만 사용해 상위 200개 종목 풀을 만든다.
        if mcap_history is not None and not mcap_history.empty:
            pit_tickers = get_top_mcap_at(mcap_history, cur_date, SCREENER_LIMIT)
            pit_available = [t for t in pit_tickers if t in hist.columns]
            if len(pit_available) < top_n:
                continue
            hist = hist[pit_available]

        # ── 종목 선정 ──
        selected = _select_for_bt(hist, strategy, top_n, precomputed, cur_date)
        if not selected:
            continue

        # 동일비중
        w = {t: 1/len(selected) for t in selected}

        # 다음 기간 수익률 (cur_date → next_date)
        period_px = price_df.loc[cur_date:next_date]
        if _KQ_BACKTEST_HELPERS_READY:
            period_ret = _kq_equal_weight_period_return(period_px, selected)
            if period_ret is None:
                continue
        else:
            if len(period_px) < 2:
                continue
            period_ret = 0.0
            valid_count = 0
            for t in selected:
                try:
                    col = period_px[t].dropna()
                    if len(col) < 2:
                        continue  # 이 기간 데이터 없는 종목 스킵
                    p0, p1 = float(col.iloc[0]), float(col.iloc[-1])
                    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
                        continue
                    r = (p1 - p0) / p0
                    period_ret += w[t] * r
                    valid_count += 1
                except Exception:
                    continue
            # 유효 종목 비율로 스케일링 (선정 종목 중 일부만 거래 가능한 경우 대비)
            if valid_count == 0:
                continue
            # 균등 가중인데 누락된 종목이 있으면 가중치 재조정
            if valid_count < len(selected):
                period_ret = period_ret * len(selected) / valid_count

        equity.append(equity[-1] * (1 + period_ret))
        eq_dates.append(next_date.strftime('%Y-%m-%d'))
        holdings_log.append(dict(date=cur_date.strftime('%Y-%m'),
                                 tickers=[UNIVERSE.get(t,(t,))[0] for t in selected]))

    eq_series = pd.Series(equity, index=pd.to_datetime(eq_dates))

    # 4) KOSPI 벤치마크 정규화 (같은 기간)
    bench_vals, bench_dates = [], []
    if kospi is not None:
        if _KQ_BACKTEST_HELPERS_READY:
            bench_vals, bench_dates = _kq_normalize_benchmark_to_equity(kospi, eq_series.index)
        else:
            kb = kospi.reindex(eq_series.index, method='ffill').dropna()
            if len(kb) > 1:
                kb = kb / kb.iloc[0] * 100
                bench_vals  = [round(float(v),2) for v in kb.tolist()]
                bench_dates = [d.strftime('%Y-%m-%d') for d in kb.index]

    # 5) 성과지표
    freq = {'M':12,'Q':4,'W':52}.get(rebalance,12)
    strat_metrics = _perf_metrics(eq_series, freq)

    bench_metrics = {}
    excess = {}
    if bench_vals:
        bench_series = pd.Series(bench_vals, index=pd.to_datetime(bench_dates))
        bench_metrics = _perf_metrics(bench_series, freq)
        # 알파/베타
        sr = eq_series.pct_change().dropna()
        br = bench_series.pct_change().dropna()
        common = sr.index.intersection(br.index)
        if len(common) > 5:
            srx, brx = sr.loc[common], br.loc[common]
            beta = float(np.cov(srx, brx)[0,1] / np.var(brx)) if np.var(brx)>0 else 1.0
            alpha_ann = (strat_metrics.get('cagr',0) - bench_metrics.get('cagr',0))
            excess = dict(beta=round(beta,3), alpha=round(alpha_ann,2),
                          excess_return=round(strat_metrics.get('total_return',0)-bench_metrics.get('total_return',0),2))

    # 6) 언더워터(낙폭) 곡선
    if _KQ_BACKTEST_HELPERS_READY:
        underwater = _kq_underwater_curve(eq_series)
    else:
        peak = eq_series.cummax()
        underwater = [round(float((eq_series.iloc[i]-peak.iloc[i])/peak.iloc[i])*100,2)
                      for i in range(len(eq_series))]

    return dict(
        strategy=strategy, top_n=top_n, rebalance=rebalance, period=period,
        equity=[round(v,2) for v in eq_series.tolist()],
        dates=[d.strftime('%Y-%m-%d') for d in eq_series.index],
        benchmark=bench_vals, benchmark_dates=bench_dates,
        underwater=underwater,
        metrics=strat_metrics,
        bench_metrics=bench_metrics,
        excess=excess,
        holdings=holdings_log[-6:],   # 최근 6개월 보유종목
        n_rebalance=len(holdings_log),
    )

def _select_for_bt(hist, strategy, top_n, precomputed=None, cur_date=None):
    if _KQ_BACKTEST_HELPERS_READY:
        return _kq_select_for_backtest(hist, strategy, top_n, precomputed, cur_date)
    """백테스트용 종목 선정 (hist = cur_date 까지 가격 데이터)
    precomputed: 사전 계산된 지표 dict (로보 전략에서만 사용)
    cur_date: 현재 리밸런싱 시점
    """
    if strategy == 'quant':
        # 기본 모멘텀 — 최근 3개월 가격 흐름
        if len(hist) < 60:
            return []
        mom = (hist.iloc[-1] / hist.iloc[-60] - 1)
        mom = mom.dropna()
        return mom.nlargest(top_n).index.tolist()

    if strategy in ('quant_s2', 's2_momentum'):
        # S2 모멘텀(12-1개월) — 21일 전 종가 / 252일 전 종가
        if len(hist) < 252:
            return []
        mom = (hist.iloc[-21] / hist.iloc[-252] - 1)
        mom = mom.dropna()
        return mom.nlargest(top_n).index.tolist()

    elif strategy == 'robo':
        if len(hist) < 60:
            return []

        # 사전 계산된 지표 lookup (200배 빠름)
        if precomputed is not None and cur_date is not None:
            # pandas 3.0에서 .asof() 동작이 변경됨 → reindex(method='ffill') 사용
            # cur_date 이하의 가장 최근 영업일 자동 매칭 (월말이 휴일인 경우 대비)
            def _at(df_, date):
                """date 이하의 가장 최근 영업일 row 반환"""
                idx = df_.index[df_.index <= date]
                if len(idx) == 0:
                    return pd.Series(np.nan, index=df_.columns)
                return df_.loc[idx[-1]]

            rsi_last = _at(precomputed['rsi'], cur_date)
            macd_bull = _at(precomputed['macd_bull'], cur_date) == 1
            ma20 = _at(precomputed['ma20'], cur_date)
            ma60 = _at(precomputed['ma60'], cur_date)
            cur = hist.iloc[-1]

            # PiT 필터로 hist 컬럼이 매월 달라질 수 있으므로,
            # 사전계산 지표도 현재 종목 풀(hist.columns)에 맞춰 정렬한다.
            cols = hist.columns
            rsi_last = rsi_last.reindex(cols)
            macd_bull = macd_bull.reindex(cols).fillna(False)
            ma20 = ma20.reindex(cols)
            ma60 = ma60.reindex(cols)
            cur = cur.reindex(cols)
        else:
            # 폴백: 매번 계산
            delta = hist.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14, min_periods=14).mean()
            avg_loss = loss.rolling(14, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi_last = rsi.iloc[-1]
            ema12 = hist.ewm(span=12, adjust=False).mean()
            ema26 = hist.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_bull = (macd_line.iloc[-1] > signal_line.iloc[-1])
            ma20 = hist.rolling(20).mean().iloc[-1]
            ma60 = hist.rolling(60).mean().iloc[-1]
            cur = hist.iloc[-1]

        # 점수 계산 (벡터 연산)
        # pandas Series끼리 직접 비교하면 PiT 필터 이후 라벨 불일치가 날 수 있어,
        # 현재 종목 풀(cols) 기준으로 재정렬한 뒤 numpy 배열로 계산한다.
        cols = hist.columns
        rsi_last = pd.to_numeric(pd.Series(rsi_last).reindex(cols), errors='coerce')
        macd_bull = pd.Series(macd_bull).reindex(cols).fillna(False).astype(bool)
        ma20 = pd.to_numeric(pd.Series(ma20).reindex(cols), errors='coerce')
        ma60 = pd.to_numeric(pd.Series(ma60).reindex(cols), errors='coerce')
        cur = pd.to_numeric(pd.Series(cur).reindex(cols), errors='coerce')

        rsi_v = rsi_last.to_numpy(dtype=float)
        macd_v = macd_bull.to_numpy(dtype=bool)
        ma20_v = ma20.to_numpy(dtype=float)
        ma60_v = ma60.to_numpy(dtype=float)
        cur_v = cur.to_numpy(dtype=float)

        rsi_score = np.where(rsi_v < 30, 20, np.where(rsi_v > 70, -20, 0))
        macd_score = np.where(macd_v, 25, -25)
        ma20_score = np.where(cur_v > ma20_v, 15, -15)
        ma60_score = np.where(cur_v > ma60_v, 20, -20)

        total_score = pd.Series(
            rsi_score + macd_score + ma20_score + ma60_score,
            index=cols
        )
        # NaN 제거 (데이터 부족 종목)
        valid_mask = rsi_last.notna() & ma20.notna() & ma60.notna() & cur.notna()
        total_score = total_score[valid_mask]

        # 양수 점수 우선
        positive = total_score[total_score > 0]
        if len(positive) >= top_n:
            return positive.nlargest(top_n).index.tolist()
        else:
            return total_score.nlargest(top_n).index.tolist()

    return list(hist.columns[:top_n])



# ── 추천 포트폴리오 리포트 ────────────────────────────────────────────────
def _normalize_weights(weights):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_normalize_weights(weights)
    clean = {k: float(v) for k, v in weights.items() if v and v > 0}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean.items()}


def _combine_weight_sets(weight_sets):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_combine_weight_sets(weight_sets)
    out = {}
    total_mix = 0.0
    for mix_w, weights in weight_sets:
        if not weights:
            continue
        total_mix += mix_w
        for t, w in weights.items():
            out[t] = out.get(t, 0.0) + mix_w * float(w)
    if total_mix > 0:
        out = {t: w / total_mix for t, w in out.items()}
    return _normalize_weights(out)


def _current_regime_snapshot():
    if _KQ_REGIME_HELPERS_READY:
        return _kq_current_regime_snapshot(
            REGIME_MODEL,
            fallback_current=MACRO.get('current', '리플레이션'),
        )
    """현재 국면과 확률, 전환 통계를 추천 포트폴리오용으로 정리."""
    try:
        if REGIME_MODEL is not None and REGIME_MODEL.trained:
            result = REGIME_MODEL.predict_current()
            probs = result.get('probs', {}) or {}
            current = result.get('current_regime') or (max(probs, key=probs.get) if probs else MACRO.get('current', '리플레이션'))
            confidence = float(probs.get(current, 0.0)) if probs else 0.0
            next_q = result.get('next_quarter', {}) or {}
            duration_avg = result.get('duration_avg', {}) or {}
            model_type = result.get('model_type', 'Rule-based')
        else:
            current = MACRO.get('current', '리플레이션')
            probs = {current: 1.0}
            confidence = 1.0
            next_q = {}
            duration_avg = {}
            model_type = 'Rule-based'
    except Exception:
        current = MACRO.get('current', '리플레이션')
        probs = {current: 1.0}
        confidence = 1.0
        next_q = {}
        duration_avg = {}
        model_type = 'Rule-based'

    return dict(current=current, confidence=confidence, probs=probs,
                next_quarter=next_q, duration_avg=duration_avg, model_type=model_type)


_LEGACY_META_COMPONENTS = [
    dict(name='동일비중', weight=0.40, reason='OOS 최저 MDD/최고 Calmar 후보'),
    dict(name='영구포트폴리오', weight=0.35, reason='OOS 최고 Sharpe 후보'),
    dict(name='올웨더', weight=0.25, reason='채권 중심 방어 배분'),
]
META_COMPONENTS = [dict(comp) for comp in (_KQ_META_COMPONENTS or _LEGACY_META_COMPONENTS)]

_LEGACY_REGIME_TARGETS = {
    '골디락스': {
        '069500.KS':0.45, '229200.KS':0.15, '148070.KS':0.20,
        '132030.KS':0.05, '153130.KS':0.10, '114260.KS':0.05,
    },
    '리플레이션': {
        '069500.KS':0.32, '229200.KS':0.08, '130680.KS':0.18,
        '132030.KS':0.22, '114260.KS':0.08, '153130.KS':0.12,
    },
    '스태그플레이션': {
        '069500.KS':0.15, '229200.KS':0.03, '130680.KS':0.12,
        '132030.KS':0.35, '114260.KS':0.10, '153130.KS':0.25,
    },
    '디플레이션': {
        '069500.KS':0.18, '229200.KS':0.02, '148070.KS':0.45,
        '114260.KS':0.15, '153130.KS':0.15, '132030.KS':0.05,
    },
}
REGIME_TARGETS = {
    regime: dict(weights)
    for regime, weights in (_KQ_REGIME_TARGETS or _LEGACY_REGIME_TARGETS).items()
}


def _meta_base_weights():
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_meta_base_weights(META_COMPONENTS, STRATEGIES)
    sets = []
    for comp in META_COMPONENTS:
        weights = STRATEGIES.get(comp['name'])
        if weights:
            sets.append((comp['weight'], weights))
    return _combine_weight_sets(sets)


def _regime_probability_blend(snapshot, current_weight=0.70, next_weight=0.30, regimes=None):
    if _KQ_PORTFOLIO_HELPERS_READY and _kq_regime_probability_blend is not None:
        return _kq_regime_probability_blend(
            snapshot,
            current_weight=current_weight,
            next_weight=next_weight,
            regimes=tuple(regimes or REGIME_TARGETS.keys()),
        )
    regime_names = list(regimes or REGIME_TARGETS.keys())
    current_probs = snapshot.get('probs') or {}
    if not isinstance(current_probs, dict):
        current_probs = {}
    next_probs = snapshot.get('next_quarter') or {}
    if not isinstance(next_probs, dict):
        next_probs = {}
    current = snapshot.get('current') or snapshot.get('current_regime')
    if not current_probs and current:
        current_probs = {str(current): 1.0}
    blended = {}
    for regime in regime_names:
        try:
            cur_prob = float(current_probs.get(regime, 0.0))
        except Exception:
            cur_prob = 0.0
        try:
            nxt_prob = float(next_probs.get(regime, 0.0))
        except Exception:
            nxt_prob = 0.0
        prob = current_weight * cur_prob + next_weight * nxt_prob
        if prob > 0:
            blended[regime] = prob
    if not blended and current in regime_names:
        blended[str(current)] = 1.0
    return _normalize_weights(blended)


def _probability_weighted_regime_target(snapshot):
    if _KQ_PORTFOLIO_HELPERS_READY and _kq_probability_weighted_regime_target is not None:
        return _kq_probability_weighted_regime_target(snapshot, REGIME_TARGETS)
    blend = _regime_probability_blend(snapshot, regimes=REGIME_TARGETS.keys())
    if not blend:
        current = snapshot.get('current') or snapshot.get('current_regime') or '리플레이션'
        return _normalize_weights(REGIME_TARGETS.get(str(current), REGIME_TARGETS['리플레이션']))
    return _combine_weight_sets(
        (prob, REGIME_TARGETS.get(regime))
        for regime, prob in blend.items()
        if REGIME_TARGETS.get(regime)
    )

def _portfolio_validity(snapshot):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_portfolio_validity(snapshot)
    current = snapshot['current']
    confidence = float(snapshot.get('confidence', 0.0) or 0.0)
    duration_q = snapshot.get('duration_avg', {}).get(current)
    try:
        duration_q = float(duration_q)
    except Exception:
        duration_q = None
    base_days = duration_q * 63 if duration_q and duration_q > 0 else 30
    conf_factor = 0.55 + min(1.0, max(0.0, confidence)) * 0.60
    valid_days = int(round(max(10, min(90, base_days * conf_factor))))
    stay_prob = None
    nq = snapshot.get('next_quarter') or {}
    if current in nq:
        try:
            stay_prob = float(nq[current])
        except Exception:
            stay_prob = None
    return dict(
        valid_days=valid_days,
        valid_weeks=round(valid_days / 5, 1),
        regime_duration_quarters=round(duration_q, 2) if duration_q else None,
        confidence=round(confidence * 100, 1),
        next_same_regime_prob=round(stay_prob * 100, 1) if stay_prob is not None else None,
        review_rule='국면 확률 60% 이하 또는 다음 분기 전환확률 40% 이상이면 조기 재평가',
        rebalance='월간 점검 / 분기 리밸런싱 기본',
    )


def _auto_regime_tilt(snapshot):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_auto_regime_tilt(snapshot)
    """국면 신뢰도에 따라 국면 틸트 강도를 보수적으로 자동 조절한다.

    v2 개선 (validation_recommend_portfolio.py 검증 결과 반영):
      - 신뢰도 50% 미만: 무틸트 (베이스 100% 유지)
      - 신뢰도 50~70%: 약한 틸트 (8%)
      - 신뢰도 70~85%: 보통 틸트 (15%)
      - 신뢰도 85%+:   강한 틸트 (20%)
      - 다음 분기 불안정 시 ×0.5 (이전 ×0.75)
      - 최종 범위: 0~22% (이전 8~35%)
    """
    confidence = float(snapshot.get('confidence', 0.0) or 0.0)
    current = snapshot.get('current')
    next_q = snapshot.get('next_quarter') or {}

    stay_prob = None
    if current in next_q:
        try:
            stay_prob = float(next_q[current])
        except Exception:
            stay_prob = None

    if confidence < 0.50:
        return dict(
            regime_tilt=0.0,
            confidence_level='무틸트',
            confidence=round(confidence * 100, 1),
            next_same_regime_prob=round(stay_prob * 100, 1) if stay_prob is not None else None,
            rule='신뢰도 50% 미만은 베이스 포트폴리오 그대로 유지 (검증 결과 기반)',
        )
    elif confidence < 0.70:
        regime_tilt = 0.08
        level = '낮음'
    elif confidence < 0.85:
        regime_tilt = 0.15
        level = '보통'
    else:
        regime_tilt = 0.20
        level = '높음'

    if stay_prob is not None:
        if stay_prob < 0.35:
            regime_tilt *= 0.50
            stability_note = '국면 전환 임박, 틸트 약화'
        elif stay_prob > 0.60:
            stability_note = '국면 안정 유지'
        else:
            stability_note = '국면 중립'
    else:
        stability_note = '다음분기 정보 없음'

    regime_tilt = float(max(0.0, min(0.22, regime_tilt)))

    return dict(
        regime_tilt=regime_tilt,
        confidence_level=level,
        confidence=round(confidence * 100, 1),
        next_same_regime_prob=round(stay_prob * 100, 1) if stay_prob is not None else None,
        stability_note=stability_note,
        rule='신뢰도 70%+ 보통 틸트, 85%+ 강한 틸트, 50% 미만은 베이스 유지 (검증 기반)',
    )

def _signal_weight_multiplier(signal):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_signal_weight_multiplier(signal)
    """로보/Alpha Decay 신호를 포트폴리오 비중 미세조정 배율로 변환한다."""
    action = signal.get('cw_signal') or signal.get('signal') or '관망'
    try:
        confidence = float(signal.get('confidence') or 0.0)
    except Exception:
        confidence = 0.0
    if confidence > 1.0:
        confidence = confidence / 10.0
    confidence = max(0.0, min(1.0, confidence))

    try:
        exit_days = signal.get('exit_days')
        exit_days = float(exit_days) if exit_days is not None else None
    except Exception:
        exit_days = None

    decay_factor = 1.0
    if exit_days is not None:
        if exit_days <= 3:
            decay_factor = 0.35
        elif exit_days <= 7:
            decay_factor = 0.70
        elif exit_days >= 20:
            decay_factor = 0.85

    strength = confidence * decay_factor
    if action == '매수':
        multiplier = 1.0 + 0.10 * strength
    elif action == '매도':
        multiplier = 1.0 - 0.12 * strength
    else:
        multiplier = 1.0

    return round(float(max(0.75, min(1.12, multiplier))), 4)


def _apply_signal_tilt(weights, signal_map):
    if _KQ_PORTFOLIO_HELPERS_READY:
        return _kq_apply_signal_tilt(weights, signal_map)
    adjusted = {}
    multipliers = {}
    for ticker, weight in weights.items():
        multiplier = _signal_weight_multiplier(signal_map.get(ticker, {}))
        multipliers[ticker] = multiplier
        adjusted[ticker] = float(weight) * multiplier
    return _normalize_weights(adjusted), multipliers


def recommend_portfolio():
    return _cached('recommend_portfolio', 300, _build_recommend_portfolio)


def _load_recommendation_regime_alpha_summary():
    if not (_KQ_PORTFOLIO_HELPERS_READY and _kq_load_regime_alpha_summary is not None):
        return []
    candidates = [
        os.path.join(BASE_DIR, 'data', 'validation', 'regime_alpha_decay_summary.csv'),
        os.path.join(BASE_DIR, 'data', 'validation', 'regime_alpha_decay_results_summary.csv'),
    ]
    for path in candidates:
        rows = _kq_load_regime_alpha_summary(path)
        if rows:
            return rows
    return []


def _build_recommend_portfolio():
    snapshot = _current_regime_snapshot()
    current = snapshot['current']
    base = _meta_base_weights()
    target = _probability_weighted_regime_target(snapshot)
    auto = _auto_regime_tilt(snapshot)
    regime_tilt = auto['regime_tilt']
    pre_signal_weights = _combine_weight_sets([(1.0 - regime_tilt, base), (regime_tilt, target)])
    validity = _portfolio_validity(snapshot)

    signal_map = {}
    for ticker in pre_signal_weights:
        signal = (
            _kq_default_asset_signal()
            if _kq_default_asset_signal is not None
            else dict(signal='관망', score=None, cw_signal='관망', cw_score=None,
                      confidence=None, exit_days=None, cur=None, chg=None, cur_date=None, error=None)
        )
        try:
            a = analyze_stock(ticker, '1y')
            if _kq_stock_analysis_to_asset_signal is not None:
                signal = _kq_stock_analysis_to_asset_signal(a)
            else:
                rb = a.get('robo', {})
                signal.update(dict(signal=rb.get('signal'), score=rb.get('score'),
                                   cw_signal=rb.get('cw_signal'), cw_score=rb.get('cw_score'),
                                   confidence=rb.get('confidence'), exit_days=rb.get('exit_days'),
                                   cur=a.get('cur'), chg=a.get('chg'), cur_date=a.get('cur_date')))
        except Exception as e:
            signal['error'] = str(e)
        signal_map[ticker] = signal

    if _KQ_PORTFOLIO_HELPERS_READY and _kq_build_recommendation_report is not None:
        return _kq_build_recommendation_report(
            snapshot=snapshot,
            base_weights=base,
            regime_target=target,
            signal_map=signal_map,
            etf_meta=ETFs,
            components=META_COMPONENTS,
            regime_alpha_summary=_load_recommendation_regime_alpha_summary(),
        )

    final_weights, signal_multipliers = _apply_signal_tilt(pre_signal_weights, signal_map)

    assets = []
    for ticker, weight in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
        if weight < 0.005:
            continue
        name, asset_type, stance = ETFs.get(ticker, (ticker, '기타', ''))
        assets.append(dict(
            ticker=ticker, name=name, asset_type=asset_type, stance=stance,
            weight=round(weight * 100, 1),
            pre_signal_weight=round(pre_signal_weights.get(ticker, 0) * 100, 1),
            signal_multiplier=signal_multipliers.get(ticker, 1.0),
            signal=signal_map.get(ticker, {}),
        ))

    action_counts = {'매수':0, '관망':0, '매도':0}
    for a in assets:
        sig = a.get('signal', {}).get('cw_signal') or a.get('signal', {}).get('signal') or '관망'
        action_counts[sig] = action_counts.get(sig, 0) + 1

    base_pct = round((1.0 - regime_tilt) * 100, 0)
    tilt_pct = round(regime_tilt * 100, 0)
    return dict(
        title='국면 기반 완성형 추천 포트폴리오',
        regime=snapshot,
        validity=validity,
        automation=auto,
        construction=dict(
            method=f'안정성 검증 포트폴리오 앙상블 {base_pct:.0f}% + 확률가중 국면 틸트 {tilt_pct:.0f}% + 로보/Alpha Decay 미세조정',
            components=META_COMPONENTS,
            base_weights={k: round(v*100, 1) for k, v in base.items()},
            regime_probability_blend={k: round(v*100, 1) for k, v in _regime_probability_blend(snapshot).items()},
            regime_target_source='현재 국면 확률 70% + 다음 분기 국면 확률 30%',
            regime_target={k: round(v*100, 1) for k, v in target.items()},
            pre_signal_weights={k: round(v*100, 1) for k, v in pre_signal_weights.items()},
            signal_multipliers=signal_multipliers,
        ),
        weights={k: round(v*100, 1) for k, v in final_weights.items()},
        assets=assets,
        action_counts=action_counts,
        message='국면은 자산 비중을 정하고, 로보신호와 Alpha Decay는 구성 자산의 진입/청산 시점을 보조합니다.',
        caveat='투자 조언이 아닌 분석용 결과입니다. ETF 분배금/세금/거래비용은 별도 고려가 필요합니다.',
    )
# ── JSON 직렬화 ──────────────────────────────────────────────────────────
def _clean(obj):
    if _kq_clean_json_value is None:
        raise RuntimeError('JSON serialization helper is unavailable')
    return _kq_clean_json_value(obj)

def _api_error_payload(error):
    if _kq_error_payload is None:
        raise RuntimeError('API error payload helper is unavailable')
    return _kq_error_payload(error)

def _attach_market_report_context(payload, regime_order=None):
    if _kq_build_market_report_context is None:
        payload['market_report'] = dict(
            available=False,
            source=None,
            summary='시장시황보고서 분석 helper를 사용할 수 없어 정량 매크로 국면만 사용합니다.',
        )
        return payload
    current = payload.get('current_regime') or payload.get('current')
    order = payload.get('regime_order') or regime_order
    payload['market_report'] = _kq_build_market_report_context(
        BASE_DIR,
        current_regime=current,
        regime_order=order,
    )
    return payload



def build_market_report_response():
    current = None
    regime_order = None
    try:
        snapshot = _current_regime_snapshot()
        current = snapshot.get('current') or snapshot.get('current_regime')
    except Exception:
        current = None
    try:
        import regime_model as _rm_mod
        regime_order = list(getattr(_rm_mod, 'REGIMES', []) or []) or None
    except Exception:
        regime_order = None
    if _kq_build_market_report_context is None:
        return dict(ok=False, market_report=dict(available=False, reports=[], summary='시장시황보고서 helper를 사용할 수 없습니다.'))
    return dict(ok=True, market_report=_kq_build_market_report_context(BASE_DIR, current_regime=current, regime_order=regime_order))


def register_market_report(payload):
    if _kq_save_user_market_report is None:
        raise RuntimeError('시장시황보고서 등록 helper를 사용할 수 없습니다.')
    payload = payload or {}
    saved = _kq_save_user_market_report(
        BASE_DIR,
        title=str(payload.get('title') or payload.get('name') or '사용자 등록 리포트'),
        text=str(payload.get('text') or payload.get('content') or ''),
        url=str(payload.get('url') or ''),
        source=str(payload.get('source') or '사용자 등록'),
    )
    response = build_market_report_response()
    response['saved'] = saved
    return response

def build_regime_ai_response():
    """AI 매크로 국면 인식 결과 반환 payload를 만든다."""
    if _kq_build_regime_ai_payload is not None:
        regime_desc = None
        regime_order = None
        try:
            import regime_model as _rm_mod
            regime_desc = _rm_mod.REGIME_DESC
            regime_order = _rm_mod.REGIMES
        except Exception:
            pass
        payload = _kq_build_regime_ai_payload(
            REGIME_MODEL,
            MACRO,
            regime_desc=regime_desc,
            regime_order=regime_order,
        )
        return _attach_market_report_context(payload, regime_order=regime_order)

    if REGIME_MODEL is None or not REGIME_MODEL.trained:
        fallback = dict(MACRO)
        fallback['model_type'] = 'Rule-based (모델 미학습)'
        fallback['model_status'] = 'not_trained'
        return _attach_market_report_context(fallback)

    result = REGIME_MODEL.predict_current()
    if result is None:
        raise RuntimeError('예측 실패')

    try:
        import regime_model as _rm_mod
        result['regime_desc'] = _rm_mod.REGIME_DESC
        result['regime_order'] = _rm_mod.REGIMES
    except Exception:
        pass

    result['model_status'] = 'trained'
    result['sample_count'] = (
        len(REGIME_MODEL.train_data) if REGIME_MODEL.train_data is not None else 0
    )
    return _attach_market_report_context(result)

def build_health_response():
    """Local startup health payload for smoke tests and team handoff."""
    module_flags = {
        'api': _kq_handle_dispatched_get_safely is not None,
        'data': bool(_KQ_DATA_HELPERS_READY),
        'analyzer': bool(_KQ_MODULAR_HELPERS_READY),
        'portfolio': bool(_KQ_PORTFOLIO_HELPERS_READY),
        'backtest': bool(_KQ_BACKTEST_HELPERS_READY),
        'regime': bool(_KQ_REGIME_HELPERS_READY),
        'screener': bool(_KQ_SCREENER_HELPERS_READY),
    }
    cache_dir = getattr(_dl_mod, 'CACHE_DIR', None) if '_dl_mod' in globals() else None
    if _kq_build_server_health_payload is not None:
        return _kq_build_server_health_payload(
            base_dir=BASE_DIR,
            module_flags=module_flags,
            excel_data=EXCEL_DATA,
            excel_fin=EXCEL_FIN,
            excel_macro=EXCEL_MACRO,
            cache_dir=cache_dir,
            regime_model=REGIME_MODEL,
            universe_count=len(UNIVERSE),
            etf_count=len(ETFs),
        )

    modules = module_flags
    data_status = {
        'excel_data': EXCEL_DATA is not None,
        'excel_fin': EXCEL_FIN is not None,
        'excel_macro': EXCEL_MACRO is not None,
        'cache_dir': cache_dir,
    }
    regime_status = {
        'trained': REGIME_MODEL is not None and getattr(REGIME_MODEL, 'trained', False),
        'model_type': (
            getattr(REGIME_MODEL.classifier, 'model_type', None)
            if REGIME_MODEL is not None and getattr(REGIME_MODEL, 'classifier', None) is not None
            else None
        ),
    }
    if _kq_build_health_payload is not None:
        return _kq_build_health_payload(
            base_dir=BASE_DIR,
            modules=modules,
            data_status=data_status,
            universe_count=len(UNIVERSE),
            etf_count=len(ETFs),
            regime_status=regime_status,
        )
    return dict(
        ok=all(modules.values()) and len(UNIVERSE) > 0 and len(ETFs) > 0,
        modules=modules,
        data=data_status,
        regime=regime_status,
        counts=dict(universe=len(UNIVERSE), etfs=len(ETFs)),
    )

# ── HTTP 서버 ─────────────────────────────────────────────────────────────
if _KQThreadingReusableHTTPServer is None:
    raise RuntimeError('HTTP server runtime helper is unavailable')

class KQServer(_KQThreadingReusableHTTPServer):
    pass

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a): pass

    def _cors(self):
        if _kq_cors_headers is None:
            raise RuntimeError('CORS header helper is unavailable')
        for key, value in _kq_cors_headers():
            self.send_header(key, value)

    def _no_cache(self):
        if _kq_no_cache_headers is None:
            raise RuntimeError('No-cache header helper is unavailable')
        for key, value in _kq_no_cache_headers():
            self.send_header(key, value)

    def do_OPTIONS(self):
        self._empty(200)

    def _response_writer(self):
        if _kq_make_response_writer is None:
            return None
        write_body = getattr(getattr(self, 'wfile', None), 'write', None)
        if write_body is None:
            return None
        return _kq_make_response_writer(
            send_response=self.send_response,
            send_header=self.send_header,
            end_headers=self.end_headers,
            write_body=write_body,
        )

    def _empty(self, code):
        writer = self._response_writer()
        if writer is not None:
            return writer.empty(code)
        if _kq_send_empty_response is None:
            raise RuntimeError('Empty response helper is unavailable')
        return _kq_send_empty_response(
            send_response=self.send_response,
            send_header=self.send_header,
            end_headers=self.end_headers,
            code=code,
        )

    def _api_services(self):
        if _kq_build_server_api_services is None:
            raise RuntimeError('API service registry helper is unavailable')
        return _kq_build_server_api_services(
            health=build_health_response,
            macro_payload=MACRO,
            stock=analyze_stock,
            screen=run_screener,
            backtest=run_backtest,
            stratbt=run_strategy_backtest,
            regime_ai=build_regime_ai_response,
            recommend_portfolio=recommend_portfolio,
            market_report=build_market_report_response,
        )

    def _read_json_body(self, max_bytes=2000000):
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0:
            return {}
        if length > max_bytes:
            raise ValueError('요청 본문이 너무 큽니다. 2MB 이하 텍스트만 등록하세요.')
        raw = self.rfile.read(length)
        return json.loads(raw.decode('utf-8-sig'))

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if path == '/api/market_report':
            return self._market_report_register()
        return self._empty(404)

    def do_GET(self):
        if _kq_handle_dispatched_get_safely is not None:
            _kq_handle_dispatched_get_safely(
                self.path,
                self._api_services(),
                send_file=self._file,
                send_json=self._json,
                send_empty=self._empty,
                send_error=lambda exc: self._json(_api_error_payload(exc), 500),
                on_error=lambda exc: traceback.print_exc(),
            )
            return

        if _kq_handle_legacy_get is not None:
            _kq_handle_legacy_get(
                self.path,
                send_file=self._file,
                send_json=self._json,
                send_empty=self._empty,
                stock=self._stock,
                screen=self._screen,
                backtest=self._backtest,
                stratbt=self._stratbt,
                macro_payload=MACRO,
                regime_ai=self._regime_ai,
                recommend_portfolio=self._recommend_portfolio,
                health=build_health_response,
            )
            return

        self._json(_api_error_payload('API routing helpers are unavailable'), 500)

    def _file(self,name,ct):
        if _kq_serve_static_file is None:
            raise RuntimeError('Static file helper is unavailable')
        writer = self._response_writer()
        if writer is None:
            raise RuntimeError('HTTP response body writer is unavailable')
        return _kq_serve_static_file(
            BASE_DIR,
            name,
            ct,
            send_body=lambda data, content_type: writer.body(data, content_type, code=200),
            send_empty=self._empty,
        )

    def _json(self, obj, code=200):
        writer = self._response_writer()
        if writer is not None:
            return writer.json(obj, code=code, cleaner=_clean)
        if _kq_send_json_response is None:
            raise RuntimeError('JSON response helper is unavailable')
        write_body = getattr(getattr(self, 'wfile', None), 'write', None)
        if write_body is None:
            raise RuntimeError('HTTP response body writer is unavailable')
        return _kq_send_json_response(
            obj,
            send_response=self.send_response,
            send_header=self.send_header,
            end_headers=self.end_headers,
            write_body=write_body,
            code=code,
            cleaner=_clean,
        )

    def _json_action(self, action):
        if _kq_send_json_action is None:
            raise RuntimeError('JSON action helper is unavailable')
        return _kq_send_json_action(
            action,
            send_json=self._json,
            error_payload_fn=_api_error_payload,
            on_error=lambda exc: traceback.print_exc(),
        )

    def _stock(self, qs):
        if _kq_run_stock_action is None:
            raise RuntimeError('Stock action helper is unavailable')
        return _kq_run_stock_action(
            qs,
            analyze_stock=analyze_stock,
            json_action=self._json_action,
        )

    def _screen(self):
        if _kq_run_json_service_action is None:
            raise RuntimeError('JSON service action helper is unavailable')
        return _kq_run_json_service_action(run_screener, json_action=self._json_action)

    def _market_report(self):
        if _kq_run_json_service_action is None:
            raise RuntimeError('JSON service action helper is unavailable')
        return _kq_run_json_service_action(build_market_report_response, json_action=self._json_action)

    def _market_report_register(self):
        try:
            payload = self._read_json_body()
            self._json(register_market_report(payload))
        except Exception as exc:
            traceback.print_exc()
            self._json(_api_error_payload(exc), 400)

    def _regime_ai(self):
        """AI 매크로 국면 인식 결과 반환 (TabPFN + HMM)"""
        if _kq_run_json_service_action is None:
            raise RuntimeError('JSON service action helper is unavailable')
        return _kq_run_json_service_action(build_regime_ai_response, json_action=self._json_action)


    def _recommend_portfolio(self):
        if _kq_run_json_service_action is None:
            raise RuntimeError('JSON service action helper is unavailable')
        return _kq_run_json_service_action(recommend_portfolio, json_action=self._json_action)
    def _backtest(self):
        if _kq_run_json_service_action is None:
            raise RuntimeError('JSON service action helper is unavailable')
        return _kq_run_json_service_action(run_backtest, json_action=self._json_action)

    def _stratbt(self, qs):
        if _kq_run_strategy_backtest_action is None:
            raise RuntimeError('Strategy backtest action helper is unavailable')
        return _kq_run_strategy_backtest_action(
            qs,
            run_strategy_backtest=run_strategy_backtest,
            json_action=self._json_action,
        )

def main():
    runtime_helpers = {
        'startup intro': _kq_startup_intro_messages,
        'yfinance status': _kq_yfinance_status_messages,
        'server URL': _kq_server_url,
        'server ready messages': _kq_server_ready_messages,
        'HTTP server factory': _kq_create_http_server,
        'browser policy runner': _kq_run_server_with_browser_policy,
    }
    missing = [name for name, helper in runtime_helpers.items() if helper is None]
    if missing:
        raise RuntimeError('Runtime helpers are unavailable: ' + ', '.join(missing))

    for line in _kq_startup_intro_messages():
        print(line)

    yfinance_connected = _ensure_yf_session()
    for line in _kq_yfinance_status_messages(yfinance_connected):
        print(line)

    url = _kq_server_url('127.0.0.1', PORT)
    for line in _kq_server_ready_messages(url):
        print(line)

    srv = _kq_create_http_server(KQServer, Handler, host='127.0.0.1', port=PORT)
    _kq_run_server_with_browser_policy(srv, os.environ, url)
    return

if __name__ == '__main__':
    main()




















