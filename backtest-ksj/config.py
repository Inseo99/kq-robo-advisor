"""백테스트 전역 설정 (backtest-ksj).

모든 파라미터를 한 곳에 모아 재현성을 보장한다.
사용자 정의(2026-07-07 확정):
  - 유니버스: Point-in-Time 시가총액 상위 300 (KOSPI+KOSDAQ, 원천 캐시 커버리지 의존)
  - 종목선정: 12-1 모멘텀(최근 1개월 제외 과거 11개월) 상위 20, 동일가중
  - 거래비용: 왕복 0.5% (편도 0.25%)  ->  cost = COST_ONE_WAY * Σ|Δw|
  - 현금수익: 연 2% (월 2%/12, 주 2%/52 단리)
  - 주간 위험회피 신호: 개월 창을 주 단위 환산(9M→39주, 10M→43주, 5M→22주)
  - 벤치마크: KODEX 200 (069500)
"""

from __future__ import annotations

import os

# ── 경로 ─────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(REPO, "data", "cache")
MACRO = os.path.join(REPO, "data", "macro")
BT_DATA = os.path.join(HERE, "data")          # 파생/외부 데이터 캐시 (VIX, credit 등)
RESULTS = os.path.join(HERE, "results")

# ── 유니버스 / 종목선정 ──────────────────────────────────────────────────
UNIVERSE_SIZE = 300          # PIT 시총 상위 N
TOP_N = 20                   # 모멘텀 상위 보유 종목 수
MOM_LOOKBACK_M = 12          # 모멘텀 기준: t-12개월
MOM_SKIP_M = 1               # 최근 1개월 제외 -> t-1개월
MCAP_KEY = "시가총액(티커-상장예정주식수 포함)(백만원)"

# ── 거래비용 / 현금 ──────────────────────────────────────────────────────
COST_ONE_WAY = 0.0025        # 편도 0.25% (왕복 0.5%). cost = COST_ONE_WAY * Σ|Δw|
CASH_ANNUAL = 0.02
CASH_MONTHLY = CASH_ANNUAL / 12     # 0.1667%/월 (사용자 명시)
CASH_WEEKLY = CASH_ANNUAL / 52      # ≈0.0385%/주

# ── 위험회피 신호 임계값 (t1 / t2) ───────────────────────────────────────
# t1: 아래 3개 중 2개 이상 위험 -> 주식비중 0%
T1_TREND_MA_M = 9            # ① kodex200 < 9개월 이동평균  (주간: 39주)
T1_VIX_TH = 18.6            # ② VIX > 18.6
T1_CREDIT_Z_TH = 1.78      # ③ 미국 신용스프레드 z > 1.78
T1_CREDIT_Z_WIN_D = 105    # ③ z-score 기준 ~5개월 = 21거래일×5 (일별 창). 월/주 공통.
T1_CREDIT_Z_WIN_M = 5      # (구) 월간 리샘플 z 창 — 5개점은 z 상한≈1.79로 사실상 미발화하여 미사용
T1_MIN_ON = 2              # 2개 이상 켜지면 risk-off

# t2: kodex200 종가 > 10개월 이동평균 -> 100% 보유 / 아니면 전액 현금
T2_MA_M = 10               # (주간: 43주)

# 개월 -> 주 환산 (사용자 확정)
WK_PER_MONTH = 4.345
T1_TREND_MA_W = 39
T2_MA_W = 43
T1_CREDIT_Z_WIN_W = 22

# ── Walk-forward 폴드 (IS / OOS) ─────────────────────────────────────────
# (start, end) 포함 구간
FOLDS = [
    {"name": "Fold1", "is": ("2014-01-01", "2018-12-31"), "oos": ("2019-01-01", "2020-12-31")},
    {"name": "Fold2", "is": ("2016-01-01", "2020-12-31"), "oos": ("2021-01-01", "2022-12-31")},
    {"name": "Fold3", "is": ("2018-01-01", "2022-12-31"), "oos": ("2023-01-01", "2024-12-31")},
    {"name": "Fold4", "is": ("2020-01-01", "2024-12-31"), "oos": ("2025-01-01", "2026-05-31")},
]

# 전체 백테스트 구간 (모멘텀 12M 워밍업 때문에 실제 매매는 2015-02부터)
BT_START = "2014-01-01"
BT_END = "2026-06-30"

# ── 전략 정의 ────────────────────────────────────────────────────────────
# variant:   s1(위험회피 없음) / s2(t1) / s3(t2)
# cadence:   'M'(월간) / 'W'(주간)
# selection: 'mom20'(12-1 모멘텀 상위20 동일가중) / 'kd200'(KODEX200 단일 보유)
STRATEGIES = [
    # 종목선정 = 모멘텀20 (기존)
    {"code": "s1m", "cadence": "M", "variant": "s1", "selection": "mom20", "label": "S1-월간 (모멘텀20)"},
    {"code": "s2m", "cadence": "M", "variant": "s2", "selection": "mom20", "label": "S2-월간 (+t1 위험회피)"},
    {"code": "s3m", "cadence": "M", "variant": "s3", "selection": "mom20", "label": "S3-월간 (+t2 추세추종)"},
    {"code": "s1w", "cadence": "W", "variant": "s1", "selection": "mom20", "label": "S1-주간 (모멘텀20)"},
    {"code": "s2w", "cadence": "W", "variant": "s2", "selection": "mom20", "label": "S2-주간 (+t1 위험회피)"},
    {"code": "s3w", "cadence": "W", "variant": "s3", "selection": "mom20", "label": "S3-주간 (+t2 추세추종)"},
    # 종목선정 = KODEX200 단일 보유 (선정방식만 변경, 나머지 동일)
    {"code": "s1m-kd200", "cadence": "M", "variant": "s1", "selection": "kd200", "label": "S1-월간 (KODEX200)"},
    {"code": "s2m-kd200", "cadence": "M", "variant": "s2", "selection": "kd200", "label": "S2-월간 (KODEX200 +t1)"},
    {"code": "s3m-kd200", "cadence": "M", "variant": "s3", "selection": "kd200", "label": "S3-월간 (KODEX200 +t2)"},
    {"code": "s1w-kd200", "cadence": "W", "variant": "s1", "selection": "kd200", "label": "S1-주간 (KODEX200)"},
    {"code": "s2w-kd200", "cadence": "W", "variant": "s2", "selection": "kd200", "label": "S2-주간 (KODEX200 +t1)"},
    {"code": "s3w-kd200", "cadence": "W", "variant": "s3", "selection": "kd200", "label": "S3-주간 (KODEX200 +t2)"},
]

BENCHMARK_CODE = "069500"   # KODEX 200
BENCHMARK_LABEL = "KODEX 200 (벤치마크)"

# 연율화 계수
PERIODS_PER_YEAR = {"M": 12, "W": 52}
