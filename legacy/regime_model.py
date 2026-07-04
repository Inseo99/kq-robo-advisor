"""
regime_model.py — 매크로 국면 인식 모델 (TabPFN + HMM 계층 구조)

1층: TabPFN — 현재 국면 분류 (4국면 확률)
2층: HMM    — 국면 전환 매트릭스 + 평균 지속 기간
3층: 국면별 Alpha Decay 적응형 임계값

설치:
  pip install tabpfn hmmlearn lightgbm
  (TabPFN이 없으면 LightGBM으로 자동 폴백)
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ─── 모델 사용 가능성 체크 ─────────────────────────────────────────
# 환경변수:
#   KQ_DISABLE_TABPFN=1  → TabPFN 강제 비활성화 (기본값: 1)
#   KQ_DISABLE_TABPFN=0  → TabPFN 활성화
#   TABPFN_API_KEY=xxx   → TabPFN 라이선스 API 키 (priorlabs.ai에서 발급)
import os as _os

# TabPFN 환경 설정 — import 전에 환경변수로 동의 처리
_api_key = _os.environ.get('TABPFN_API_KEY', '').strip()
DISABLE_TABPFN = _os.environ.get('KQ_DISABLE_TABPFN', '1').strip() == '1'

if not DISABLE_TABPFN and _api_key:
    # TabPFN 라이선스 자동 승인 (브라우저 열림 방지)
    _os.environ['TABPFN_USER_TOKEN'] = _api_key       # 신버전
    _os.environ['TABPFN_LOGIN_TOKEN'] = _api_key      # 구버전
    _os.environ['TABPFN_API_KEY'] = _api_key
    _os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

HAS_TABPFN  = False
HAS_LGBM    = False
HAS_HMM     = False

if not DISABLE_TABPFN:
    try:
        from tabpfn import TabPFNClassifier
        HAS_TABPFN = True
    except ImportError:
        pass

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    pass

try:
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    pass


REGIMES = ['골디락스', '리플레이션', '스태그플레이션', '디플레이션']
REGIME_DESC = {
    '골디락스':       dict(label='성장↑·물가↓', color='#3fb950',
                          desc='주식 강세, 채권 중립',
                          best_assets=['주식', '성장주', '기술주']),
    '리플레이션':     dict(label='성장↑·물가↑', color='#d29922',
                          desc='원자재 수혜, 채권 약세',
                          best_assets=['원자재', '에너지', '금융주', '가치주']),
    '스태그플레이션': dict(label='성장↓·물가↑', color='#f85149',
                          desc='금·원자재 헤지, 현금 방어',
                          best_assets=['금', '원자재', '현금', '필수소비재']),
    '디플레이션':     dict(label='성장↓·물가↓', color='#388bfd',
                          desc='국채 강세, 방어주 선호',
                          best_assets=['국채', '방어주', '유틸리티', '현금']),
}


# ─────────────────────────────────────────────────────────────────
#  1) 국면 라벨링 — 매크로 데이터 → 4국면 자동 라벨
# ─────────────────────────────────────────────────────────────────
def make_regime_labels(macro_data, return_features=False):
    """
    매크로 데이터로부터 시점별 국면 라벨 생성.
    macro_data: data_loader.load_macro() 결과 dict
                {'gdp':DF, 'rate':DF, 'fx':DF, 'trade':DF, ...}

    분기 단위로 다음 피처 + 라벨 산출:
      - gdp_growth: 분기 GDP 성장률 (전년대비 또는 전분기대비)
      - spread:     장단기 금리 스프레드 (10년 - 1년)
      - cpi_proxy:  스프레드 역수 (인플레 프록시) — CPI 데이터 없을 때
      - usd_change: USD 환율 전분기대비 변화율
      - export_yoy: 수출 전년대비 증가율 (있으면)

    라벨 룰:
      성장↑ if gdp_growth > 중앙값 else 성장↓
      물가↑ if spread < 중앙값  else 물가↓  (스프레드 좁아짐 = 긴축/인플레)
      → 4국면 매핑
    """
    if not macro_data:
        return pd.DataFrame()

    # GDP 분기 수익률
    gdp_df = macro_data.get('gdp')
    rate_df = macro_data.get('rate')
    fx_df = macro_data.get('fx')

    if gdp_df is None or rate_df is None:
        return pd.DataFrame()

    # GDP 성장률 컬럼 찾기
    gdp_cols = [c for c in gdp_df.columns if '성장률' in c or 'growth' in c.lower()]
    if not gdp_cols:
        # 컬럼명에 '성장률' 없으면 첫 숫자 컬럼 사용
        numeric_cols = gdp_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()
        gdp_col = numeric_cols[0]
    else:
        gdp_col = gdp_cols[0]

    gdp_s = gdp_df[gdp_col].dropna()

    # 금리 스프레드
    long_cols  = [c for c in rate_df.columns if '국고10년' in c or '10년' in c]
    short_cols = [c for c in rate_df.columns if '국고1년' in c or '1년' in c]
    if not long_cols or not short_cols:
        return pd.DataFrame()
    rate_long  = rate_df[long_cols[0]].dropna()
    rate_short = rate_df[short_cols[0]].dropna()

    # 분기 단위로 리샘플
    rate_long_q  = rate_long.resample('QE').last()
    rate_short_q = rate_short.resample('QE').last()
    spread_q = (rate_long_q - rate_short_q).dropna()

    # GDP를 분기 인덱스로
    gdp_s.index = pd.to_datetime(gdp_s.index)
    gdp_q = gdp_s.resample('QE').last().ffill()

    # 환율 분기 변화율
    usd_change_q = pd.Series(dtype=float)
    if fx_df is not None:
        usd_cols = [c for c in fx_df.columns if '미국' in c or '달러' in c or 'USD' in c]
        if usd_cols:
            fx_s = fx_df[usd_cols[0]].dropna().resample('QE').last()
            usd_change_q = fx_s.pct_change()

    # 정렬 & 결합
    df = pd.DataFrame({
        'gdp_growth': gdp_q,
        'spread':     spread_q,
        'usd_change': usd_change_q,
    }).dropna(subset=['gdp_growth', 'spread'])
    df['usd_change'] = df['usd_change'].fillna(0)

    if df.empty:
        return pd.DataFrame()

    # 국면 라벨링 (중앙값 기준)
    growth_med  = df['gdp_growth'].median()
    spread_med  = df['spread'].median()
    df['growth_up']    = df['gdp_growth'] > growth_med
    df['inflation_up'] = df['spread'] < spread_med   # 스프레드 작아짐 = 인플레/긴축

    def label_row(row):
        if row['growth_up'] and not row['inflation_up']:    return '골디락스'
        elif row['growth_up'] and row['inflation_up']:      return '리플레이션'
        elif not row['growth_up'] and row['inflation_up']:  return '스태그플레이션'
        else:                                                return '디플레이션'
    df['regime'] = df.apply(label_row, axis=1)

    if return_features:
        return df[['gdp_growth', 'spread', 'usd_change', 'regime']]
    return df['regime']


# ─────────────────────────────────────────────────────────────────
#  2) TabPFN 또는 LightGBM 분류기 (자동 선택)
# ─────────────────────────────────────────────────────────────────
class RegimeClassifier:
    """
    1층 분류기: 현재 매크로 → 4국면 확률
    TabPFN 우선 (소표본 최적), 없으면 LightGBM 사용
    """
    def __init__(self):
        self.model = None
        self.regime_names = REGIMES
        self.model_type = None
        self.feature_cols = ['gdp_growth', 'spread', 'usd_change']

    def fit(self, X, y):
        """X: DataFrame[gdp_growth, spread, usd_change], y: 국면 라벨"""
        # 라벨을 정수로
        regime_to_idx = {r:i for i,r in enumerate(REGIMES)}
        y_int = np.array([regime_to_idx[r] for r in y])

        if HAS_TABPFN:
            # TabPFN 버전별 API 차이 처리
            try:
                # 신버전 (2024+): 단순 초기화
                self.model = TabPFNClassifier(device='cpu')
            except TypeError:
                # 구버전 호환
                try:
                    self.model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
                except Exception:
                    self.model = TabPFNClassifier()
            # fit 호출 — 버전별 인자 차이 대응
            try:
                self.model.fit(X.values, y_int, overwrite_warning=True)
            except TypeError:
                self.model.fit(X.values, y_int)
            self.model_type = 'TabPFN'
        elif HAS_LGBM:
            self.model = lgb.LGBMClassifier(
                objective='multiclass', num_class=4,
                n_estimators=200, learning_rate=0.05,
                max_depth=4, num_leaves=8,
                min_child_samples=2, verbose=-1
            )
            self.model.fit(X.values, y_int)
            self.model_type = 'LightGBM'
        else:
            self.model_type = 'Rule-based'
        return self

    def predict_proba(self, X):
        """입력 X에 대한 4국면 확률 반환"""
        if self.model is None:
            # 규칙 기반 폴백 (확률 대신 100% / 0%)
            return self._rule_based_proba(X)
        try:
            proba = self.model.predict_proba(X.values)
            # 4개 클래스 못 채우면 padding
            if proba.shape[1] < 4:
                full = np.zeros((proba.shape[0], 4))
                for j, c in enumerate(self.model.classes_):
                    full[:, int(c)] = proba[:, j]
                proba = full
            return proba
        except Exception:
            return self._rule_based_proba(X)

    def _rule_based_proba(self, X):
        """모델 없을 때 단순 규칙 기반"""
        out = np.zeros((len(X), 4))
        for i, row in enumerate(X.itertuples(index=False)):
            growth_up = row.gdp_growth > 0
            inflation_up = row.spread < 0.5
            if growth_up and not inflation_up:    out[i, 0] = 1.0  # 골디락스
            elif growth_up and inflation_up:      out[i, 1] = 1.0  # 리플레이션
            elif not growth_up and inflation_up:  out[i, 2] = 1.0  # 스태그플레이션
            else:                                  out[i, 3] = 1.0  # 디플레이션
        return out

    def predict_current(self, current_features):
        """
        현재 시점 매크로 → 국면별 확률 dict
        current_features: dict 또는 1행 DataFrame
        """
        if isinstance(current_features, dict):
            df = pd.DataFrame([current_features])[self.feature_cols]
        else:
            df = current_features[self.feature_cols].tail(1)
        proba = self.predict_proba(df)[0]
        return {r: float(proba[i]) for i, r in enumerate(REGIMES)}


# ─────────────────────────────────────────────────────────────────
#  3) HMM — 국면 전환 매트릭스
# ─────────────────────────────────────────────────────────────────
class RegimeTransitionModel:
    """
    2층: 과거 국면 시퀀스 → 전환 확률 매트릭스 + 평균 지속 기간
    """
    def __init__(self):
        self.transition_matrix = None  # 4x4
        self.duration_avg = {}          # 국면별 평균 지속 분기 수
        self.regime_names = REGIMES

    def fit(self, regime_sequence):
        """
        regime_sequence: pd.Series of regime labels (시간순)
        """
        seq = regime_sequence.dropna().tolist()
        if len(seq) < 4:
            self.transition_matrix = np.eye(4)
            return self

        # 전환 카운트
        n = len(REGIMES)
        counts = np.zeros((n, n))
        idx = {r:i for i,r in enumerate(REGIMES)}
        for a, b in zip(seq[:-1], seq[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
        # 행별 정규화 (각 국면에서 다음 국면 확률)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = counts / row_sums

        # 평균 지속 기간 (run-length encoding)
        runs = {r: [] for r in REGIMES}
        cur, cnt = seq[0], 1
        for s in seq[1:]:
            if s == cur:
                cnt += 1
            else:
                runs[cur].append(cnt)
                cur, cnt = s, 1
        runs[cur].append(cnt)
        self.duration_avg = {r: (np.mean(v) if v else 0) for r, v in runs.items()}

        return self

    def predict_next(self, current_regime, n_steps=1):
        """
        현재 국면 → n_steps 후 국면 확률 분포
        """
        if self.transition_matrix is None:
            return {r: 0.25 for r in REGIMES}
        idx = REGIMES.index(current_regime)
        # 시작 분포: 현재 국면 100%
        dist = np.zeros(4)
        dist[idx] = 1.0
        # 매트릭스 곱셈 n_steps 회
        for _ in range(n_steps):
            dist = dist @ self.transition_matrix
        return {r: float(dist[i]) for i, r in enumerate(REGIMES)}

    def get_transition_dict(self):
        """4x4 매트릭스를 dict로 반환 (UI 표시용)"""
        if self.transition_matrix is None:
            return {}
        out = {}
        for i, a in enumerate(REGIMES):
            out[a] = {b: round(float(self.transition_matrix[i, j]), 3)
                      for j, b in enumerate(REGIMES)}
        return out


# ─────────────────────────────────────────────────────────────────
#  4) 국면별 Alpha Decay 측정
# ─────────────────────────────────────────────────────────────────
def measure_regime_alpha_decay(price_series, regime_series, signal_func, horizons=None):
    """
    3층: 국면별로 신호 반감기 측정

    price_series: 종목 가격 시계열 (datetime index)
    regime_series: 국면 라벨 시계열 (분기, datetime index)
    signal_func: 가격 시계열 → 신호 발생 bool 시리즈 함수
    horizons: 측정할 미래 일수 [1, 3, 5, 7, 10, 15, 20]

    반환: {국면: {horizon: 평균 수익률}}
    """
    if horizons is None:
        horizons = [1, 3, 5, 7, 10, 15, 20]

    if price_series is None or regime_series is None or len(price_series) < 60:
        return {}

    # 각 일자에 국면 매핑 (분기 라벨을 일별로 ffill)
    regime_daily = regime_series.reindex(price_series.index, method='ffill')

    # 신호 발생 시점
    signal_mask = signal_func(price_series)
    signal_dates = signal_mask[signal_mask].index

    # 국면별로 분류
    result = {r: {h: [] for h in horizons} for r in REGIMES}
    for dt in signal_dates:
        regime = regime_daily.get(dt)
        if regime not in REGIMES:
            continue
        idx = price_series.index.get_loc(dt)
        for h in horizons:
            if idx + h < len(price_series):
                ret = float((price_series.iloc[idx+h] - price_series.iloc[idx]) / price_series.iloc[idx])
                result[regime][h].append(ret)

    # 평균 계산 + 반감기 피팅
    out = {}
    for regime, horizon_dict in result.items():
        avg_rets = {}
        for h, rets in horizon_dict.items():
            avg_rets[h] = float(np.mean(rets)) if len(rets) >= 3 else None
        # 반감기 계산 (간단 버전: 첫 양수 → 0에 도달하는 시간)
        valid = [(h, r) for h, r in avg_rets.items() if r is not None]
        half_life = None
        if len(valid) >= 4:
            try:
                from scipy.optimize import curve_fit
                ta = np.array([x[0] for x in valid])
                ya = np.array([x[1] for x in valid])
                popt, _ = curve_fit(
                    lambda t, a, lam, c: a * np.exp(-lam * t) + c,
                    ta, ya, p0=[ya[0], 0.1, 0.], maxfev=3000
                )
                if popt[1] > 0:
                    half_life = round(float(np.log(2) / popt[1]), 1)
            except Exception:
                pass
        out[regime] = dict(
            avg_returns={str(h): round(r*100, 3) if r else None for h, r in avg_rets.items()},
            half_life=half_life,
            sample_count=sum(len(v) for v in horizon_dict.values()),
        )
    return out


# ─────────────────────────────────────────────────────────────────
#  5) 통합 파이프라인
# ─────────────────────────────────────────────────────────────────
class HierarchicalRegimeModel:
    """
    전체 계층 모델 통합
    1층: 분류 (TabPFN/LightGBM)
    2층: 전환 (HMM 카운팅)
    3층: 국면별 Alpha Decay (별도 호출)
    """
    def __init__(self):
        self.classifier = RegimeClassifier()
        self.transitions = RegimeTransitionModel()
        self.trained = False
        self.train_data = None
        self.regime_history = None

    def fit(self, macro_data):
        """macro_data: data_loader.load_macro() 결과"""
        df = make_regime_labels(macro_data, return_features=True)
        if df.empty or len(df) < 4:
            return self

        X = df[['gdp_growth', 'spread', 'usd_change']]
        y = df['regime']

        self.classifier.fit(X, y)
        self.transitions.fit(df['regime'])
        self.train_data = df
        self.regime_history = df['regime']
        self.trained = True
        return self

    def predict_current(self, macro_data=None):
        """현재 시점 국면 확률 + 전환 예측"""
        if not self.trained:
            return None
        # 가장 최근 데이터 사용
        latest = self.train_data.tail(1).iloc[0]
        current_features = {
            'gdp_growth': float(latest['gdp_growth']),
            'spread':     float(latest['spread']),
            'usd_change': float(latest['usd_change']),
        }
        # 1층: 국면 확률
        probs = self.classifier.predict_current(current_features)
        # 가장 확률 높은 국면
        current_regime = max(probs, key=probs.get)
        # 2층: 다음 국면 전환 확률
        next_q = self.transitions.predict_next(current_regime, n_steps=1)
        next_2q = self.transitions.predict_next(current_regime, n_steps=2)

        return dict(
            current_features=current_features,
            current_regime=current_regime,
            probs=probs,
            next_quarter=next_q,
            next_2quarters=next_2q,
            duration_avg=self.transitions.duration_avg,
            transition_matrix=self.transitions.get_transition_dict(),
            model_type=self.classifier.model_type or 'Rule-based',
        )


# ─── 모듈 정보 ────────────────────────────────────────────────────
def get_model_info():
    return dict(
        tabpfn_available=HAS_TABPFN,
        lightgbm_available=HAS_LGBM,
        hmm_available=HAS_HMM,
        active_classifier=('TabPFN' if HAS_TABPFN else 'LightGBM' if HAS_LGBM else 'Rule-based'),
    )


if __name__ == '__main__':
    print('=== Regime Model 모듈 진단 ===')
    info = get_model_info()
    for k, v in info.items():
        print(f'  {k}: {v}')
