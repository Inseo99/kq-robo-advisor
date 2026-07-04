"""
regime_pit.py — Point-in-Time 안전한 매크로 국면 분류

배경:
  regime_model.py의 make_regime_labels()는 전체 기간(2014~2026)의
  median으로 국면을 라벨링한다. 이는 2014년 시점에서 2026년까지의
  분포를 미리 안다고 가정하는 look-ahead bias다.

해결:
  매월 리밸런싱 시점 t에서, "t까지 관측된 매크로 데이터만"으로
  expanding median을 계산해 그 시점의 국면을 판단한다.
  이렇게 만든 국면 시퀀스를 국면 조건부 백테스트의 입력으로 쓴다.

주의:
  이 모듈은 "국면 조건부 자산배분이 알파를 만드는가"를 검증하기 위한
  것으로, AI 시장 국면 탭(TabPFN/LightGBM 분류기)과는 별개의
  단순/투명한 규칙 기반 판별기다. 검증용이므로 모델 복잡도를
  최소화해 재현성을 우선한다.
"""
import numpy as np
import pandas as pd

REGIMES = ['골디락스', '리플레이션', '스태그플레이션', '디플레이션']

# 매크로 발표 지연 가정 (실거래 가능 시점 보정)
# - GDP 속보치: 분기 마감 후 약 25~30일
# - 금리/환율: 거래일 기준 거의 실시간 (지연 미적용)
GDP_PUBLICATION_LAG_DAYS = 28


def build_pit_macro_features(macro_data):
    """매크로 데이터를 분기별 피처로 변환하되, 각 분기 데이터에
    "그 분기 데이터를 실제로 알 수 있게 된 날짜"를 함께 기록한다.

    Returns:
        DataFrame[gdp_growth, spread, usd_change, available_from]
        index: 분기말(quarter end) 날짜
        available_from: 그 분기 데이터를 실제로 사용 가능해지는 날짜
                         (GDP 발표 지연 반영)
    """
    gdp_df = macro_data.get('gdp')
    rate_df = macro_data.get('rate')
    fx_df = macro_data.get('fx')

    if gdp_df is None or rate_df is None:
        return pd.DataFrame()

    gdp_cols = [c for c in gdp_df.columns if '성장률' in c or 'growth' in c.lower()]
    if not gdp_cols:
        numeric_cols = gdp_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()
        gdp_col = numeric_cols[0]
    else:
        gdp_col = gdp_cols[0]
    gdp_s = gdp_df[gdp_col].dropna()
    gdp_s.index = pd.to_datetime(gdp_s.index)

    long_cols = [c for c in rate_df.columns if '국고10년' in c or '10년' in c]
    short_cols = [c for c in rate_df.columns if '국고1년' in c or '1년' in c]
    if not long_cols or not short_cols:
        return pd.DataFrame()
    rate_long = rate_df[long_cols[0]].dropna()
    rate_short = rate_df[short_cols[0]].dropna()

    rate_long_q = rate_long.resample('QE').last()
    rate_short_q = rate_short.resample('QE').last()
    spread_q = (rate_long_q - rate_short_q).dropna()

    gdp_q = gdp_s.resample('QE').last().ffill()

    usd_change_q = pd.Series(dtype=float)
    if fx_df is not None:
        usd_cols = [c for c in fx_df.columns if '미국' in c or '달러' in c or 'USD' in c]
        if usd_cols:
            fx_s = fx_df[usd_cols[0]].dropna().resample('QE').last()
            usd_change_q = fx_s.pct_change()

    df = pd.DataFrame({
        'gdp_growth': gdp_q,
        'spread': spread_q,
        'usd_change': usd_change_q,
    }).dropna(subset=['gdp_growth', 'spread'])
    df['usd_change'] = df['usd_change'].fillna(0)
    if df.empty:
        return pd.DataFrame()

    # 분기말(quarter end) + 발표 지연 = 실제 사용 가능 시점
    # 금리/환율은 지연 없음으로 가정 (분기말 종가 그대로 사용 가능)
    # GDP만 지연 적용 — 보수적으로 전체 행에 동일 지연 적용
    df['available_from'] = df.index + pd.Timedelta(days=GDP_PUBLICATION_LAG_DAYS)

    return df


def classify_regime_at(features_df, as_of_date, min_quarters=4):
    """as_of_date 시점에서 "그 시점에 알 수 있었던 데이터만" 사용해
    국면을 판정한다 (expanding window median).

    Args:
        features_df: build_pit_macro_features() 결과
        as_of_date: 판단 기준 시점 (리밸런싱 날짜)
        min_quarters: 라벨링에 필요한 최소 분기 수
                      (부족하면 None 반환 → 호출측에서 중립/스킵 처리)

    Returns:
        (regime: str or None, features: dict or None)
    """
    if features_df is None or features_df.empty:
        return None, None

    # as_of_date 시점에 "이미 발표된" 분기 데이터만 사용
    visible = features_df[features_df['available_from'] <= as_of_date]
    if len(visible) < min_quarters:
        return None, None

    growth_med = visible['gdp_growth'].median()
    spread_med = visible['spread'].median()

    latest = visible.iloc[-1]
    growth_up = latest['gdp_growth'] > growth_med
    inflation_up = latest['spread'] < spread_med  # 스프레드 좁아짐 = 인플레/긴축

    if growth_up and not inflation_up:
        regime = '골디락스'
    elif growth_up and inflation_up:
        regime = '리플레이션'
    elif not growth_up and inflation_up:
        regime = '스태그플레이션'
    else:
        regime = '디플레이션'

    feat = dict(
        gdp_growth=float(latest['gdp_growth']),
        spread=float(latest['spread']),
        usd_change=float(latest['usd_change']),
        n_quarters_used=len(visible),
    )
    return regime, feat


def build_pit_regime_series(macro_data, dates, min_quarters=4):
    """여러 리밸런싱 날짜에 대해 PiT 국면을 일괄 계산.

    Args:
        macro_data: data_loader.load_macro() 결과
        dates: 리밸런싱 날짜 리스트/인덱스 (시간순 정렬됨 가정)

    Returns:
        pd.Series, index=dates, value=국면명 (또는 None)
    """
    features_df = build_pit_macro_features(macro_data)
    if features_df.empty:
        return pd.Series(index=dates, dtype=object)

    regimes = []
    for d in dates:
        r, _ = classify_regime_at(features_df, d, min_quarters=min_quarters)
        regimes.append(r)
    return pd.Series(regimes, index=dates, dtype=object)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, '.')
    import data_loader as dl

    print("=" * 60)
    print("regime_pit.py 단위 테스트")
    print("=" * 60)

    macro = dl.load_macro()
    features_df = build_pit_macro_features(macro)
    print(f"\n피처 데이터: {features_df.shape}")
    print(features_df.head())
    print(features_df.tail())

    # 월별 리밸런싱 날짜 생성 (2015~2026)
    dates = pd.date_range('2015-01-31', '2026-06-30', freq='ME')
    regime_series = build_pit_regime_series(macro, dates)

    print(f"\nPiT 국면 시퀀스 ({len(regime_series)}개월):")
    print(regime_series.value_counts())
    print(f"\n결측(데이터 부족) 개월 수: {regime_series.isna().sum()}")
    print(f"\n최근 12개월 국면:")
    print(regime_series.tail(12))
