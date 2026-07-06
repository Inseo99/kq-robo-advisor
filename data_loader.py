"""
data_loader.py — 엑셀 데이터 로더 (parquet 자동 캐싱)
첫 실행: 엑셀 파싱 → parquet 저장 (15~20초)
이후:    parquet에서 직접 로드 (1~2초)
"""
import os, glob, sys
import pandas as pd
import numpy as np

DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 엑셀 파일 패턴 (이름에 부분일치) — 공백/밑줄 둘 다 매칭
_PATTERNS = {
    'stocks':  ['1__', '1.'],         # 종목 정보
    'fin':     ['3__', '3.'],         # 재무 분기
    'macro':   ['주식_이외', '주식 이외'],  # 매크로
}

def _find(patterns):
    """patterns: list of substring 또는 단일 string"""
    if isinstance(patterns, str):
        patterns = [patterns]
    for p in glob.glob(os.path.join(DATA_DIR, '*.xlsx')):
        basename = os.path.basename(p)
        for pat in patterns:
            if pat in basename:
                return p
    return None

def _needs_rebuild(parquet, source):
    """parquet이 없거나 엑셀이 더 최신이면 재생성 필요"""
    if not os.path.exists(parquet): return True
    if not source or not os.path.exists(source): return False
    return os.path.getmtime(parquet) < os.path.getmtime(source)

# ────────────────────────────────────────────────────────────────────────
#  1) 종목 정보 (2014~2026 매년 1월 초)
# ────────────────────────────────────────────────────────────────────────
def load_stocks(markets=('KOSPI','KOSDAQ')):
    """
    반환: {ticker(6자리): {'name','market','sector','industry','first_year'}}
    가장 최근 시트(26) 기준으로 정보 수집 + 2014~2026 전체 종목 포함
    """
    pq = os.path.join(CACHE_DIR, 'stocks.parquet')
    src = _find(_PATTERNS['stocks'])

    if _needs_rebuild(pq, src):
        if src is None:
            return {}
        print('  [data] 종목 정보 파싱 중...')
        xl = pd.ExcelFile(src)
        all_rows = []
        for sheet in xl.sheet_names:
            year = 2000 + int(sheet)
            df = pd.read_excel(src, sheet_name=sheet, dtype={'코드':str})
            df['year'] = year
            all_rows.append(df)
        full = pd.concat(all_rows, ignore_index=True)
        full['ticker'] = full['코드'].str.lstrip('A').str.zfill(6)
        # 가장 최근 정보를 우선 + first_year 보존
        full = full.sort_values('year')
        first_year = full.groupby('ticker')['year'].min()
        latest     = full.groupby('ticker').last()
        latest['first_year'] = first_year
        latest.to_parquet(pq)
    df = pd.read_parquet(pq)

    df = df[df['상장된 시장'].isin(markets)]
    out = {}
    for ticker, row in df.iterrows():
        out[ticker] = dict(
            name      = str(row.get('코드명','')),
            market    = str(row.get('상장된 시장','')),
            sector    = str(row.get('FnGuide Sector','')),
            industry  = str(row.get('FnGuide Industry','')),
            first_year= int(row.get('first_year', 2014)),
        )
    return out

# ────────────────────────────────────────────────────────────────────────
#  2) 재무 데이터 (분기)
# ────────────────────────────────────────────────────────────────────────
_FIN_DF_CACHE = None  # parquet DataFrame 캐시
_FIN_TICKERS_CACHE = None  # 종목 집합 캐시

def load_financials():
    """
    DataFrame 그대로 반환 (lazy). 종목별 조회는 get_latest_fin에서 처리.
    이전엔 dict로 변환했더니 60초 걸렸지만 그냥 DataFrame이면 0.5초.
    """
    global _FIN_DF_CACHE, _FIN_TICKERS_CACHE
    if _FIN_DF_CACHE is not None:
        return _FIN_DF_CACHE

    pq = os.path.join(CACHE_DIR, 'fin.parquet')
    src = _find(_PATTERNS['fin'])

    if _needs_rebuild(pq, src):
        if src is None:
            return None
        print('  [data] 재무 데이터 파싱 중...')
        df = pd.read_excel(src, sheet_name='RAW', dtype={'코드':str})
        df['ticker'] = df['코드'].str.lstrip('A').str.zfill(6)
        import datetime as _dt
        id_cols = ['ticker','코드명','아이템명']
        date_cols = [c for c in df.columns
                     if isinstance(c, (pd.Timestamp, _dt.datetime, _dt.date))]
        long = df[id_cols + date_cols].melt(
            id_vars=id_cols, var_name='date', value_name='value'
        )
        long = long.dropna(subset=['value'])
        long['date'] = pd.to_datetime(long['date'])
        long.to_parquet(pq)

    df = pd.read_parquet(pq)
    # ticker별 빠른 조회를 위해 인덱싱
    df = df.set_index(['ticker','아이템명']).sort_index()
    _FIN_DF_CACHE = df
    _FIN_TICKERS_CACHE = set(df.index.get_level_values(0).unique())
    return df

def get_latest_fin(fin_df, ticker):
    """ticker의 최근 4분기 평균 재무지표"""
    if fin_df is None or ticker not in _FIN_TICKERS_CACHE:
        return {}
    try:
        sub = fin_df.loc[ticker]
    except KeyError:
        return {}
    out = {}
    for key in ['매출액(천원)','영업이익(천원)','당기순이익(천원)','ROE(%)','ROA(%)',
                '자산총계(천원)','부채총계(천원)','자본총계(천원)',
                '영업활동으로인한현금흐름(천원)','시가총액(티커-상장예정주식수 포함)(백만원)',
                '기말발행주식수(보통주)(주)','외국인지분율(%)']:
        if key in sub.index:
            try:
                rows = sub.loc[key]
                if isinstance(rows, pd.Series):
                    rows = rows.to_frame().T
                vals = rows['value'].dropna().tail(4)
                if len(vals):
                    out[key] = float(vals.mean())
            except Exception:
                pass
    return out

def get_latest_fin_asof(fin_df, ticker, asof=None):
    """ticker의 최근 4분기 평균 재무지표 (Point-in-Time).

    `date`는 회계기간 종료일이므로 그대로 쓰면 미래 정보 누수가 생긴다.
    src/kq_tool/data/financial_pit.py의 공시 지연 규칙(Q1~Q3 +45일,
    Q4 +90일)을 적용해 asof 시점에 관측 가능한 행만 사용한다.
    """
    if fin_df is None:
        return {}
    try:
        from kq_tool.data.financial_pit import latest_financials_asof
    except Exception:
        src_dir = os.path.join(os.path.dirname(__file__), 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from kq_tool.data.financial_pit import latest_financials_asof
    return latest_financials_asof(fin_df, ticker, asof=asof)

# ────────────────────────────────────────────────────────────────────────
#  3) 매크로 데이터
# ────────────────────────────────────────────────────────────────────────
def load_macro():
    """
    반환: dict with keys: 'fx','rate','gdp','trade','rp'
    각 값은 datetime index를 가진 DataFrame
    """
    pq = os.path.join(CACHE_DIR, 'macro.parquet')
    src = _find(_PATTERNS['macro'])

    if _needs_rebuild(pq, src):
        if src is None:
            return {}
        print('  [data] 매크로 데이터 파싱 중...')
        result = {}
        sheet_map = {
            'fx':    '2014년 이후 환율 데이터',
            'rate':  '2014년 이후 금리 자료',
            'rp':    '2014년 이후 금리 자료(RP)',
            'gdp':   '2. 2014년 이후 매크로 데이터(GDP 성장률)',
            'trade': '2. 2014년 이후 매크로 데이터(수입수출외환보유)',
            'bond':  '2. 2014년 이후 매크로 데이터(국채발행잔액)',
        }
        for k, sn in sheet_map.items():
            try:
                df = pd.read_excel(src, sheet_name=sn)
                # 첫 컬럼이 날짜 (Unnamed: 0)
                first = df.columns[0]
                # RP 시트는 yyyymmdd 정수
                if k == 'rp':
                    df[first] = pd.to_datetime(df[first].astype(str), format='%Y%m%d', errors='coerce')
                else:
                    df[first] = pd.to_datetime(df[first], errors='coerce')
                df = df.dropna(subset=[first]).set_index(first)
                df.index.name = 'date'
                # 시트별로 컬럼명 prefix
                df.columns = [f'{k}_{c}' for c in df.columns]
                result[k] = df
            except Exception as e:
                print(f'    매크로 시트 {sn} 오류: {e}')

        # 통합 parquet
        combined = []
        for k, df in result.items():
            combined.append(df.reset_index().assign(_src=k))
        if combined:
            big = pd.concat(combined, ignore_index=True)
            big.to_parquet(pq)
    if not os.path.exists(pq):
        return {}
    big = pd.read_parquet(pq)
    out = {}
    for src_name, grp in big.groupby('_src'):
        df = grp.drop(columns=['_src']).set_index('date').sort_index()
        df = df.dropna(how='all', axis=1)  # 다른 시트 컬럼 제거
        out[src_name] = df
    return out

# ────────────────────────────────────────────────────────────────────────
#  4) 매크로 → 시장 국면 분류 (성장×물가/금리 2×2)
# ────────────────────────────────────────────────────────────────────────
def classify_regime(macro):
    """매크로 데이터로 현재 국면 분류"""
    try:
        gdp = macro.get('gdp')
        rate = macro.get('rate')

        # 최근 GDP 성장률
        gdp_col = [c for c in gdp.columns if '성장률' in c][0]
        gdp_recent = gdp[gdp_col].dropna().iloc[-4:].mean() * 100  # 최근 1년 평균
        # 최근 장단기 스프레드 (국고10년 - 국고1년)
        long_col  = [c for c in rate.columns if '국고10년' in c][0]
        short_col = [c for c in rate.columns if '국고1년' in c][0]
        spread = (rate[long_col] - rate[short_col]).dropna().iloc[-30:].mean()

        # 단순 4분면 분류
        growth_up = gdp_recent > 0.5      # 분기 0.5% 이상
        inflation_up = spread < 0.5         # 장단기 역전 = 인플레/긴축

        if growth_up and not inflation_up:   regime = '골디락스'
        elif growth_up and inflation_up:     regime = '리플레이션'
        elif not growth_up and inflation_up: regime = '스태그플레이션'
        else:                                regime = '디플레이션'

        return dict(regime=regime, gdp=round(gdp_recent,2),
                    spread=round(spread,2), growth_up=growth_up, inflation_up=inflation_up)
    except Exception as e:
        return dict(regime='리플레이션', gdp=0, spread=0)

# ────────────────────────────────────────────────────────────────────────
#  5) 주가 데이터 (이미 convert.py로 parquet 캐시 생성됨)
# ────────────────────────────────────────────────────────────────────────
_PRICE_DF_CACHE = {}    # {parquet_path: DataFrame}

# 주가 시트명 → 캐시 파일명 매핑
PRICE_SHEETS = {
    'close_adj':  'price_수정종가.parquet',
    'open_adj':   'price_수정시가.parquet',
    'high_adj':   'price_수정고가.parquet',
    'low_adj':    'price_수정저가.parquet',
    'close':      'price_종가.parquet',
    'open':       'price_시가.parquet',
    'high':       'price_고가.parquet',
    'low':        'price_저가.parquet',
    'volume':     'price_거래량.parquet',
    'value':      'price_거래대금원.parquet',
}
METRIC_SHEETS = {
    'per':         'metric_PER.parquet',
    'pbr':         'metric_PBR.parquet',
    'eps':         'metric_EPS.parquet',
    'bps':         'metric_BPS.parquet',
    'dps':         'metric_DPS.parquet',
    'dividend':    'metric_배당금.parquet',
    'div_yield':   'metric_배당수익률.parquet',
    'target':      'metric_목표주가.parquet',
    'opinion':     'metric_투자의견.parquet',
}

_PRICE_GROUPS = {}  # {filename: {ticker: Series}} 미리 분해해서 저장

def _load_price_parquet(filename):
    """parquet 파일 로드 (한 번만 + 종목별 사전 분해)"""
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        return None
    if path in _PRICE_DF_CACHE:
        return _PRICE_DF_CACHE[path]

    df = pd.read_parquet(path)
    # ticker 컬럼을 6자리 문자열로 정규화 (혼합 타입 방지)
    df['ticker'] = df['ticker'].astype(str).str.lstrip('A').str.zfill(6)
    df['date'] = pd.to_datetime(df['date'])

    # 종목별로 미리 분해해서 저장 (조회 시점에 인덱싱 불필요)
    print(f'  [data] {filename} 종목별 분해 중...')
    groups = {}
    for ticker, grp in df.groupby('ticker'):
        s = grp.set_index('date')['value'].sort_index()
        # 중복 날짜 제거 (마지막 값 유지)
        s = s[~s.index.duplicated(keep='last')]
        groups[ticker] = s
    _PRICE_GROUPS[path] = groups
    _PRICE_DF_CACHE[path] = df  # 호환성을 위해 유지
    print(f'  [data] {filename} OK {len(groups)} 종목 인덱싱 완료')
    return df

def get_price_series(ticker, kind='close_adj'):
    """종목의 시계열 가격 데이터 반환"""
    fname = PRICE_SHEETS.get(kind)
    if not fname:
        return None
    path = os.path.join(CACHE_DIR, fname)
    # 캐시 보장
    if path not in _PRICE_GROUPS:
        _load_price_parquet(fname)
    if path not in _PRICE_GROUPS:
        return None
    # ticker 정규화 (6자리 zfill)
    ticker = str(ticker).lstrip('A').zfill(6)
    s = _PRICE_GROUPS[path].get(ticker)
    if s is None:
        return None
    s = s.copy()
    s.name = kind
    return s

def get_ohlcv_df(ticker, adjusted=False):
    """
    종목의 OHLCV DataFrame 반환
    adjusted=False (기본): 실제 종가 (분할/배당 반영 안 됨, 현재가와 일치)
    adjusted=True: 수정 종가 (장기 추세 분석용)
    """
    suffix = '_adj' if adjusted else ''
    out = {}
    for col_name, kind in [('Open',f'open{suffix}'),('High',f'high{suffix}'),
                            ('Low',f'low{suffix}'),('Close',f'close{suffix}'),
                            ('Volume','volume')]:
        s = get_price_series(ticker, kind)
        if s is not None:
            out[col_name] = s
    if not out:
        return None
    df = pd.DataFrame(out).dropna(how='all')
    return df if len(df) > 0 else None

def get_metric_value(ticker, metric='per', latest=True):
    """
    종목의 메트릭 값 (PER/PBR/EPS/BPS/DPS 등)
    latest=True: 최신 값 1개, False: 전체 시계열
    """
    fname = METRIC_SHEETS.get(metric)
    if not fname:
        return None
    path = os.path.join(CACHE_DIR, fname)
    if path not in _PRICE_GROUPS:
        _load_price_parquet(fname)
    if path not in _PRICE_GROUPS:
        return None
    ticker = str(ticker).lstrip('A').zfill(6)
    s = _PRICE_GROUPS[path].get(ticker)
    if s is None:
        return None
    if latest:
        recent = s.dropna()
        return float(recent.iloc[-1]) if len(recent) else None
    return s.copy()

def list_available_tickers(kind='close_adj'):
    """주가 데이터가 있는 모든 종목 코드 목록"""
    fname = PRICE_SHEETS.get(kind)
    if not fname:
        return []
    path = os.path.join(CACHE_DIR, fname)
    if path not in _PRICE_GROUPS:
        _load_price_parquet(fname)
    return sorted(_PRICE_GROUPS.get(path, {}).keys())

# ────────────────────────────────────────────────────────────────────────
#  진단 도구
# ────────────────────────────────────────────────────────────────────────
def diagnose():
    """현재 데이터 가용성 확인"""
    info = {}
    s = load_stocks()
    info['stocks_total']  = len(s)
    info['stocks_kospi']  = sum(1 for v in s.values() if v['market']=='KOSPI')
    info['stocks_kosdaq'] = sum(1 for v in s.values() if v['market']=='KOSDAQ')

    f = load_financials()
    info['fin_tickers']   = len(f)

    m = load_macro()
    info['macro_sheets']  = list(m.keys())

    r = classify_regime(m)
    info['regime']        = r

    return info

if __name__ == '__main__':
    print('=== 데이터 로더 진단 ===')
    info = diagnose()
    for k, v in info.items():
        print(f'  {k}: {v}')
