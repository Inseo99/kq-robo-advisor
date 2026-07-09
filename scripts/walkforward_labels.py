r"""scripts/walkforward_labels.py — 국면 라벨 워크포워드 재생성 (IS/OOS 철학 적용)
원리: 매월 t마다 매크로 데이터를 (t−창, t]로 절단해 make_regime_labels에 입력하고
      t월 라벨만 취한다. 함수 내부가 무엇이든(고정 규칙/전기간 통계) 미래 데이터가
      입력에 없으므로 look-ahead가 원천 차단된다.

산출물 (data/analysis_outputs/):
  labels_walkforward_10y.csv   ← 주 라벨 (120개월 롤링)
  labels_walkforward_5y.csv    ← 민감도 (60개월, 교수님 예시값)
  labels_walkforward_exp.csv   ← 민감도 (확장창, 최소 60개월)
  window_agreement.csv         ← 월별 3창 라벨 + 일치 여부
화면 출력: 창별 상호 일치율, 위기 구간 채점(2008·2020·2022), 구간 요약(10y)

주의: data/regime/labels.csv를 자동으로 덮지 않는다 — 검토 후 명시적 채택:
  Copy-Item data\regime\labels.csv data\regime\labels_fullsample_backup.csv
  Copy-Item data\analysis_outputs\labels_walkforward_10y.csv data\regime\labels.csv
  (채택 시 하류 연쇄: 타임라인·조건부 성과 재실행, ERC 재산출 검토, 서버 재시작)
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT_DIR = os.path.join(ROOT, "data", "analysis_outputs")

WINDOWS = [("10y", 120), ("5y", 60), ("exp", None)]   # None = 확장창(최소 60개월)
MIN_EXPANDING = 60
LABEL_END = "2025-12-31"   # 홀드아웃: 2026년은 라벨 생성 제외 — 실운영 판정과의 비교군으로 보존
INFLATION_AXIS = "cpi"     # "spread"(V1: 10y−3y 프록시) | "cpi"(V2: CPI YoY 실측)
SUFFIX = "" if INFLATION_AXIS == "spread" else f"_{INFLATION_AXIS}"


MACRO_DIR = os.path.join(ROOT, "data", "macro")


def _read_macro_csv(name):
    """data/macro/<name>.csv → 날짜 인덱스 Series (첫 숫자 컬럼)."""
    path = os.path.join(MACRO_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return num.iloc[:, 0] if num.shape[1] else None


def _load_macro_from_csv():
    """data/macro CSV들을 make_regime_labels가 기대하는 dict로 조립.
    단기금리: 1년물 부재 → 3년물 사용 (팀 관행 yield_spread_10y_3y와 동일).
    컬럼명은 함수의 매칭 규칙('10년'/'1년'/'USD')을 만족하도록 명명."""
    gdp = _read_macro_csv("gdp_qoq")
    t10 = _read_macro_csv("treasury_10y")
    t3 = _read_macro_csv("treasury_3y")
    fx = _read_macro_csv("usdkrw")
    if gdp is None or t10 is None or t3 is None:
        return None
    macro = {
        "gdp": pd.DataFrame({"GDP성장률": gdp}),
        "rate": pd.DataFrame({"국고10년": t10, "국고3년(단기, 1년물 대용)": t3}),
    }
    if fx is not None:
        macro["fx"] = pd.DataFrame({"USD_KRW": fx})
    exp = _read_macro_csv("exports_yoy")
    if exp is not None:
        macro["trade"] = pd.DataFrame({"수출YoY": exp})
    cpi = _read_macro_csv("cpi_yoy")
    if cpi is not None:
        macro["cpi"] = pd.DataFrame({"CPI_YoY": cpi})
    print("[입력] data/macro CSV 조립: gdp_qoq, treasury_10y, treasury_3y"
          + (", usdkrw" if fx is not None else "")
          + (", exports_yoy" if exp is not None else "")
          + (", cpi_yoy" if cpi is not None else "")
          + " · 스프레드 = 10년 − 3년 (팀 관행)")
    if INFLATION_AXIS == "cpi" and cpi is None:
        raise SystemExit("[stop] V2(CPI 축) 지정됐으나 data/macro/cpi_yoy.csv 없음")
    return macro


def _make_labels_v2(macro_data):
    """V2 라벨 룰 — V1(make_regime_labels docstring)과 동일 구조, 물가 축만 교체:
      성장↑ if gdp_growth > median (V1과 동일)
      물가↑ if CPI_YoY > median   (V1: spread < median 프록시 → 실측 교체)
    분기 리샘플·median 상대 기준·4국면 매핑 전부 V1 유지. 입력 절단은 드라이버가 수행."""
    if not macro_data or "gdp" not in macro_data or "cpi" not in macro_data:
        return pd.DataFrame()
    gdp_s = macro_data["gdp"].iloc[:, 0].dropna()
    cpi_s = macro_data["cpi"].iloc[:, 0].dropna()
    gdp_q = gdp_s.resample("QE").last().ffill()
    cpi_q = cpi_s.resample("QE").last().ffill()
    df = pd.DataFrame({"gdp_growth": gdp_q, "cpi_yoy": cpi_q}).dropna()
    if len(df) < 4:
        return pd.DataFrame()
    hi_g = df["gdp_growth"] > df["gdp_growth"].median()
    hi_p = df["cpi_yoy"] > df["cpi_yoy"].median()
    lab = pd.Series("골디락스", index=df.index)
    lab[hi_g & hi_p] = "리플레이션"
    lab[~hi_g & hi_p] = "스태그플레이션"
    lab[~hi_g & ~hi_p] = "디플레이션"
    df["regime"] = lab
    return df


def _load_pipeline():
    try:
        from regime_model import make_regime_labels
    except ImportError as e:
        raise SystemExit(f"[stop] regime_model import 실패: {e} — 루트에서 실행하세요")
    macro = _load_macro_from_csv()
    if macro is None:
        try:
            from data_loader import load_macro
            macro = load_macro()
        except Exception:
            macro = None
    if not macro:
        raise SystemExit("[stop] 매크로 데이터 없음 — data/macro/gdp_qoq.csv 등 확인")
    return make_regime_labels, macro


def _slice_macro(macro, t0, t1):
    """dict/DataFrame 양쪽 지원 — 각 프레임을 (t0, t1] 로 절단."""
    if isinstance(macro, dict):
        out = {}
        for k, v in macro.items():
            idx = pd.to_datetime(v.index)
            mask = (idx > t0) & (idx <= t1) if t0 is not None else (idx <= t1)
            out[k] = v.loc[mask]
        return out
    idx = pd.to_datetime(macro.index)
    mask = (idx > t0) & (idx <= t1) if t0 is not None else (idx <= t1)
    return macro.loc[mask]


def _extract_label(result, t):
    """make_regime_labels 반환(Series/DataFrame/tuple)에서 t월 라벨 추출."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, pd.DataFrame):
        col = "regime" if "regime" in result.columns else next(
            (c for c in result.columns if result[c].dtype == object), result.columns[-1])
        s = result[col]
    else:
        s = result
    s = s.dropna()
    if s.empty:
        return None
    s.index = pd.to_datetime(s.index)
    upto = s[s.index <= t + pd.offsets.MonthEnd(0)]
    return str(upto.iloc[-1]) if len(upto) else None


def _macro_month_range(macro):
    frames = macro.values() if isinstance(macro, dict) else [macro]
    starts, ends = [], []
    for v in frames:
        idx = pd.to_datetime(v.index)
        if len(idx):
            starts.append(idx.min()); ends.append(idx.max())
    return max(starts), min(ends)


def run():
    make_regime_labels, macro = _load_pipeline()
    data_start, data_end = _macro_month_range(macro)
    cutoff = min(data_end, pd.Timestamp(LABEL_END))
    months = pd.date_range(data_start, cutoff, freq="ME")
    if cutoff < data_end:
        print(f"[홀드아웃] 라벨 생성은 {cutoff:%Y-%m}까지 — 이후({cutoff + pd.offsets.MonthBegin(1):%Y-%m}~{data_end:%Y-%m})는 "
              f"실운영 판정과의 비교군으로 보존")
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for name, win in WINDOWS:
        labels = {}
        min_hist = win if win is not None else MIN_EXPANDING
        eligible = [t for t in months if t >= data_start + pd.DateOffset(months=min_hist)]
        print(f"\n[{name}] {len(eligible)}개월 생성 중 (시작 {eligible[0]:%Y-%m})...")
        for i, t in enumerate(eligible):
            t0 = (t - pd.DateOffset(months=win)) if win is not None else None
            try:
                sliced = _slice_macro(macro, t0, t)
                res = _make_labels_v2(sliced) if INFLATION_AXIS == "cpi" else make_regime_labels(sliced)
                lab = _extract_label(res, t)
            except Exception as e:
                lab = None
                if i < 3:
                    print(f"  경고 {t:%Y-%m}: {e}")
            if lab:
                labels[t] = lab
            if (i + 1) % 24 == 0:
                print(f"  ... {t:%Y-%m} ({i+1}/{len(eligible)})")
        s = pd.Series(labels, name="regime")
        s.index.name = "month"
        path = os.path.join(OUT_DIR, f"labels_walkforward_{name}{SUFFIX}.csv")
        s.to_csv(path, encoding="utf-8-sig", date_format="%Y-%m-%d")
        results[name] = s
        print(f"[{name}] {len(s)}개월 → {path}")

    # 일치율 표
    merged = pd.DataFrame(results).dropna()
    merged["agree_all"] = merged.nunique(axis=1) == 1
    merged.to_csv(os.path.join(OUT_DIR, f"window_agreement{SUFFIX}.csv"),
                  encoding="utf-8-sig", date_format="%Y-%m-%d")
    print("\n=== 창 선택 민감도 (공통 구간 {}~{}) ===".format(
        merged.index[0].strftime("%Y-%m"), merged.index[-1].strftime("%Y-%m")))
    names = [n for n, _ in WINDOWS]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            agr = (merged[names[i]] == merged[names[j]]).mean()
            print(f"  {names[i]} vs {names[j]}: 일치율 {agr:.1%}")
    print(f"  3창 전체 일치: {merged['agree_all'].mean():.1%}")

    # 위기 구간 채점
    print("\n=== 위기 구간 채점 (기대: 디플레/스태그 계열) ===")
    for period, desc in [("2008-09", "금융위기"), ("2020-0", "코로나"), ("2022", "긴축 하락장")]:
        for name in names:
            s = results[name]
            hit = s[s.index.strftime("%Y-%m").str.startswith(period)]
            if len(hit):
                print(f"  {desc} [{name}]: " + ", ".join(
                    f"{d:%Y-%m}={v}" for d, v in list(hit.items())[:6]))

    # 10y 구간 요약
    main = results["10y"]
    print("\n=== 주 라벨(10y) 구간 요약 ===")
    cur, start = None, None
    for dt, lab in main.items():
        if lab != cur:
            if cur is not None:
                print(f"  {start:%Y-%m} ~ {prev:%Y-%m}  {cur}")
            cur, start = lab, dt
        prev = dt
    print(f"  {start:%Y-%m} ~ {prev:%Y-%m}  {cur}")
    v1_path = os.path.join(OUT_DIR, "labels_walkforward_10y.csv")
    if SUFFIX and os.path.exists(v1_path):
        v1 = pd.read_csv(v1_path, index_col=0, parse_dates=True)["regime"]
        v2 = results["10y"]
        both = pd.DataFrame({"V1_spread": v1, "V2_cpi": v2}).dropna()
        print(f"\n=== V1(스프레드) vs V2(CPI) — 사전 등록 채점 (전체 일치율 {(both['V1_spread']==both['V2_cpi']).mean():.1%}) ===")
        checks = [("2020-03", "코로나 전환", "디플레이션 유지가 합격 (CPI 지연 시험대)"),
                  ("2021-09", "인플레기", "리플레/스태그로 교정되면 V2 승"),
                  ("2021-10", "인플레기", "동일"),
                  ("2022-03", "긴축 전환", "스태그 유지가 합격"),
                  ("2018-07", "무역분쟁", "위험 국면 유지가 합격"),
                  ("2020-12", "회복 후반", "리플레 전환이면 V2 승")]
        for ym, desc, crit in checks:
            row = both[both.index.strftime("%Y-%m") == ym]
            if len(row):
                print(f"  {ym} {desc}: V1={row['V1_spread'].iloc[0]} → V2={row['V2_cpi'].iloc[0]}  [{crit}]")
    print("\n[다음 단계] 위 채점·일치율 검토 후, 채택하려면 파일 상단 주석의 복사 명령 실행")


if __name__ == "__main__":
    run()
