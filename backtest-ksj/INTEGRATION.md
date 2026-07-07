# backtest-ksj 통합 메모

`backtest-ksj`는 KQ Quant 앱의 실시간 전략검증 탭을 바로 대체하는 코드가 아니라,
발표/검증용 독립 백테스트 엔진이다.

## 프로젝트 안에서의 역할

- 앱 본체: 단일 종목 로보신호, 스크리너, 추천 포트폴리오, 화면용 전략검증을 제공한다.
- `backtest-ksj`: 사전 고정된 12개 모멘텀/위험회피 규칙을 동일 조건으로 검증하고,
  CSV/엑셀/마크다운 보고서를 생성한다.

이렇게 분리하면 앱 화면의 실시간 기능과 발표용 검증 엔진이 서로를 깨지 않고,
동시에 같은 프로젝트 안에서 재현 가능한 산출물을 관리할 수 있다.

## 입력

- 공통 가격/재무 캐시: `../data/cache/`
- 외부 위험회피 지표: `data/market_2000_2026.csv`, `data/liquidity_2000_2026.csv`
- 자동 재생성 캐시: `data/*.parquet`

`data/*.parquet`는 `data.py`가 실행 중 재생성하는 파생 캐시이므로 git에 올리지 않는다.

## 출력

- `results/metrics_all.csv`: 전략 x 폴드 x IS/OOS 전체 지표
- `results/summary_full_oos.csv`: 전체 OOS 요약
- `results/period_records_*.csv`: 전략별 거래/비용/수익 감사 추적
- `results/backtest_summary.xlsx`: 발표용 요약 워크북
- `results/report.md`: 마크다운 보고서

## 실행

```powershell
cd C:\Users\paenco0313\Downloads\kq_tool\backtest-ksj
python run.py
python make_summary_workbook.py
```

최초 실행은 가격 패널과 PIT 시총 캐시 생성 때문에 오래 걸릴 수 있다.
두 번째 실행부터는 `data/*.parquet` 캐시를 사용해 빨라진다.

## 발표에서의 표현

안전한 표현:

> 앱 화면은 실시간 RA 의사결정 도구이고, `backtest-ksj`는 별도 검증 엔진입니다.
> 사전 고정 규칙 12개를 동일 유니버스, 동일 비용, 동일 벤치마크 조건에서
> 4개 구간으로 비교해 전략의 강건성을 점검했습니다.

피해야 할 표현:

> 앱의 추천 포트폴리오가 `backtest-ksj` 결과를 그대로 사용한다.

현재 구조에서는 `backtest-ksj` 결과를 앱 추천에 직접 대입하지 않는다. 결과가 유의하면
차후 후보 전략으로 승격하는 검증 레이어로 사용한다.
