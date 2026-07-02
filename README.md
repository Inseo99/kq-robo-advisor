# KQ Quant Tool

한국 주식시장 데이터를 기반으로 종목 분석, 퀀트 스크리닝, 시장 국면 분석,
자산배분 백테스트, 추천 포트폴리오, Alpha Decay 유효기간을 함께 검증하는
로보어드바이저 실험 프로젝트입니다.

## 현재 핵심 기능

- 단일 종목 로보신호: RSI, MACD, 이동평균, 볼린저밴드, ATR 기반 목표가/손절가
- 퀀트 스크리너: KOSPI 시총 상위 종목 중심의 밸류/퀄리티/모멘텀/S2모멘텀 필터
- 시장 국면 분석: 매크로 데이터 기반 4국면 분류와 확률 표시
- 전략 검증: 정적/동적 포트폴리오와 KOSPI 벤치마크 비교
- 추천 포트폴리오: 기본 자산배분, 국면 참고, 자산별 로보신호, 유효기간 통합
- Alpha Decay: 신호 발생 후 어느 기간까지 재점검 가치가 있는지 IS/OOS로 검증

## 실행

가장 쉬운 방법은 프로젝트 폴더에서 아래 파일을 더블클릭하는 것입니다.

```text
RUN_KQ_TOOL.bat
```

이 파일은 Python을 찾고, 필요한 패키지를 설치한 뒤 서버를 실행합니다.
Chrome이 설치되어 있으면 Chrome으로 `http://127.0.0.1:8888/`을 엽니다.

팀원에게 전달할 때는 [docs/TEAM_RUN_GUIDE.md](docs/TEAM_RUN_GUIDE.md)를 같이 보내면 됩니다.

수동 실행이 필요하면 PowerShell에서:

```powershell
cd C:\Users\"본인주소"\Downloads\kq_tool
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
python -m kq_tool
```

WSL/bash:

```bash
cd /mnt/c/Users/"본인주소"/Downloads/kq_tool
python3 -m pip install -r requirements.txt
PYTHONPATH="$PWD/src:$PYTHONPATH" python3 -m kq_tool
```

기존 호환 실행도 계속 지원합니다.

```powershell
python server.py
```

서버가 실행되면 Chrome에서 아래 주소를 여세요.

```text
http://127.0.0.1:8888/
```

## 시황/애널리스트 리포트 수집

AI 시장 국면 탭은 `data/market_reports/latest.md`와 `data/market_reports/user_reports.md`를 읽어 정량 국면의 보조 근거로 사용합니다.
공개 RSS나 웹페이지 URL은 `data/report_sources.json`에 직접 등록합니다. 유료/로그인/무단 복제 페이지는 넣지 마세요.

샘플 설정 파일 생성:

```powershell
python collect_market_reports.py --init-sample
```

`data/report_sources.json`에서 사용할 소스의 `enabled`를 `true`로 바꾼 뒤 수집:

```powershell
python collect_market_reports.py
```

수집 결과는 `data/market_reports/latest.md`에 저장되고, `/api/regime_ai`와 AI 시장 국면 탭에 자동 반영됩니다.
브라우저의 AI 시장 국면 탭에서도 보유한 `.txt`/`.md` 보고서 파일을 선택하거나 본문을 붙여넣어 등록할 수 있습니다. 직접 등록한 보고서는 `data/market_reports/user_reports.md`에 누적 저장되며, 보고서 목록은 탭 안에서 스크롤로 확인하고 URL이 있으면 원문 링크로 바로 열 수 있습니다.

현재 서버는 Microsoft Edge를 자동으로 띄우지 않도록 조정되어 있습니다.
자동 브라우저 열기가 필요할 때만 `KQ_AUTO_OPEN_BROWSER=1`을 설정합니다.
TabPFN은 기본 비활성화되어 있고, 명시적으로 테스트할 때만
`KQ_ENABLE_TABPFN=1`을 설정합니다. Windows에서 Python 3.14를 쓰는 경우
`hmmlearn`은 wheel 미지원으로 자동 설치를 건너뛰며, 앱은 내장 국면 전환
fallback으로 계속 실행됩니다. 국면 모델 전체 의존성을 가장 안정적으로 쓰려면
Python 3.10~3.13을 권장합니다.

## 검증 스크립트

로컬 릴리즈 전 기본 회귀검증은 아래 파일을 더블클릭합니다.

```text
RUN_REGRESSION_CHECKS.bat
```

또는 PowerShell에서:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_REGRESSION_CHECKS.ps1
```

세부 체크리스트는 [docs/final-regression-checks.md](docs/final-regression-checks.md)를 기준으로 합니다.
팀원 공유/데모 전 요약은 [docs/release-readiness.md](docs/release-readiness.md)를 확인하면 됩니다.

연구/검증용 주요 명령은 아래와 같습니다.

```powershell
python -m pytest tests\unit -q
python tests\smoke_api.py
python tests\validation_signal_quality_alpha_decay.py --universe-source etf --top 3 --n 1 --min-quality 0.7 --cost-bps 10 --slippage-bps 5 --output tests\signal_quality_cost_smoke.npz
python tests\validation_regime_alpha_decay.py --top 50 --n 200 --mode quality --min-quality 0.6 --signal-side buy
python tests\export_dsr_inputs.py --start 2014-06-26
```

## 현재 구조

기존 `server.py`와 `index.html`은 계속 실행 가능한 상태로 유지하고,
핵심 로직은 `src/kq_tool` 아래로 점진적으로 분리했습니다.

```text
src/kq_tool/
  config.py
  analyzer/
    indicators.py
    alpha_decay.py
    signal_quality.py
    chart.py
    robo.py
    stock_analyzer.py
    dcf.py
  api/
    dispatcher.py
    health.py
    http_response.py
    params.py
    runtime.py
    services.py
    serialization.py
    static_files.py
  backtest/
    engine.py
    selector.py
    preparation.py
    orchestrator.py
    metrics.py
    costs.py
  data/
    cache.py
    price.py
    fundamental.py
    marketcap.py
    universe.py
    repositories.py
  portfolio/
    allocation.py
    recommender.py
    risk_based.py
    validity.py
    weights.py
  regime/
    classifier.py
    macro_builder.py
    response.py
  screener/
    engine.py
    strategies.py
  validation/
    costs.py
    factor_analysis.py
    regime_alpha_decay.py
    reporting.py
    segments.py
  utils/
    tickers.py
```

이 방식은 기능을 한 번에 갈아엎지 않고, 기존 앱을 살린 상태에서
검증 가능한 도메인 모듈을 분리하는 전략입니다.
ETF 목록, 10개 자산배분 전략, 위험기반 전략 키, 스크리너 종목 수 상한, 수익률 가정,
Alpha Decay 신호 방향, 로보 점수 가중치/라벨, 로보 매수/매도 임계값은 `src/kq_tool/config.py`를 기준으로 관리하며,
추천 포트폴리오 구성과 국면별 목표비중은 `src/kq_tool/portfolio/recommender.py`를
기준으로 관리합니다. 추천탭의 국면 타깃은 현재 국면 확률 70%와 다음 분기 국면 확률 30%를
가중평균해 단일 국면 과신을 줄입니다. `server.py`는 이 설정들을 재사용합니다.

## 현재 검증 상태

- 단위 테스트: `tests/unit` 기준 333개 통과
- API smoke: `/api/ping`, `/api/health` 기본 확인 가능
- Health endpoint: 데이터 파일, 모듈 import, 국면 모델, universe/ETF 준비 상태 조립을 `src/kq_tool/api/health.py`로 분리
- 공통 유틸: 외부 API 재시도/backoff helper를 `src/kq_tool/utils/retry.py`로 분리
- API 응답: JSON-safe serialization helper를 `src/kq_tool/api/serialization.py`로 분리
- API 응답: CORS/no-cache/content headers와 JSON body encoding을 `src/kq_tool/api/http_response.py`로 분리
- API 파라미터: 종목/전략검증 query parsing을 `src/kq_tool/api/params.py`로 분리
- API 실행: 자동 브라우저 열기와 Chrome 우선 실행 정책을 `src/kq_tool/api/runtime.py`로 분리
- API 서비스: dispatcher가 기대하는 서비스 registry 계약을 `src/kq_tool/api/services.py`로 분리
- API 정적파일: index.html 읽기/경로 검증 helper를 `src/kq_tool/api/static_files.py`로 분리
- 국면 API: 매크로 국면 payload와 `/api/regime_ai` payload 조립을 `src/kq_tool/regime`로 분리
- 국면 리포트: `/api/market_report` 조회/등록 API와 AI 시장 국면 탭의 스크롤형 보고서 목록/직접 등록 UI 추가
- 학술 검증: Fama-French 3/5 팩터 회귀 helper를 추가해 포트폴리오 alpha와 t-stat을 표준 팩터 기준으로 검증 가능
- 학술 검증 실행: `python tests\validation_factor_regression.py --model ff5 --top-n 500`
- 학술 검증 입력: `tests\export_dsr_inputs.py`로 DSR/국면별 성과분해 공용 입력 3개 CSV 생성 가능
- 실행 의존성: 새 환경에서도 LightGBM을 쓰며, `hmmlearn`은 Python 3.10~3.13에서만 설치하도록 조건부 처리
- 차트: 일/월/년 봉과 조회 기간을 분리하고 전체 기간 확대/이동/저장 지원
- 가격 기간: 차트/전략 기간 코드는 `src/kq_tool/data/price.py`의 공용 helper 기준으로 정리
- 가격 데이터: yfinance 실시간 현재가 필드 선택 정책을 `src/kq_tool/data/price.py` helper로 분리
- 가격 데이터: Yahoo Finance 세션 워밍업/이력 유효성 판단을 `src/kq_tool/data/price.py` helper로 분리
- 가격 데이터: 기간별 최소 가격행 수 판단을 `src/kq_tool/data/price.py` helper로 분리
- 가격 데이터: yfinance 다운로드 프레임 정리/검증을 `src/kq_tool/data/price.py` helper로 분리
- 가격 데이터: 실시간 현재가/출처/기준일 context 판단을 `src/kq_tool/data/price.py` helper로 분리
- 기술적 지표: legacy `server.py`가 `src/kq_tool/analyzer/indicators.py`를 재사용하는지 회귀 테스트 추가
- 재무 데이터: yfinance 재무 fallback 사용 가능성 판단을 `src/kq_tool/data/fundamental.py` helper로 분리
- 자산배분: 기존 9개 전략에 정적 60/40을 추가해 총 10개 전략 비교
- 자산배분 계산: ETF 동적 전략 생성과 전략별 성과/위험기여도 결과 생성은 `src/kq_tool/portfolio/allocation.py`로 분리
- 추천 포트폴리오: 종목 분석 결과를 추천용 신호 요약으로 바꾸는 helper를 `src/kq_tool/portfolio/recommender.py`로 분리
- 스크리너: 엑셀 종가 기준일 계산 helper를 `src/kq_tool/screener/engine.py`로 분리
- 스크리너: 병렬 실행 전 parquet 캐시 워밍업 정책을 `src/kq_tool/screener/engine.py`로 분리
- 전략검증: 거래비용/슬리피지 bps 입력과 비용 요약 반영
- 전략검증: 퀀트(최근 모멘텀), 퀀트(S2 12-1 모멘텀), KOSPI 3자 비교를 `quant_compare` 백엔드 응답으로 제공
- 전략검증: 기간 정책, 가격 프레임 정리, 벤치마크 기간 필터를 `src/kq_tool/backtest/preparation.py`로 분리
- 전략검증: 로보 전략 지표 사전계산은 `src/kq_tool/backtest/selector.py` helper로 분리
- Alpha Decay: 신호품질 필터, 대형/중소형, ETF/개별주, 매수/매도, 가격제한폭 포함 여부 분리 검증 경로 추가
- Alpha Decay: PiT 매크로 국면별 OOS 신호 edge와 같은 국면 내 무작위 날짜 placebo 비교 경로 추가
- 추천 포트폴리오: `data/validation/regime_alpha_decay_summary.csv`가 있고 placebo 반복 수가 충분하면 현재 국면의 Alpha Decay 검증 결과로 로보 신호 틸트 강도를 조정
- 운영 기준: 비용 차감 후 채택 조건은 `docs/operating-thresholds-after-costs.md` 기준

## 현재 해석 기준

검증 결과상 시장 국면은 아직 독립적인 초과수익 엔진이라기보다,
자산배분의 위험 상태와 재점검 트리거로 쓰는 편이 더 타당합니다.

Alpha Decay는 특히 매수형 기술 신호의 유효기간/재점검 기간을 정하는 데
근거가 있으며, 매도형 신호는 별도 검증이 더 필요합니다.

운영 기준과 비용 반영 후 채택 조건은 아래 문서를 기준으로 합니다.

- [docs/operating-thresholds-after-costs.md](docs/operating-thresholds-after-costs.md)
- [docs/signal_quality_segmentation.md](docs/signal_quality_segmentation.md)


























