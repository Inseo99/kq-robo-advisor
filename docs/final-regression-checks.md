# Final Regression Checks

이 문서는 KQ Quant Tool을 팀원에게 공유하거나 데모하기 전에 로컬에서 확인할
최소 회귀검증 절차입니다.

## 1. 기본 실행 환경

Windows PowerShell에서 프로젝트 폴더로 이동합니다.

```powershell
cd C:\Users\paenco0313\Downloads\kq_tool
```

필요 패키지를 설치합니다.

```powershell
python -m pip install -r requirements.txt
```

Python 실행이 안 되면 먼저 `docs/TEAM_RUN_GUIDE.md`의 Python 설치 안내를
확인합니다.

## 2. 자동 회귀검증

가장 쉬운 방법은 아래 파일을 더블클릭하는 것입니다.

```text
RUN_REGRESSION_CHECKS.bat
```

또는 PowerShell에서 직접 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_REGRESSION_CHECKS.ps1
```

자동 검증은 아래 항목을 확인합니다.

- `pytest`가 없으면 설치 시도
- `server.py`, 핵심 검증 스크립트, API smoke 스크립트 문법 검사
- `src/kq_tool` 전체 컴파일 검사
- `tests/unit` 단위 테스트 실행
- 문서에 적힌 `tests/unit` 개수가 실제 pytest 수집 개수와 일치하는지 확인

## 3. 서버 실행 확인

앱 실행은 아래 파일을 더블클릭합니다.

```text
RUN_KQ_TOOL.bat
```

정상 실행 후 Chrome에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8888/
```

## 4. API Smoke Check

서버가 켜진 상태에서 새 PowerShell을 열고 실행합니다.

```powershell
python tests\smoke_api.py
```

기본 smoke는 `/api/ping`, `/api/health`만 확인합니다. 단일종목 API까지
확인하려면 아래처럼 실행합니다.

```powershell
python tests\smoke_api.py --include-stock --timeout 120
```

주요 API 흐름(스크리너, 전략검증, 추천 포트폴리오)까지 확인하려면 아래처럼
실행합니다. 서버가 데이터 캐시를 읽어야 하므로 처음 실행은 시간이 더 걸릴 수
있습니다.

```powershell
python tests\smoke_api.py --include-stock --include-core --timeout 180
```

`--include-core`는 스크리너의 `S2모멘텀`, 전략검증의 `quant`, `quant_s2`,
`quant_compare`, 추천 포트폴리오 응답까지 확인합니다.

## 5. 비용 반영 Alpha Decay Smoke

비용/슬리피지 반영 로직까지 빠르게 확인하려면 아래 명령을 실행합니다.

```powershell
python tests\validation_signal_quality_alpha_decay.py --universe-source etf --top 3 --n 1 --min-quality 0.7 --cost-bps 10 --slippage-bps 5 --output tests\signal_quality_cost_smoke.npz
```

이 검증은 전체 연구 검증이 아니라, 비용 차감 경로가 정상 작동하는지 확인하는
가벼운 smoke 용도입니다.

## 통과 기준

로컬 릴리즈 전 최소 기준은 다음과 같습니다.

- `RUN_REGRESSION_CHECKS.bat` 통과
- `RUN_KQ_TOOL.bat`으로 서버 실행
- `python tests\smoke_api.py` 통과
- 데모 전 `python tests\smoke_api.py --include-stock --include-core --timeout 180` 통과
  (`quant_compare` 3자 비교 응답 포함)
- 주요 화면에서 로보신호, 스크리너, 전략검증, 추천 포트폴리오 탭이 열림
