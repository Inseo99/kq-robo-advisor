# 팀원 실행 안내

## 가장 쉬운 실행 방법

프로젝트 폴더에서 아래 파일을 더블클릭합니다.

```text
RUN_KQ_TOOL.bat
```

실행되면 검은 창 또는 PowerShell 창이 열리고, 잠시 후 브라우저에서 아래 주소가 열립니다.

```text
http://127.0.0.1:8888/
```

도구를 사용하는 동안 실행 창을 닫으면 안 됩니다. 창을 닫으면 서버도 같이 꺼집니다.

## Python이 없다고 나오는 경우

컴퓨터에 Python이 설치되어 있지 않은 상태입니다.

1. https://www.python.org/downloads/ 에서 Python 3.10 이상을 설치합니다.
2. 설치 첫 화면에서 `Add python.exe to PATH`를 반드시 체크합니다.
3. 설치가 끝나면 프로젝트 폴더에서 `RUN_KQ_TOOL.bat`를 다시 더블클릭합니다.

Anaconda를 쓰는 팀원은 Anaconda Prompt에서 아래처럼 실행해도 됩니다.

```bat
cd /d C:\Users\paenco0313\Downloads\kq_tool
set PYTHONPATH=%CD%\src;%PYTHONPATH%
python -m kq_tool
```

## 수동 실행 방법

PowerShell 또는 명령 프롬프트에서:

```bat
cd /d C:\Users\paenco0313\Downloads\kq_tool
python -m pip install -r requirements.txt
set PYTHONPATH=%CD%\src;%PYTHONPATH%
python -m kq_tool
```

기존 방식도 호환용으로 계속 동작합니다.

```bat
python server.py
```

서버가 켜지면 Chrome에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8888/
```

## 자주 나는 문제

### `python`을 찾을 수 없다고 나옴

Python이 없거나 PATH에 등록되지 않은 상태입니다. Python을 다시 설치하면서
`Add python.exe to PATH`를 체크해야 합니다.

### 브라우저가 Edge로 열림

`RUN_KQ_TOOL.bat`는 Chrome이 있으면 Chrome을 먼저 열도록 되어 있습니다.
그래도 Edge가 열리면 Chrome 주소창에 직접 아래 주소를 입력하면 됩니다.

```text
http://127.0.0.1:8888/
```

### `Address already in use` 오류

이미 서버가 켜져 있는 상태입니다. 기존 실행 창을 닫거나 `Ctrl+C`로 멈춘 뒤 다시 실행합니다.

### `hmmlearn` 설치 중 Microsoft Visual C++ 오류

Python 3.14에서 오래된 실행 파일이나 오래된 `requirements.txt`로 실행하면
`hmmlearn`이 직접 빌드되면서 Microsoft Visual C++ 오류가 날 수 있습니다.
최신 프로젝트 파일에서는 Python 3.14일 때 `hmmlearn` 설치를 자동으로 건너뛰고
내장 국면 전환 fallback으로 실행합니다.

이 오류가 보이면 최신 `requirements.txt`, `RUN_WINDOWS.ps1`, `RUN_KQ_TOOL.bat`가
같이 복사되어 있는지 확인한 뒤 `RUN_KQ_TOOL.bat`를 다시 실행하세요.
국면 모델 의존성을 가장 안정적으로 쓰려면 Python 3.10~3.13을 권장합니다.

### 실행 창에 오류가 계속 남

필요 패키지 설치가 실패했거나 데이터 파일이 빠진 경우입니다. 실행 창의 마지막 오류 메시지를 캡처해서 공유하면 됩니다.

## 서버 상태 빠른 확인

서버가 켜진 상태에서 PowerShell을 하나 더 열고 아래 명령을 실행하면 됩니다.

```bat
python tests\smoke_api.py
```

정상이면 `ping`과 `health`가 `OK`로 표시됩니다.

종목 분석 API까지 같이 확인하려면 시간이 더 걸릴 수 있지만 아래처럼 실행합니다.

```bat
python tests\smoke_api.py --include-stock --timeout 120
```

데모 전 주요 기능 API까지 한 번에 확인하려면 아래처럼 실행합니다.

```bat
python tests\smoke_api.py --include-stock --include-core --timeout 180
```

## 시황/애널리스트 리포트 수집

공개 RSS나 웹페이지를 국면진단 보조 근거로 쓰려면 먼저 샘플 설정을 만듭니다.

```powershell
python collect_market_reports.py --init-sample
```

`data\report_sources.json`에서 허용된 공개 소스 URL을 넣고 `enabled`를 `true`로 바꾼 뒤 실행합니다.

```powershell
python collect_market_reports.py
```

결과는 `data\market_reports\latest.md`에 저장되고, AI 시장 국면 탭의 시장시황보고서 보조 진단에 표시됩니다.
브라우저의 AI 시장 국면 탭에서 보유한 `.txt`/`.md` 리포트 파일을 선택하거나 본문을 붙여넣어 직접 등록할 수도 있습니다. 직접 등록한 리포트는 `data\market_reports\user_reports.md`에 누적 저장됩니다.
유료 리포트, 로그인 페이지, 크롤링 금지 페이지는 사용하지 마세요.

