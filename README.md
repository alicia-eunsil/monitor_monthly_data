# 경제활동인구 데이터 모니터링

KOSIS Open API를 사용해 취업/고용 관련 지표를 모니터링하는 Streamlit 앱이다.  
두 가지 조회 범위를 지원한다.

- `전국+17개 시도` 모드: 월별(경제활동인구조사)
- `경기 31개 시군` 모드: 반기별(지역별고용조사)

## 주요 기능

- 데이터셋 5종 모니터링
  - 경제활동인구현황
  - 연령별 취업자
  - 종사상지위별 취업자
  - 산업별 취업자수
  - 직종별 취업자수
- 공통 분석
  - 원자료 추이
  - 전년동월(또는 전년동기) 대비 증감/증감률 추이
  - 전체기간 최고/최저
  - 최근 5년 최고/최저
  - 최신값이 극값 갱신 시 `NEW` 표시
- `NEW HISTORY`
  - 극값 갱신 이력(원자료/증감/증감률, 최고/최저, 전체기간/최근5년)
  - 이벤트 분석 차트
- `REPORT`
  - 월별 주요 NEW 이벤트를 마크다운 리포트로 요약
  - 하단에 `AI INSIGHTS` 표시(전국+17개 시도 모드에서만 표시)
    - 영향요인분해(전국 내 경기도 비중)
    - 영향요인분해(지역별)
    - Robust Z-score 기반 이상탐지
    - 수치 근거 중심 AI 해설 문장 출력

## 데이터 소스 (KOSIS)

### 1) 전국+17개 시도 (월별)
- 경제활동인구현황: `DT_1DA7004S`
- 연령별 취업자: `DT_1DA7031S`
- 종사상지위별 취업자: `DT_1DA7035S`
- 산업별 취업자수: `DT_1DA7E33S_NEW`
- 직종별 취업자수: `DT_1DA7E34S_NEW`

### 2) 경기 31개 시군 (반기)
- 경제활동인구현황: `DT_1ES3A01S`
- 연령별 취업자: `DT_1ES3A03_A01S`
- 종사상지위별 취업자: `DT_1ES3A07S`
- 산업별 취업자수: `DT_1ES3A30S`
- 직종별 취업자수: `DT_1ES3A31S`

## 설치

```powershell
cd monitor_monthly_data
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## API Key 설정

아래 4개 중 하나로 설정하면 된다.

1. 환경변수 `api_key`
2. 환경변수 `API_KEY`
3. Streamlit secrets `api_key`
4. Streamlit secrets `API_KEY`

로컬 예시:

```powershell
$env:api_key="YOUR_KOSIS_API_KEY"
```

Streamlit Cloud 예시:
- App settings > Secrets에 `API_KEY` 또는 `api_key` 등록

## 접속코드 설정

앱은 접속코드가 설정되어 있어야 실행된다.

아래 4개 중 하나로 설정하면 된다.

1. 환경변수 `access_code`
2. 환경변수 `ACCESS_CODE`
3. Streamlit secrets `access_code`
4. Streamlit secrets `ACCESS_CODE`

로컬 예시:

```powershell
$env:ACCESS_CODE="YOUR_ACCESS_CODE"
```

Streamlit Cloud 예시:
- App settings > Secrets에 `ACCESS_CODE` 또는 `access_code` 등록

## 실행

```powershell
streamlit run app.py
```

## 동작 메모

- 조회 시 로딩바/상태 문구를 표시한다.
- 초기 로딩 안내 문구: `데이터 불러오는 중... (약 10분 소요예정)`
- `데이터 새로고침` 버튼으로 세션 캐시를 초기화할 수 있다.
- 월별 데이터의 종료시점은 실행 시점 기준 현재월을 사용한다.
- 반기 데이터(경기 31시군)는 현재 `202502`를 최신 반기값으로 고정 조회한다.
- 이상탐지 표는 `이상점수 50점 이상`만 표시하며, `분류 > 기준시점(최신순) > 이상점수`로 정렬한다.
