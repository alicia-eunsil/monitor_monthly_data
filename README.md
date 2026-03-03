# KOSIS 월별 일자리 모니터링 (MVP)

## 기능 범위
- 데이터셋
  - 경제활동인구현황 (`DT_1DA7004S`)
  - 산업별 취업자수 대분류 (`DT_1DA7E33S_NEW`)
  - 직종별 취업자수 대분류 (`DT_1DA7E34S_NEW`)
- 지역: 전국 + 17개 시도(세종 포함)
- 분석
  - 월별 원자료 추이
  - 전년동월대비 절대증감/증감률 추이
  - 전체기간 최고/최저
  - 최근 5년 최고/최저
  - 최신값이 극값 갱신 시 붉은 `NEW` 표시
- 리포트: 앱 화면 내 표 형식 리포트

## 설치
```powershell
cd kosis_employment_monitor
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## API Key 설정
```powershell
$env:KOSIS_API_KEY="YOUR_NEW_KEY"
```

주의:
- 키가 외부에 노출되면 즉시 폐기/재발급하세요.
- 코드에 키를 하드코딩하지 마세요.

## 실행
```powershell
streamlit run app.py
```

## 수집 스크립트(선택)
```powershell
python scripts/collect_api.py --end-prd-de 202603 --outdir data
```

생성 파일:
- `data/raw_*.json`
- `data/normalized_*.csv`
- `data/normalized_all_with_yoy.csv`

