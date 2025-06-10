# 의료비 지출 예측 프로젝트 (Medical Expense Prediction)

해당 프로젝트는 한국의료패털 1기 데이터를 활용해, 당뇨병 환자의 의료비 지출을 예측하는 머신 러닝 프로젝트입니다. 다양한 의료 서비스 이용 패턴과 개인/가구 특성이 의료비 지출에 미치는 영향을 분석합니다.

## 프로젝트 구조

```
MedExpensePrediction/
├── data/
│   └── medical/          # 원본 SAS 데이터 파일들
├── plots/               # 생성된 시각화 결과물
├── loader.py           # 데이터 로딩 및 전처리
├── preprocessing.py    # 데이터 전처리 및 변환
├── analysis.py         # 데이터 분석 및 특성 선택
├── plot.py            # 시각화 함수
└── main.ipynb         # 메인 실행 파일
```

## 설치 및 실행 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/MedExpensePrediction.git
cd MedExpensePrediction
```

2. 데이터 준비
- 구글 드라이브 공유 링크 (https://drive.google.com/drive/folders/1NX0Trrf_O2NaQRQervlmoziZPW-4Zmco)에서 전체 데이터셋을 다운받습니다.
- `data/medical/` 폴더에 한국의료패널 SAS 데이터 파일들을 위치시킵니다.
- 코드별 파일:
  - in: 입원 서비스
  - ind: 개인 정보
  - cd: 만성질환
  - appen: 부가진료
  - hh: 가구 정보
  - er: 응급실
  - ou: 외래진료

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

4. Jupyter Notebook 실행
```bash
jupyter notebook
```

5. `main.ipynb` 실행
- 노트북의 모든 셀을 순서대로 실행하여 전체 분석 과정을 수행합니다.
- 시각화 파일들은 `plots/` 폴더에 저장됩니다.

## 파일 설명

### loader.py
- 데이터 로딩 및 초기 전처리를 담당
- SAS 파일들을 DB 코드별로 분류하고 로드
- 연도별 데이터를 하나의 데이터프레임으로 결합

### preprocessing.py
- 데이터 전처리 및 변환 함수들을 포함
- 결측치 처리, 변수 타입 변환, 원-핫 인코딩 등 수행
- 로그 변환, 매핑 등의 기능 제공

### analysis.py
- 데이터 분석 및 특성 선택 기능 제공
- 상관관계 분석, 이상치 처리, 시계열 특성 생성
- 모델 평가 및 예측 함수 포함

### plot.py
- 데이터 시각화 함수들을 포함
- 분포도, 상관관계, 모델 해석 등을 위한 시각화 기능 제공
- Feature Importance 및 LIME을 활용한 모델 해석 시각화

### main.ipynb
- 전체 분석 과정을 순차적으로 실행하는 메인 파일
- 데이터 로딩부터 모델 학습, 평가까지의 전체 파이프라인 포함
- 각 단계별 결과를 시각화하고 저장

## 실행 결과

`main.ipynb`를 실행하면 다음과 같은 결과를 얻을 수 있습니다:
1. 데이터 전처리 및 특성 엔지니어링 결과
2. 다양한 시각화 결과물 (plots/ 폴더에 저장)
3. 모델 성능 평가 지표
4. 특성 중요도 분석 결과
5. 예측 결과 및 해석

## 주의사항

- 데이터 파일은 반드시 `data/medical/` 폴더에 위치해야 합니다.
- 모든 의존성 패키지가 설치되어 있어야 합니다.