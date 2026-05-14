# Bank Marketing MLflow

은행 고객 데이터를 분석해 마케팅 타겟 모델을 최적화하고 MLflow로 상품화하는 프로젝트입니다.
데이터 전처리부터 모델 학습, 실험 추적, MLflow 기반 배포까지 전체 ML Cycle을 다룹니다.

## 프로젝트 구조

```
├── Bank Marketing 데이터를 활용한 ML Cycle 이해와 MLFlow 기반의 모델 효율화.ipynb
└── data/
    ├── bank-additional-full.csv
    ├── dataset_preprocessing.csv
    └── train.csv
```

## 데이터셋

- **출처**: Bank Marketing Dataset (`bank-additional-full.csv`)
- **크기**: 21개 변수
  - 수치형: 10개
  - 범주형: 11개 (타겟 변수 포함)
- **타겟**: `y` (마케팅 캠페인 구독 여부, 이진 분류)
- **데이터 품질**: 중복 12개 제거, 클래스 불균형 존재

## 주요 내용

### 1. EDA
- 데이터 타입 및 기초 통계 분석
- 범주형 변수: 카이제곱 검정으로 유의미한 피처 8개 선택
- 수치형 변수: Kruskal-Wallis 검정, 상관관계 분석 (다중공선성 제거)
- 왜도/첨도 분석 및 Box-Cox 변환

### 2. 데이터 전처리
- Label Encoding, One-Hot Encoding
- StandardScaler / MinMaxScaler
- 클래스 불균형 처리: **SMOTE**, **SMOTEENN**

### 3. 모델 학습 및 최적화
- **MLPClassifier** (Multi-Layer Perceptron)
- Stratified K-Fold (n_splits=3) 교차 검증
- 샘플링 전략 3가지 × 하이퍼파라미터 조합 → 총 9개 모델 비교
- 평가 지표: Precision, Recall, F1, ROC-AUC

### 4. 모델 해석
- SHAP (Shapley Values) 기반 피처 중요도 분석

### 5. MLflow 실험 관리 및 배포
- **Experiment Tracking**: 파라미터/메트릭/아티팩트 로깅
- **Model Registry**: `models:/bank_marketing/Production`으로 등록
- **REST API 서빙**: `mlflow models serve` (포트 5001)
- **온라인 추론**: HTTP POST `/invocations` 엔드포인트

```python
# 모델 서빙 예시
url = "http://127.0.0.1:5001/invocations"
json_data = {"dataframe_split": X_validation[:10].to_dict(orient="split")}
response = requests.post(url, json=json_data)
```

## 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 처리 | pandas, numpy |
| 시각화 | matplotlib, seaborn |
| 통계 분석 | scipy |
| 머신러닝 | scikit-learn |
| 불균형 처리 | imbalanced-learn (SMOTE, SMOTEENN) |
| 모델 해석 | shap |
| 실험 관리/배포 | MLflow |
