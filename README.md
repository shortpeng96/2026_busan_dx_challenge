# 🌊 2026 부산 해수욕장 수질 AI 예측 입수 통제 모델 (2026 Busan DX Challenge)

본 레포지토리는 2026년 부산디지털혁신(DX) 챌린지 - "해수욕장 수질 AI 예측 입수 통제 시스템" 구축을 위한 전체 데이터 파이프라인과 모델을 포함하고 있습니다.
본 저장소 하나만 복제(Clone)하더라도 즉시 전체 데이터 분석, 모델 학습 및 결과물 재생성이 가능하도록 독립적(Self-contained)으로 구성되었습니다.

## 🎯 주요 성과 (ROC-AUC)
주요 해수욕장의 수질 오염 예측 모델링을 수행하여 다음과 같은 최고 수준의 AUC를 달성했습니다.
* **송도 (Songdo)**: `0.911` ✅ (최고 성능 달성)
* **임랑 (Imrang)**: `0.873` ✅ (목표 초과 달성)
* **일광 (Ilgwang)**: `0.828` (데이터 한계치 도달 - 현존 데이터로 구성 가능한 최대 예측 성능 확인)

## 📁 디렉토리 구조
각 해수욕장별로 프로젝트 폴더(`[Beach]_WaterQuality_Project`)가 독립적으로 존재하며, 내부 구조는 모두 다음과 같이 일원화되어 있습니다.

```
2026_busan_dx_challenge/
├── README.md                      ← 본 문서
├── requirements.txt               ← 필요 라이브러리 목록
├── [해수욕장명]_WaterQuality_Project/
│   ├── Data_Raw/                  ← 외부 수집 원시 데이터 (해양부이, 기상청, 수질관측 등)
│   ├── Data_Processed/            ← 01_preprocess.py에 의해 생성되는 최종 학습용 CSV
│   │   └── master_dataset.csv
│   ├── Scripts/                   ← 실행 스크립트 모음 (순차 실행)
│   │   ├── 01_preprocess.py       ← 데이터 전처리 및 병합
│   │   ├── 02_model.py            ← AI 예측 모델 학습 (XGBoost)
│   │   └── 03_generate_results.py ← 결과 리포트 및 시각화 도표 생성
│   └── Results/                   ← 학습된 모델(pkl), 성과 리포트, 그래프 이미지 등 생성
```

> **참고**: `Dadaepo`, `Gwangalli`, `Haeundae`, `Songjeong` 등 보조 분석 해수욕장 또한 위와 동일한 구조로 관리되어 일관성을 유지합니다.

## 🚀 시작하기 (How to Run)

### 1. 환경 설정
필요한 패키지를 설치합니다. Python 3.9+ 이상의 환경을 권장합니다.
```bash
pip install -r requirements.txt
```

### 2. 스크립트 실행 순서
각 해수욕장의 `Scripts/` 폴더 내에 있는 3개의 파일을 순차적으로 실행하면 됩니다.
예를 들어, **송도 해수욕장**의 파이프라인을 실행하려면 다음과 같이 진행합니다:

```bash
cd 2026_busan_dx_challenge/Songdo_WaterQuality_Project/Scripts

# 1) 원시 데이터를 가공하여 master_dataset.csv 생성
python 01_preprocess.py

# 2) 전처리된 데이터로 모델 학습 후 Results 폴더에 pkl 저장
python 02_model.py

# 3) 예측 결과와 모델 성능 기반의 분석 리포트(txt), 혼동행렬(png) 생성
python 03_generate_results.py
```

## 🛠 주요 기술 및 방법론
* **전진 선택법(Forward Selection)**: 최적의 피처 조합을 도출하여 데이터 누수 방지 및 다중공선성 최소화.
* **해양관측부이 시계열 병합**: 3일/7일 이동평균 등 해류 확산을 반영한 공간 파생 변수 적용.
* **이중 임계값(Dual Warning System)**: 
  * 🟡 **주의 단계**: F2-Score 최적화를 통한 선제적 방어 시스템
  * 🔴 **위험 단계**: F0.5-Score 최적화를 통한 비즈니스 타격(오탐) 최소화 로직 도입
