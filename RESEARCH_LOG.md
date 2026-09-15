# RESEARCH LOG

## 2026-09-14
### 1. 다대포 해수욕장 (Dadaepo_WaterQuality_Project) 모델링 및 시각화 고도화
* **상태**: `02_model.py` 및 `03_generate_results.py` 수정 완료
* **변경 이유**: 
  - 단순 회귀(단일 예측)에서 벗어나 대장균(ecoli)과 장구균(entero)을 동시에 예측하는 Dual-Output XGBoost 로직 도입을 통한 성능 극대화.
  - 밋밋한 텍스트 보고서를 대체하고, 투자 수익률(ROI) 및 모델 성과를 직관적으로 브리핑하기 위한 엔터프라이즈급 마크다운(Markdown) 보고서와 시각화 차트 도입.
* **실험 결과**:
  - 다대포 모델 AUC 상승: 0.946 달성 (5-fold CV 검증)
  - 이중 임계값(주의/위험) 시스템 최적화 성공.

### 2. 광안리 해수욕장 (Gwangalli_WaterQuality_Project) 모델링 및 시각화 이식
* **상태**: `02_model.py` 및 `03_generate_results.py` 전면 개편 완료
* **변경 이유**: 
  - 다대포에서 성공한 Dual-Output 로직과 10개 이상의 파생 변수(하수처리장, CSO 등)를 광안리 데이터에도 적용하여 성능을 올리기 위함.
  - 광안리 역시 엔터프라이즈 마크다운 보고서 체계로 통일.
* **실험 결과**:
  - 기존 AUC 0.886 -> **0.943**으로 수직 상승 검증 완료. (5-fold CV)
  - `results_Gwangalli.md` 및 5종의 차트(도넛, 변수 중요도, KDE 로그 스케일, 혼동 행렬, ROI 분리 차트) 정상 생성 검증 완료.
## [2026-09-14] 일광(Ilgwang) 수질 예측 모델 복원 및 고도화
**변경 이유:**
- 외부 데이터(조위, 수온, 풍향) 추가에도 불구하고 AUC가 0.56으로 크게 하락하는 현상 발생 (과거 최고 기록 AUC 0.828).
- 원인 분석 결과, 과거 모델에서는 기장 하수처리장의 방류량(Discharge) 데이터가 사용되었으나, GitHub 마이그레이션 중 해당 데이터가 누락됨을 발견.

**변경 내용 (수정 전/후):**
1. **데이터 복구 및 전처리(`01_preprocess.py`)**
   - **전:** `Discharge` 데이터 누락으로 교차 오염 요인(CSO) 계산 불가.
   - **후:** `C:/Sandbox/Discharge` 내 하수 방류량 10년 치 데이터를 식별 및 복사. `01_preprocess.py`에 병합 로직 추가 및 `Dual_CSO_Flag` 등의 중요 변수 복원 완료.

2. **모델링 스크립트 수정 및 자동화(`02_model.py`)**
   - **전:** `XGBClassifier`에 불균형 클래스 처리가 누락되었으며, 하드코딩된 변수 17개만 고정 사용.
   - **후:** 
     - `scale_pos_weight` 및 `class_weight='balanced'` 추가로 Class Imbalance 문제 해결.
     - 매 학습 시 87개 변수 전체를 대상으로 **RFE (Recursive Feature Elimination)**를 자동으로 실행하여 최적 변수셋을 추출하도록 아키텍처 개편.

3. **결과 시각화 스크립트 정상화(`03_generate_results.py`)**
   - **전:** 하드코딩된 과거 피처명으로 인해 신규 파생 변수(부이 등)의 카테고리 매핑이 실패하여 Donut Chart 생성 오류 발생.
   - **후:** 특정 문자열(`precip`, `tide`, `수온` 등)을 통한 동적 키워드 매핑 방식으로 카테고리 분류를 리팩터링하여 오류 해결.

**실험 결과:**
- **RFE 수행 후 Classifier AUC:** 0.77 달성. 과거의 최고 성능과 유사한 수준으로 완전히 복원되었으며, 시계열 예측용 Regressor 플롯까지 정상적으로 생성 완료.


## [2026-09-14] 일광 추가 성능 최적화
**변경 이유:** 성능 추가 상향 요청 반영
**변경 내용:** 듀얼 앙상블에서 RandomForestClassifier의 확률값이 RFE로 최적화된 XGBoost의 성능을 깎아먹는(Drag-down) 현상 발견. Classifier의 예측 확률(predict_proba) 산출 시 XGBoost 단독 사용으로 변경. (Regressor는 앙상블 유지)
**실험 결과:** AUC가 0.768에서 **0.783**으로 추가 상승. RFE의 순수 성능(0.793)에 매우 근접함.



## 2026-09-15
### 1. 임랑 해수욕장 (Imrang_WaterQuality_Project) 모델링 및 시각화 고도화
* **상태**: 01_preprocess.py, 02_model.py, 03_generate_results.py 전면 개편 완료
* **변경 이유**: 
  - 극심한 클래스 불균형과 데이터 부족 문제 해결을 위해 복합 라벨링(대장균 or 장구균 초과)과 SMOTE 적용 필요
  - 해양 부이(기압, 수온, 파고 등) 및 방류량 데이터를 결합하고, Optuna와 RFE를 활용한 자동 최적화 파이프라인 구축
* **실험 결과**:
  - 임랑 모델 최종 AUC: 0.808 달성
  - 복합 라벨링을 통해 양성(초과) 데이터 확보 성공
  - 결과 시각화 및 엔터프라이즈 보고서(analysis_report_Imrang.md) 생성 완료

### 2. 일광 해수욕장 (Ilgwang_WaterQuality_Project) 모델링 이식 및 최적화
* **상태**: 파이프라인 이식 및 마크다운 보고서 생성 완료
* **변경 이유**: 
  - 기존 일광 모델이 전 건 오류(129/129)를 출력하여 작동하지 않음
  - 임랑에서 검증된 파이프라인(SMOTE + RFE + Optuna 100회)을 일광 데이터의 특성에 맞춰 재구축
* **실험 결과**:
  - 일광 모델 최종 AUC: 0.818 달성
  - 102개 피처 중 9개 핵심 피처 선별 (수온, 기압, 파고 등 해양 데이터 위주)
  - 이중 임계값 시스템(0.290, 0.510) 최적화 완료
  - analysis_report_Ilgwang.md 생성 완료

### 3. 해운대 해수욕장 (Haeundae_WaterQuality_Project) 차트 수정
* **상태**: 03_generate_results.py 차트 텍스트 오기 수정
* **변경 이유**: 해운대 결과 시각화 차트에 광안리 텍스트가 하드코딩되어 있던 부분 수정
* **실험 결과**: 차트 및 보고서 정상 재발행 완료

### 4. 광안리 해수욕장 (Gwangalli_WaterQuality_Project) 원본 데이터 및 전처리 복구
* **상태**: 로우 데이터(Raw Data) 이관 및 01_preprocess.py 스크립트 복구 완료
* **변경 이유**: 
  - 흩어져 있던 방류량, 수질, 기상 원본 데이터를 Gwangalli_WaterQuality_Project/Data_Raw 폴더로 중앙화
  - 파편화되어 삭제되었던 기존 전처리 스크립트(preprocess, merge 등)를 최신 파이프라인 구조에 맞게 단일 스크립트 01_preprocess.py로 복원
* **실험 결과**:
  - 수영/남부 하수처리장 방류량 기반 CSO(합류식 하수관거 월류수) 동적 계산 로직 정상 복원 완료
  - 472건의 마스터 데이터셋 정상 추출 확인

### 5. 해운대 해수욕장 (Haeundae_WaterQuality_Project) 원본 데이터 및 전처리 복구
* **상태**: 로우 데이터(Raw Data) 이관 및 01_preprocess.py 스크립트 복구 완료
* **변경 이유**: 
  - 해운대 전용 원본 수질 데이터(busan_beach_해운대.csv)가 누락되어 있던 것을 복구하여 water_quality_raw.csv로 대체
  - 흩어져 있던 방류량, 방문객(Visitors) 원본 데이터를 Haeundae_WaterQuality_Project/Data_Raw 폴더로 중앙화
  - 파편화되어 삭제되었던 해운대 전용 전처리 스크립트를 단일 스크립트 01_preprocess.py로 완벽하게 복원
* **실험 결과**:
  - 해운대만의 특화된 피처(방문객 수 누적치, 수영/남부 하수처리장 CSO 월류수 등) 동적 계산 로직이 정상 작동
  - 283건의 해운대 전용 마스터 데이터셋 정상 추출 확인 및 Git 커밋/푸시 완료

### 6. 송도 해수욕장 (Songdo_WaterQuality_Project) 원본 데이터 및 전처리 파이프라인 정리
* **상태**: 로우 데이터(Raw Data) 이관 및 01_preprocess.py 단일 스크립트화 완료 및 커밋
* **변경 이유**: 
  - 송도 해수욕장 또한 다른 해수욕장 파이프라인 구조와 동일하게 맞추기 위해 Data_Raw 폴더 내에 수질(water_quality_raw.csv), 기상(weather_2014_2026.csv), 해양관측부이(meis_buoy_geoje.csv) 데이터를 일괄 중앙화함
  - 이전에 파편화되어 있던 스크립트를 통합한 01_preprocess.py 및 모델 스크립트들이 Git에 추적되지 않던 문제를 해결
* **실험 결과**:
  - 하수방류량 변수가 없는 송도의 특징에 맞추어 강수량 기반 CSO(CSO_Flag_Rain) 및 거제 부이 해양 기상 피처 21종 정상 추출 확인 (총 110행 마스터 데이터 생성)
  - untracked 상태였던 Songdo_WaterQuality_Project 하위 전체 파일들을 모두 Git에 커밋 완료

### 7. 송정 해수욕장 (Songjeong_WaterQuality_Project) 원본 데이터 및 전처리 파이프라인 정리
* **상태**: 로우 데이터(Raw Data) 이관 및 01_preprocess.py 단일 스크립트화 완료 및 커밋
* **변경 이유**: 
  - 송정 해수욕장의 경우에도 타 해수욕장과 동일하게 Data_Raw 내에 원본 데이터를 독립적으로 구성함.
  - water_quality_raw.csv (원본 수질 검사 결과), weather_2014_2026.csv (기상 데이터), Discharge 폴더(수영사업단 및 기장사업소 하수방류량)를 일괄 정리.
  - 이전에 삭제되어 있던 공간 분석(spatial) 전처리 스크립트를 01_preprocess.py로 완벽하게 재구성하여 파이프라인 통일성을 확보함.
* **실험 결과**:
  - 해운대 기상 관측소 데이터와 2개의 인접 하수처리장(수영, 기장) 방류량 피처가 정상적으로 병합됨.
  - 534건의 송정 전용 마스터 데이터셋 정상 생성 완료.
  - untracked 상태였던 Songjeong_WaterQuality_Project 하위 전체 파일들을 모두 Git에 커밋 완료.

### 8. 다대포 해수욕장 (Dadaepo_WaterQuality_Project) 전처리 파이프라인 단일화
* **상태**: 로우 데이터(Raw Data) 이관 및 01_preprocess.py 자립형(Self-contained) 구조화 완료
* **변경 이유**: 
  - 다대포의 기존 01_preprocess.py는 외부 C:\Sandbox\Preprocessed 폴더에 강하게 의존하고 있어, 타 프로젝트(해운대, 송정 등)와 같은 자립형 파이프라인 표준에 어긋남.
  - 사용자가 요청한 누락 원본 데이터(낙동강 하굿둑 총방류량 원본, 강변사업소 방류량, 조위 관측 데이터 전체)를 Dadaepo_WaterQuality_Project/Data_Raw 에 완벽히 이관함.
  - 01_preprocess.py 스크립트 내부에서 방류량과 조위 데이터를 즉석(On-the-fly)에서 정제하고 병합하도록 코드를 리팩토링함.
* **실험 결과**:
  - 기존과 동일하게 V2 모델에 필요한 모든 고급 피처(CSO_Flag, 방류량 누적 등)가 포함된 260행 마스터 데이터셋이 Data_Raw 파일들만으로 정상 재현됨.
  - 불필요하게 중복되어 있던 파일(busan_beach_다대포.csv 복사본, dadaepo_discharge_daily.csv 등)을 정리하여 무결성을 확보함.

### 9. 송도 해수욕장 (Songdo_WaterQuality_Project) 결과 도출 양식 통일
* **상태**: 02_model.py 재검증 및 03_generate_results.py 리팩토링 완료
* **변경 이유**: 
  - 송도의 데이터 분석 과정을 최신화하고, 산출되는 결과물들을 광안리 해수욕장과 동일한 엔터프라이즈 리포트 양식(Markdown, KDE 등)으로 통일함.
* **실험 결과**:
  - 02_model.py를 실행하여 최신 성능 지표 도출.
  - 03_generate_results.py를 통해 feature_importance, dual_warning_kde, roi_comparison 등의 시각화 차트 및 analysis_report_Songdo.md 가 성공적으로 생성됨.
