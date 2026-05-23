# 설명가능한 AI 생성 이미지 탐지 시스템

이 프로젝트는 업로드한 이미지가 **AI 생성 이미지**인지, 또는 **실제/사람 생성 이미지**인지 판별하는 이진 이미지 분류 연구 데모입니다. 단순히 `AI` / `Real` 결과만 출력하지 않고, 확률, confidence, uncertainty, heatmap, LLM 기반 자연어 설명까지 함께 제공하는 **XAI 기반 AI-generated image detection system**입니다.

기존 Streamlit 앱은 이미지 업로드와 예측을 담당하고, `analysis/` 디렉터리는 교수님 요구 산출물 생성을 위한 독립 분석 파이프라인을 담당합니다. 분석 파이프라인은 기존 앱을 대체하지 않으며, 이미지 프로젝트를 억지로 tabular project로 바꾸지 않습니다. 대신 이미지 통계 특징과 vision backbone embedding을 사용해 TableOne, CCA, SHAP, LIME, subgroup analysis 등을 이미지 분류 문제에 맞게 적용합니다.

## 주요 기능

- Streamlit 기반 이미지 업로드 UI
- AI 생성 이미지 / 실제 이미지 이진 분류
- `dinov3_linear`, `siglip2_linear`, `aimv2_linear` baseline 비교
- `proposed_mnff_edl` 제안 모델
- AI 확률, Real 확률 출력
- EDL 기반 confidence / uncertainty 계산
- uncertainty threshold 기반 자동 판정 / 수동 검토 권장
- gradient saliency heatmap 기반 시각적 설명
- OpenAI API 또는 Ollama 기반 LLM 자연어 설명
- 사용자 피드백 저장
- 교수님 요구 분석 산출물 자동 생성 파이프라인
- 한국어 원고 초안 생성

## 프로젝트 구조

```text
.
├─ app.py                         # Streamlit 데모 앱
├─ train.py                       # 모델 학습
├─ evaluate.py                    # 모델 평가
├─ model.py                       # backbone 및 proposed 모델 정의
├─ inference.py                   # 추론 로직
├─ metrics.py                     # EDL 및 평가 지표
├─ gradcam.py                     # heatmap / saliency 생성
├─ explanation.py                 # 규칙 기반 및 LLM 설명 생성
├─ compare_results.py             # proposed vs baseline 개선율 비교
├─ select_best_backbone.py        # best backbone 자동 선택
├─ analysis/                      # 독립 분석 파이프라인
├─ datasets/                      # 이미지 데이터셋
├─ weights/                       # 학습된 checkpoint
├─ results/                       # 실험 결과, 그림, 원고
└─ feedback/                      # 사용자 피드백
```

## 데이터셋 구조

현재 데이터 로더는 ImageFolder 구조를 사용합니다. 각 split 아래에는 `real`과 `ai` 폴더가 있어야 합니다.

```text
datasets/
  ai_vs_real/
    train/
      real/
      ai/
    val/
      real/
      ai/
    test/
      real/
      ai/
```

라벨은 코드에서 명시적으로 다음처럼 매핑됩니다.

```text
real = 0
ai   = 1
```

코드상 `train.py`와 `evaluate.py`는 `genimage`, `cifake`, `ai_vs_real` 선택지를 지원하지만, 현재 로컬 프로젝트의 주 데이터셋은 `ai_vs_real`입니다.

## 사용 모델

### Foundation 기반 분류 모델

| 모델명 | 설명 |
|---|---|
| `dinov3_linear` | DINOv3 계열 vision foundation backbone + linear classifier |
| `siglip2_linear` | SigLIP2 계열 vision foundation backbone + linear classifier |
| `aimv2_linear` | AIMv2 계열 vision foundation backbone + linear classifier |
| `proposed_mnff_edl` | MNFF 모듈 + EDL uncertainty head를 포함한 제안 모델 |

현재 실험 결과 기준으로 best baseline은 `aimv2_linear`입니다.

```text
best baseline: aimv2_linear
F1: 0.8596
AUROC: 0.8847
```

### LLM 설명 모델

| 제공자 | 상태 |
|---|---|
| OpenAI API | Responses API 연결 구현 |
| Ollama 로컬 | `llama3.1:8b` 기본값으로 연결 구현 |
| Gemini | UI 옵션만 있음, 현재는 규칙 기반 설명으로 fallback |
| none | 규칙 기반 설명만 사용 |

중요한 점은 LLM이 이미지를 직접 판별하지 않는다는 것입니다. LLM에는 이미지 원본이 전송되지 않고, 아래와 같은 구조화된 예측 메타데이터만 전달됩니다.

```text
predicted_label
prob_real
prob_ai
confidence
uncertainty
uncertainty_threshold
action
heatmap_summary
selected_backbone
```

## 환경 설정

이 프로젝트는 Windows + Anaconda 기준으로 정리되어 있습니다. 현재 실제 실행 기준 환경 이름은 `ai-real-gpu`입니다.

### 1. 새 환경을 만드는 경우

```powershell
cd C:\Users\mhyun\Documents\image
conda env create -f environment.yml
conda activate ai-real-gpu
```

`environment.yml`에는 Streamlit 앱, sklearn 분석, TableOne-style 분석, CCA, UMAP, SHAP, LIME 실행에 필요한 기본 패키지가 포함되어 있습니다. 단, PyTorch는 GPU/CPU 선택이 필요하므로 아래에서 별도로 설치합니다.

### 2. 이미 만든 환경을 쓰는 경우

```powershell
conda activate ai-real-gpu
cd C:\Users\mhyun\Documents\image
python -m pip install -r requirements.txt
```

### 3. PyTorch 설치

NVIDIA GPU를 사용할 경우:

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

CPU만 사용할 경우:

```powershell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 4. timm 확인

`timm`은 `environment.yml`과 `requirements.txt`에 포함되어 있습니다. 누락된 경우에만 아래 명령을 실행하면 됩니다.

```powershell
python -m pip install timm
```

### 5. 분석 패키지 확인

UMAP, odds ratio, SHAP, LIME 산출물까지 모두 생성하려면 아래 패키지들이 필요합니다.

```powershell
python -c "import matplotlib, umap, statsmodels, shap, lime, skimage; print('analysis deps ok')"
```

누락된 패키지가 있다면 현재 활성화된 `ai-real-gpu` 환경에서 실행합니다.

```powershell
python -m pip install matplotlib umap-learn statsmodels shap lime scikit-image
```

### 6. 설치 확인

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import streamlit, timm, sklearn, cv2; print('app deps ok')"
python -c "import pandas, numpy, scipy; print('analysis base deps ok')"
```

## Streamlit 앱 실행

```powershell
cd C:\Users\mhyun\Documents\image
streamlit run app.py
```

앱에서 기본적으로 할 수 있는 일:

- 이미지 업로드
- AI / Real 예측
- AI 확률, Real 확률 확인
- confidence / uncertainty 확인
- 자동 판정 가능 여부 확인
- heatmap 시각화
- LLM 기반 사용자 친화 설명 확인
- 사용자 피드백 저장
- 실험 결과표 확인

## OpenAI API 설명 사용

OpenAI API는 ChatGPT Plus와 별도입니다. API 키는 [OpenAI Platform](https://platform.openai.com/)에서 발급하고, API 사용량은 별도로 과금됩니다.

PowerShell에서 환경변수로 설정:

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:OPENAI_MODEL="gpt-5.4-mini"
streamlit run app.py
```

또는 Streamlit 사이드바의 `OpenAI API Key` 입력칸에 키를 넣을 수 있습니다. 이 키는 결과 CSV나 피드백 파일에 저장되지 않습니다.

앱 설정:

```text
설명 방식 = LLM 보조 설명
LLM 제공자 = OpenAI API
```

## Ollama / Llama 로컬 설명 사용

OpenAI API 키 없이 로컬 LLM을 사용하려면 Ollama를 설치하고 모델을 받아야 합니다.

```powershell
ollama pull llama3.1:8b
ollama list
```

앱 실행 전 환경변수 설정:

```powershell
$env:OLLAMA_HOST="http://localhost:11434"
$env:OLLAMA_MODEL="llama3.1:8b"
streamlit run app.py
```

앱 설정:

```text
설명 방식 = LLM 보조 설명
LLM 제공자 = Ollama 로컬
Ollama 모델 = llama3.1:8b
```

주의: Ollama가 켜져 있어도 모델이 다운로드되어 있지 않으면 `model not found` 오류가 납니다. `ollama list`에 표시되는 모델명과 앱의 모델명이 정확히 같아야 합니다.

## 모델 학습

baseline 모델 학습 예시:

```powershell
python train.py --dataset_name ai_vs_real --model_name dinov3_linear --epochs 5 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
python train.py --dataset_name ai_vs_real --model_name siglip2_linear --epochs 5 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
python train.py --dataset_name ai_vs_real --model_name aimv2_linear --epochs 5 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
```

proposed 모델 학습 예시:

```powershell
python train.py --dataset_name ai_vs_real --model_name proposed_mnff_edl --base_backbone_name auto --epochs 5 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
```

checkpoint 저장 위치:

```text
weights/{dataset_name}_{model_name}_best.pt
```

학습 결과는 다음 파일에 누적됩니다.

```text
results/results.csv
```

## 모델 평가

```powershell
python evaluate.py --dataset_name ai_vs_real --model_name aimv2_linear --batch_size 16 --image_size 224 --device auto
```

평가 지표:

- accuracy
- balanced accuracy
- F1-score
- AUROC
- mean uncertainty
- image당 평균 inference time

## Best Backbone 선택

baseline 모델 결과를 바탕으로 가장 좋은 backbone을 자동 선택합니다.

```powershell
python select_best_backbone.py --metric f1
```

지원 metric:

```text
f1
auroc
balanced_accuracy
accuracy
```

출력 파일:

```text
results/best_backbone_config.json
```

Streamlit의 `Auto Best Backbone` 모드는 이 JSON 파일을 읽어 현재 가장 좋은 모델을 자동으로 사용합니다.

## Proposed 모델 개선율 비교

```powershell
python compare_results.py
```

출력 파일:

```text
results/improvement_report.csv
```

개선율 계산식:

```text
improvement_percent = ((proposed_f1 - best_baseline_f1) / best_baseline_f1) * 100
```

현재 기록 기준으로는 proposed 모델이 best baseline보다 낮기 때문에 5%, 10%, 20% 개선 기준을 통과하지 못했습니다.

## 독립 분석 파이프라인

교수님 요구 산출물은 `analysis/` 디렉터리의 독립 스크립트로 생성합니다.

분석 파이프라인은 raw image pixel을 sklearn 모델에 직접 넣지 않습니다. 대신 다음 두 종류의 feature를 사용합니다.

1. 이미지 통계 특징
2. DINOv3 / SigLIP2 / AIMv2 backbone embedding

### 분석용 선택 패키지 설치

아래 패키지는 모든 분석에 필수는 아니지만, UMAP, odds ratio, SHAP, LIME 산출물을 완전히 만들려면 필요합니다.

```powershell
conda activate ai-real-gpu
cd C:\Users\mhyun\Documents\image
python -m pip install matplotlib umap-learn statsmodels shap lime scikit-image
```

### 추천 실행 순서

1. 이미지 통계 특징 추출

```powershell
python analysis/extract_image_features.py --dataset_name ai_vs_real --splits train,val,test
```

출력:

```text
results/metrics/image_statistical_features.csv
```

2. backbone embedding 추출

```powershell
python analysis/extract_backbone_embeddings.py --dataset_name ai_vs_real --splits train,val,test --models dinov3,siglip2,aimv2 --device auto --batch_size 16
```

출력:

```text
results/embeddings/dinov3_embeddings.npy
results/embeddings/siglip2_embeddings.npy
results/embeddings/aimv2_embeddings.npy
results/embeddings/labels.npy
results/embeddings/image_paths.csv
```

3. sklearn baseline 학습

```powershell
python analysis/train_sklearn_baselines.py --embedding_model aimv2
```

포함 모델:

- LogisticRegression
- SVC
- RandomForestClassifier
- ExtraTreesClassifier
- HistGradientBoostingClassifier
- KNeighborsClassifier
- MLPClassifier

출력:

```text
results/metrics/sklearn_baselines.csv
results/sklearn_models/
results/predictions/
```

4. sklearn + foundation 모델 통합 비교

```powershell
python analysis/evaluate_all_models.py --dataset_name ai_vs_real --device auto
```

출력:

```text
results/metrics/model_comparison.csv
```

5. confusion matrix, ROC curve, PR curve 생성

```powershell
python analysis/plot_confusion_roc_pr.py
```

출력:

```text
results/figures/confusion_matrix.png
results/figures/roc_curve.png
results/figures/pr_curve.png
```

6. TableOne-style 분석

```powershell
python analysis/make_tableone.py --embedding_model aimv2
```

출력:

```text
results/metrics/tableone_image_features.csv
```

7. t-SNE / UMAP 시각화

```powershell
python analysis/visualize_tsne_umap.py --embedding_model aimv2
```

출력:

```text
results/figures/tsne_embedding.png
results/figures/umap_embedding.png
```

`umap-learn`이 설치되어 있지 않으면 UMAP은 건너뛰고 설치 안내를 출력합니다.

8. CCA 및 correlation 분석

```powershell
python analysis/cca_correlation_analysis.py
```

출력:

```text
results/metrics/cca_results.csv
results/metrics/correlation_results.csv
```

CCA 분석:

- DINOv3 embedding vs SigLIP2 embedding
- DINOv3 embedding vs AIMv2 embedding
- SigLIP2 embedding vs AIMv2 embedding

correlation 분석:

- 이미지 통계 특징 vs target
- embedding PCA component vs target
- confidence / uncertainty vs prediction error

9. Odds ratio, confidence interval, p-value

```powershell
python analysis/odds_ratio_analysis.py --embedding_model aimv2
```

출력:

```text
results/metrics/odds_ratio_results.csv
```

`statsmodels`가 설치되어 있지 않으면 실제 OR 계산 대신 설치 안내 fallback CSV가 생성됩니다.

10. SHAP / LIME 분석

```powershell
python analysis/shap_lime_analysis.py --embedding_model aimv2 --sklearn_model sklearn_extra_trees --foundation_model aimv2_linear --device auto
```

출력:

```text
results/figures/shap_summary.png
results/figures/shap_dependency.png
results/figures/shap_force_example.html
results/figures/lime_ai_example.png
results/figures/lime_real_example.png
```

`shap`, `lime`, `scikit-image`가 설치되어 있지 않으면 해당 산출물은 건너뛰고 설치 안내를 출력합니다.

11. Subgroup analysis

```powershell
python analysis/subgroup_analysis.py
```

출력:

```text
results/metrics/subgroup_analysis.csv
```

이미지 프로젝트이므로 demographic subgroup은 사용하지 않습니다. 대신 다음 기준으로 subgroup을 구성합니다.

- low / medium / high uncertainty
- low / high confidence
- low / medium / high AI probability
- brightness quantile groups
- sharpness quantile groups

12. Actual value vs prediction value 시각화

```powershell
python analysis/actual_vs_prediction_plot.py
```

출력:

```text
results/figures/actual_vs_predicted.png
results/figures/probability_distribution_by_actual_class.png
results/figures/confidence_correct_vs_wrong.png
```

13. 한국어 원고 초안 생성

```powershell
python analysis/generate_report_tables.py
```

출력:

```text
results/manuscript/draft.md
results/metrics/analysis_artifact_index.csv
```

원고 초안 포함 항목:

- 서론
- 관련 연구
- 제안 방법
- 실험 방법
- 실험 결과
- 결론
- 레퍼런스

## 현재 생성된 주요 산출물

현재 프로젝트에서 실제 생성 확인된 주요 파일:

```text
results/metrics/image_statistical_features.csv
results/metrics/model_comparison.csv
results/metrics/sklearn_baselines.csv
results/metrics/tableone_image_features.csv
results/metrics/cca_results.csv
results/metrics/correlation_results.csv
results/metrics/odds_ratio_results.csv
results/metrics/subgroup_analysis.csv
results/figures/confusion_matrix.png
results/figures/roc_curve.png
results/figures/pr_curve.png
results/figures/tsne_embedding.png
results/figures/actual_vs_predicted.png
results/figures/probability_distribution_by_actual_class.png
results/figures/confidence_correct_vs_wrong.png
results/embeddings/dinov3_embeddings.npy
results/embeddings/siglip2_embeddings.npy
results/embeddings/aimv2_embeddings.npy
results/manuscript/draft.md
```

현재 통합 비교표 기준 최고 모델은 `sklearn_hist_gradient_boosting`이며, F1-score는 `0.8750`입니다. 기존 foundation baseline 중 최고 모델은 `aimv2_linear`이며, F1-score는 `0.8596`입니다.

## 주의사항

- 이 프로젝트는 이미지 분류 프로젝트입니다. sklearn baseline은 raw pixel이 아니라 backbone embedding을 사용합니다.
- LLM은 이미지를 직접 판별하지 않습니다. 분류 모델 결과를 일반인이 이해하기 쉽게 설명하는 역할만 합니다.
- Heatmap은 보조 설명 자료이며, 단독으로 판정 근거를 확정하면 안 됩니다.
- OpenAI API는 ChatGPT Plus와 별도 과금입니다.
- Ollama는 로컬 PC 성능에 따라 응답 속도가 달라집니다.
- SHAP, LIME, UMAP, statsmodels 기반 분석은 선택 패키지 설치 상태에 따라 실행 여부가 달라집니다.
