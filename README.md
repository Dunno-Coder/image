# Explainable AI-Generated vs Real Image Classification

This project is a binary image classification research prototype for detecting:

- `real`: real-world or human-created images
- `ai`: AI-generated images

It is implemented as a single Streamlit application with PyTorch training, evaluation, EDL uncertainty outputs, and saliency heatmaps. It is not a fruit disease classification project and does not use FastAPI or React.

## Dataset Preparation

Place three binary AI-vs-real datasets under `datasets/`:

```text
datasets/
  genimage/
    train/
      real/
      ai/
    val/
      real/
      ai/
    test/
      real/
      ai/
  cifake/
    train/
      real/
      ai/
    val/
      real/
      ai/
    test/
      real/
      ai/
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

Supported dataset names are `genimage`, `cifake`, and `ai_vs_real`.

The loader is ImageFolder-based but explicitly remaps labels to:

```text
real = 0
ai   = 1
```

## Models

Traditional class models such as ResNet, VGG, AlexNet, DenseNet, and a basic CNN are excluded from the main comparison.

Main comparison models:

- `dinov3_linear`: DINOv3-style foundation backbone plus a linear classifier
- `siglip2_linear`: SigLIP 2-style foundation backbone plus a linear classifier
- `aimv2_linear`: AIMv2-style foundation backbone plus a linear classifier
- `proposed_mnff_edl`: best foundation backbone plus MNFF plus EDL uncertainty head

The implementation uses `timm` candidate model names as practical fallbacks. The registry in `model.py` is structured so Hugging Face model wrappers can be added later.

By default, foundation backbones are frozen and only the classifier head is trained. For the proposed model, the MNFF module and classifier are trainable.

When training `proposed_mnff_edl`, `train.py` uses `--proposed_backbone auto` by default. If prior baseline rows exist in `results/results.csv`, it selects the best baseline F1-score for that dataset. If no prior baseline rows exist, it falls back to `dinov3_linear`. You can override this with `--proposed_backbone siglip2_linear`, `--proposed_backbone dinov3_linear`, or `--proposed_backbone aimv2_linear`.

## Windows Anaconda Setup

Use Anaconda Prompt or PowerShell after Anaconda is on your `PATH`.

The recommended setup is conda-first. `environment.yml` intentionally does not install PyTorch or `timm` because `timm` depends on PyTorch, and installing it too early can cause conda to choose a CPU PyTorch build before you decide whether this machine should use CUDA.

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ai-real-image
```

Install PyTorch with CUDA for an NVIDIA GPU:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

CPU fallback if CUDA is unavailable or the CUDA solver fails:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Install `timm` after PyTorch is installed:

```bash
conda install -c conda-forge timm
```

If you prefer to create the environment manually instead of using `environment.yml`, run:

```bash
conda create -n ai-real-image python=3.11 pip -y
conda activate ai-real-image
conda install -c conda-forge streamlit pandas numpy scikit-learn pillow opencv tqdm -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge timm -y
```

For a manual CPU-only install, replace the CUDA PyTorch line with:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

`requirements.txt` is included only as a pip fallback for restricted machines where conda packages are not available:

```bash
python -m pip install -r requirements.txt
```

PyTorch should still be installed separately before using that fallback.

Verify the environment:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import streamlit, timm, sklearn, cv2; print('dependencies ok')"
```

## Training Commands

Train each baseline and the proposed model for each dataset.

```bash
python train.py --dataset_name genimage --model_name dinov3_linear --epochs 10 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
python train.py --dataset_name genimage --model_name siglip2_linear --epochs 10 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
python train.py --dataset_name genimage --model_name aimv2_linear --epochs 10 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
python train.py --dataset_name genimage --model_name proposed_mnff_edl --epochs 10 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
```

Repeat with:

```bash
--dataset_name cifake
--dataset_name ai_vs_real
```

Best checkpoints are saved to:

```text
weights/{dataset_name}_{model_name}_best.pt
```

Training appends test metrics to:

```text
results/results.csv
```

## Evaluation Commands

```bash
python evaluate.py --dataset_name genimage --model_name proposed_mnff_edl --batch_size 16 --image_size 224 --device auto
```

Evaluation computes:

- Accuracy
- Balanced Accuracy
- F1-score
- AUROC
- Mean uncertainty
- Average inference time per image in milliseconds

## Compare Results

After training/evaluating all baseline and proposed models, run:

```bash
python compare_results.py
```

This creates:

```text
results/improvement_report.csv
```

For each dataset, the proposed model is compared against the best F1-score among the three baseline foundation models.

Performance improvement formula:

```text
improvement_percent = ((proposed_f1 - best_baseline_f1) / best_baseline_f1) * 100
```

The report automatically marks whether the average improvement is at least 5%, 10%, or 20%.

## Best Backbone Selection Workflow

The system can compare candidate Vision Foundation backbones and automatically select the best one for user-facing image analysis.

Candidate backbones:

```text
dinov3_linear
siglip2_linear
aimv2_linear
```

`proposed_mnff_edl` is not included in best baseline backbone selection. The proposed model can later use the selected best backbone as its base.

First, train and evaluate the candidate backbones. For example, with `cifake`:

```bash
python train.py --dataset_name cifake --model_name dinov3_linear --epochs 5 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
python train.py --dataset_name cifake --model_name siglip2_linear --epochs 5 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
python train.py --dataset_name cifake --model_name aimv2_linear --epochs 5 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
```

Repeat the same commands for each available dataset, such as `genimage` and `ai_vs_real`, if you are using multiple datasets.

Then select the best backbone:

```bash
python select_best_backbone.py --metric f1
```

Other supported selection metrics:

```bash
python select_best_backbone.py --metric auroc
python select_best_backbone.py --metric balanced_accuracy
python select_best_backbone.py --metric accuracy
```

The selector reads:

```text
results/results.csv
```

and saves:

```text
results/best_backbone_config.json
```

The JSON stores the selected metric, best model name, selected score, mean performance values, available datasets, candidate models, timestamp, and warning messages if some candidate models are missing.

Run Streamlit:

```bash
streamlit run app.py
```

In the Streamlit sidebar, use:

```text
Analysis Mode = Auto Best Backbone
```

In this mode, the app loads `results/best_backbone_config.json` and uses `best_model_name` as the default image analysis model. The user does not need to manually decide which backbone is best. If the config file does not exist yet, the app shows a friendly warning and allows fallback to manual model selection.

The LLM explanation layer still does not classify images. It only explains the classifier output using structured metadata such as the selected backbone, predicted label, probabilities, confidence, uncertainty, recommended action, threshold, and heatmap summary. The uploaded image itself is not sent to an LLM.

## Proposed MNFF + EDL With Selected Backbone

The proposed model can now use the automatically selected best backbone as its base. The intended flow is:

```text
compare candidate foundation backbones
-> save best backbone config
-> train proposed_mnff_edl using that selected backbone
-> use the trained proposed model in Streamlit
```

Example:

```bash
python select_best_backbone.py --metric f1
python train.py --dataset_name cifake --model_name proposed_mnff_edl --base_backbone_name auto --epochs 5 --batch_size 16 --lr 1e-4 --image_size 224 --device auto
streamlit run app.py
```

Supported `--base_backbone_name` values:

```text
auto
dinov3_linear
siglip2_linear
aimv2_linear
```

When `--base_backbone_name auto` is used, the training and inference code reads:

```text
results/best_backbone_config.json
```

and uses `best_model_name` as the proposed model's foundation backbone. If the config is missing, the code falls back to the default configured backbone so older workflows still run, but the recommended workflow is to run `select_best_backbone.py` first.

In Streamlit, choose:

```text
Analysis Mode = Manual Model Selection
Model = proposed_mnff_edl
Base Backbone = auto
```

This builds the user-facing proposed model on top of the best backbone selected from prior experiments.

## Streamlit App

Run:

```bash
streamlit run app.py
```

The app supports:

- Image upload
- AI/Real prediction
- `prob_real`
- `prob_ai`
- EDL confidence
- EDL uncertainty
- Uncertainty threshold action: `manual_review` or `auto_decision`
- Gradient saliency heatmap
- Performance table from `results/results.csv`
- Improvement table from `results/improvement_report.csv`
- Average improvement judgment for 5%, 10%, and 20%
- User-friendly Korean explanation of the prediction
- Optional LLM-assisted explanation mode
- Optional user feedback saved to `feedback/feedback.csv`
- Auto Best Backbone analysis mode using `results/best_backbone_config.json`

The app does not save uploaded images. Feedback logging stores only prediction metadata, the original uploaded filename, and the selected feedback value.

## LLM Explanation Layer

The image classifier remains responsible for the actual AI-generated vs real prediction. The LLM is not the classifier. It is only an explanation generator that translates structured model evidence into user-friendly Korean text. The LLM explanation layer does not classify the image and does not receive the uploaded image itself.

The explanation layer receives only structured prediction metadata:

```text
predicted_label
prob_real
prob_ai
confidence
uncertainty
action
uncertainty_threshold
heatmap_summary, if available
selected_backbone
```

This metadata is converted into a concise Korean explanation for non-expert users. The explanation is designed to avoid overclaiming: it should say that the result is a model-based estimate, not a 100% certain judgment. If uncertainty is high, it should clearly explain why manual review is recommended.

External LLM usage is optional. The system works without any API key by using the rule-based explanation in `explanation.py`.

Supported explanation providers:

```text
none    -> rule-based explanation only
openai  -> safe placeholder wrapper, falls back if not configured
gemini  -> safe placeholder wrapper, falls back if not configured
ollama  -> safe placeholder wrapper, falls back if not configured
```

Environment variable examples:

```bash
set OPENAI_API_KEY=your_key_here
set GEMINI_API_KEY=your_key_here
set OLLAMA_HOST=http://localhost:11434
```

PowerShell examples:

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:GEMINI_API_KEY="your_key_here"
$env:OLLAMA_HOST="http://localhost:11434"
```

At this stage the external provider wrappers are intentionally safe placeholders. If a provider is missing or unavailable, Streamlit shows a warning and uses the rule-based explanation instead.

## EDL Output Interpretation

For logits from the model:

```text
evidence = softplus(logits)
alpha = evidence + 1
S = sum(alpha)
prob = alpha / S
uncertainty = K / S
confidence = 1 - uncertainty
K = 2
```

High uncertainty means the model has weak evidence for both classes. In the Streamlit app, if uncertainty is above the selected threshold, the action becomes `manual_review`.

## Execution Order

1. Create the conda environment with `conda env create -f environment.yml`.
2. Activate it with `conda activate ai-real-image`.
3. Install PyTorch with either the CUDA command or the CPU fallback command.
4. Install `timm` with `conda install -c conda-forge timm`.
5. Prepare datasets under `datasets/` using the required `real` and `ai` folders.
6. Train all three baseline models for all three datasets.
7. Run `python select_best_backbone.py --metric f1` to create `results/best_backbone_config.json`.
8. Train `proposed_mnff_edl` for all three datasets if you need the proposed-model comparison.
9. Run `evaluate.py` if you need additional test rows in `results/results.csv`.
10. Run `compare_results.py` to create `results/improvement_report.csv`.
11. Run `streamlit run app.py`.
12. Use `Analysis Mode = Auto Best Backbone`, upload an image, and inspect prediction, uncertainty, action, explanation, heatmap, and performance tables.
