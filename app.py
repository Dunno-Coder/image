import csv
import json
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    BEST_BACKBONE_CONFIG_JSON,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_UNCERTAINTY_THRESHOLD,
    FEEDBACK_CSV,
    IMPROVEMENT_REPORT_CSV,
    RESULTS_CSV,
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
)
from explanation import build_evidence_summary, generate_explanation_with_status
from gradcam import generate_heatmap
from inference import load_model_for_inference, predict_pil_image
from utils import get_device, weights_path


APP_TITLE = "Explainable AI-Generated Image Detection System"


st.set_page_config(page_title=APP_TITLE, layout="wide")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def display_progress(label: str, value: float) -> None:
    st.write(label)
    st.progress(clamp01(value))
    st.caption(f"{value:.4f}")


def _weights_mtime(dataset_name: str, model_name: str) -> float:
    path = weights_path(dataset_name, model_name)
    return path.stat().st_mtime if path.exists() else 0.0


def _best_backbone_config_mtime() -> float:
    return BEST_BACKBONE_CONFIG_JSON.stat().st_mtime if BEST_BACKBONE_CONFIG_JSON.exists() else 0.0


@st.cache_resource(show_spinner=False)
def load_cached_model(
    dataset_name: str,
    model_name: str,
    image_size: int,
    device_arg: str,
    weights_mtime: float,
    base_backbone_name: str,
    best_backbone_config_mtime: float,
):
    del weights_mtime
    del best_backbone_config_mtime
    return load_model_for_inference(
        dataset_name=dataset_name,
        model_name=model_name,
        image_size=image_size,
        device_arg=device_arg,
        require_weights=True,
        base_backbone_name=base_backbone_name,
    )


def read_csv_safely(path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not load {path}: {exc}")
        return None


def read_best_backbone_config_safely() -> Optional[Dict[str, object]]:
    try:
        if not BEST_BACKBONE_CONFIG_JSON.exists():
            return None
        return json.loads(BEST_BACKBONE_CONFIG_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load {BEST_BACKBONE_CONFIG_JSON}: {exc}")
        return None


def format_score(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def resolve_base_backbone_for_ui(base_backbone_name: str, best_config: Optional[Dict[str, object]]) -> str:
    if base_backbone_name != "auto":
        return base_backbone_name
    if best_config:
        selected = str(best_config.get("best_model_name", "")).strip()
        if selected in ("dinov3_linear", "siglip2_linear", "aimv2_linear"):
            return selected
    return "auto"


def compact_label(result: Dict[str, object]) -> str:
    return "AI-generated" if int(result["class_index"]) == 1 else "Real"


def action_message(result: Dict[str, object], threshold: float) -> None:
    uncertainty = float(result["uncertainty"])
    if uncertainty <= threshold:
        st.success("Auto decision is allowed.")
    else:
        st.warning("Manual review is recommended because uncertainty is high.")


def save_feedback(
    *,
    feedback_value: str,
    filename: str,
    dataset_name: str,
    model_name: str,
    prediction: Dict[str, object],
) -> None:
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_CSV.exists()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "predicted_label": compact_label(prediction),
        "prob_real": prediction["prob_real"],
        "prob_ai": prediction["prob_ai"],
        "confidence": prediction["confidence"],
        "uncertainty": prediction["uncertainty"],
        "action": prediction["action"],
        "user_feedback": feedback_value,
    }
    fieldnames = [
        "timestamp",
        "filename",
        "dataset_name",
        "model_name",
        "predicted_label",
        "prob_real",
        "prob_ai",
        "confidence",
        "uncertainty",
        "action",
        "user_feedback",
    ]
    with FEEDBACK_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def render_result_card(result: Dict[str, object], threshold: float) -> None:
    label = compact_label(result)
    label_help = "The model predicts this image is AI-generated." if label == "AI-generated" else "The model predicts this image is real."

    with st.container(border=True):
        st.subheader("Prediction Result")
        st.metric("Predicted label", label)
        st.caption(label_help)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AI probability", f"{float(result['prob_ai']):.2%}")
        m2.metric("Real probability", f"{float(result['prob_real']):.2%}")
        m3.metric("Confidence", f"{float(result['confidence']):.2%}")
        m4.metric("Uncertainty", f"{float(result['uncertainty']):.2%}")

        p1, p2 = st.columns(2)
        with p1:
            display_progress("AI probability", float(result["prob_ai"]))
            display_progress("Confidence", float(result["confidence"]))
        with p2:
            display_progress("Real probability", float(result["prob_real"]))
            display_progress("Uncertainty", float(result["uncertainty"]))

        st.metric("Recommended action", str(result["action"]))
        action_message(result, threshold)


def render_explanation_section(image: Image.Image, heatmap: Optional[Image.Image], heatmap_error: Optional[str]) -> None:
    st.subheader("Visual Explanation")
    c1, c2 = st.columns(2)
    c1.image(image, caption="Original image", use_container_width=True)
    if heatmap is not None:
        c2.image(heatmap, caption="Grad-CAM / saliency heatmap", use_container_width=True)
    elif heatmap_error:
        c2.info(f"Heatmap could not be generated for this model/image. {heatmap_error}")
    else:
        c2.info("Heatmap could not be generated for this model/image, but the prediction completed.")


def describe_heatmap_region(heatmap: Image.Image) -> Optional[str]:
    try:
        arr = np.asarray(heatmap.convert("RGB")).astype(np.float32)
        red_score = arr[:, :, 0] - 0.5 * (arr[:, :, 1] + arr[:, :, 2])
        cutoff = np.percentile(red_score, 92)
        mask = red_score >= cutoff
        if mask.sum() < 10:
            return None

        ys, xs = np.where(mask)
        x_center = float(xs.mean()) / max(arr.shape[1] - 1, 1)
        y_center = float(ys.mean()) / max(arr.shape[0] - 1, 1)

        vertical = "위쪽" if y_center < 0.33 else "아래쪽" if y_center > 0.67 else "중앙"
        horizontal = "왼쪽" if x_center < 0.33 else "오른쪽" if x_center > 0.67 else "중앙"

        if vertical == "중앙" and horizontal == "중앙":
            region = "이미지 중앙 부근"
        elif vertical == "중앙":
            region = f"이미지 {horizontal} 부근"
        elif horizontal == "중앙":
            region = f"이미지 {vertical} 부근"
        else:
            region = f"이미지 {vertical} {horizontal} 부근"
        return f"모델이 {region}에 상대적으로 더 민감하게 반응한 것으로 보입니다."
    except Exception:
        return None


def build_heatmap_summary(heatmap: Optional[Image.Image], heatmap_error: Optional[str]) -> Optional[str]:
    if heatmap is not None:
        region_summary = describe_heatmap_region(heatmap)
        if region_summary:
            return region_summary
        return "heatmap/saliency 시각화가 생성되었지만, 뚜렷하게 한 영역만 강조되지는 않은 것으로 보입니다."
    if heatmap_error:
        return None
    return None


def render_user_friendly_explanation(
    prediction: Dict[str, object],
    threshold: float,
    explanation_mode: str,
    llm_provider: str,
    heatmap: Optional[Image.Image],
    heatmap_error: Optional[str],
    selected_backbone: Optional[str],
) -> None:
    st.subheader("User-Friendly Explanation")
    provider = llm_provider if explanation_mode == "LLM-assisted" else "none"
    evidence = build_evidence_summary(
        predicted_label=compact_label(prediction),
        prob_real=float(prediction["prob_real"]),
        prob_ai=float(prediction["prob_ai"]),
        confidence=float(prediction["confidence"]),
        uncertainty=float(prediction["uncertainty"]),
        action=str(prediction["action"]),
        threshold=float(threshold),
        heatmap_summary=build_heatmap_summary(heatmap, heatmap_error),
        selected_backbone=selected_backbone,
    )
    explanation, warning = generate_explanation_with_status(evidence, provider=provider)
    if warning and explanation_mode == "LLM-assisted":
        st.warning(warning)
    st.info(explanation)
    st.caption("The explanation uses only structured prediction metadata. The uploaded image is not sent to an LLM.")


def render_experiment_results() -> None:
    st.divider()
    st.subheader("Experiment Results")

    results_df = read_csv_safely(RESULTS_CSV)
    if results_df is None:
        st.info(f"No model performance table found yet. Run training or evaluation to create {RESULTS_CSV}.")
    elif results_df.empty:
        st.info(f"{RESULTS_CSV} exists but contains no rows yet.")
    else:
        st.write("Model performance comparison")
        st.dataframe(results_df, use_container_width=True)

    report_df = read_csv_safely(IMPROVEMENT_REPORT_CSV)
    if report_df is None:
        st.info(f"No improvement report found yet. Run `python compare_results.py` to create {IMPROVEMENT_REPORT_CSV}.")
        return
    if report_df.empty:
        st.info(f"{IMPROVEMENT_REPORT_CSV} exists but contains no rows yet.")
        return

    st.write("Dataset-wise performance improvement")
    st.dataframe(report_df, use_container_width=True)

    average_rows = report_df[report_df["dataset_name"] == "AVERAGE"] if "dataset_name" in report_df.columns else pd.DataFrame()
    if average_rows.empty or "improvement_percent" not in report_df.columns:
        st.info("Average improvement row is not available in the report.")
        return

    try:
        average = float(average_rows.iloc[0]["improvement_percent"])
        st.metric("Average performance improvement", f"{average:.2f}%")
        c1, c2, c3 = st.columns(3)
        c1.metric("Passed 5% threshold", "Yes" if average >= 5.0 else "No")
        c2.metric("Passed 10% threshold", "Yes" if average >= 10.0 else "No")
        c3.metric("Passed 20% threshold", "Yes" if average >= 20.0 else "No")
    except Exception as exc:
        st.info(f"Average improvement could not be parsed: {exc}")


def render_feedback_section() -> None:
    prediction = st.session_state.get("prediction")
    filename = st.session_state.get("prediction_filename")
    dataset_name = st.session_state.get("prediction_dataset_name")
    model_name = st.session_state.get("prediction_model_name")
    if not prediction or not filename or not dataset_name or not model_name:
        return

    st.divider()
    st.subheader("User Feedback")
    st.write("Do you think this prediction is correct?")
    st.caption("The uploaded image itself is not saved. Only prediction metadata and your feedback are stored.")

    b1, b2, b3 = st.columns(3)
    feedback_clicked = None
    if b1.button("Correct", use_container_width=True):
        feedback_clicked = "Correct"
    if b2.button("Wrong", use_container_width=True):
        feedback_clicked = "Wrong"
    if b3.button("Not sure", use_container_width=True):
        feedback_clicked = "Not sure"

    if feedback_clicked:
        try:
            save_feedback(
                feedback_value=feedback_clicked,
                filename=filename,
                dataset_name=dataset_name,
                model_name=model_name,
                prediction=prediction,
            )
            st.success(f"Feedback saved to {FEEDBACK_CSV}.")
        except Exception as exc:
            st.error(f"Could not save feedback: {exc}")


st.title(APP_TITLE)
st.markdown(
    "This research demo estimates whether an uploaded image is AI-generated or a real human-created / real-world image. "
    "It combines foundation-model features with evidential uncertainty so the result includes probabilities, confidence, uncertainty, and a review recommendation."
)
st.info(
    "Binary task: `AI-generated` means the image appears synthetic; `Real` means the image appears human-created or captured from the real world."
)

with st.sidebar:
    st.header("Demo Controls")
    dataset_name = st.selectbox("Dataset", SUPPORTED_DATASETS, index=0)
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Auto Best Backbone", "Manual Model Selection"],
        index=0,
    )
    best_backbone_config = None
    if analysis_mode == "Auto Best Backbone":
        best_backbone_config = read_best_backbone_config_safely()
        if best_backbone_config is None:
            st.warning("Best backbone has not been selected yet. Please run select_best_backbone.py first.")
            model_name = st.selectbox("Fallback model", SUPPORTED_MODELS, index=3, key="fallback_model_name")
        else:
            selected_best_model = str(best_backbone_config.get("best_model_name", ""))
            if selected_best_model not in SUPPORTED_MODELS:
                st.warning(
                    "The selected best backbone config contains an unsupported model. "
                    "Please rerun select_best_backbone.py or use manual selection."
                )
                model_name = st.selectbox("Fallback model", SUPPORTED_MODELS, index=3, key="invalid_config_fallback_model")
            else:
                model_name = selected_best_model
                st.success(f"Selected Best Backbone: {model_name}")
                st.write(f"Selection Metric: `{best_backbone_config.get('selection_metric', 'n/a')}`")
                st.write(f"Selected Score: `{format_score(best_backbone_config.get('selected_score'))}`")
    else:
        model_name = st.selectbox("Model", SUPPORTED_MODELS, index=3, key="manual_model_name")
    base_backbone_name = "auto"
    effective_base_backbone_name = model_name
    if model_name == "proposed_mnff_edl":
        if best_backbone_config is None:
            best_backbone_config = read_best_backbone_config_safely()
        base_backbone_name = st.selectbox(
            "Base Backbone",
            ["auto", "dinov3_linear", "siglip2_linear", "aimv2_linear"],
            index=0,
        )
        effective_base_backbone_name = resolve_base_backbone_for_ui(base_backbone_name, best_backbone_config)
        if base_backbone_name == "auto":
            if effective_base_backbone_name == "auto":
                st.warning(
                    "Auto base backbone needs results/best_backbone_config.json. "
                    "Run select_best_backbone.py first or choose a base backbone manually."
                )
            else:
                st.info(f"Proposed model base backbone: {effective_base_backbone_name}")
        else:
            st.info(f"Proposed model base backbone: {base_backbone_name}")
    image_size = st.selectbox("Image size", [224, 256, 384], index=[224, 256, 384].index(DEFAULT_IMAGE_SIZE))
    uncertainty_threshold = st.slider(
        "Uncertainty threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_UNCERTAINTY_THRESHOLD,
        step=0.01,
    )
    show_heatmap = st.checkbox("Show heatmap", value=True)
    show_experiment_results = st.checkbox("Show experiment results", value=True)
    show_user_feedback = st.checkbox("Show user feedback section", value=True)
    explanation_mode = st.selectbox("Explanation mode", ["Rule-based", "LLM-assisted"], index=0)
    llm_provider = st.selectbox("LLM provider", ["none", "openai", "gemini", "ollama"], index=0)
    device_arg = "auto"
    resolved_device = get_device(device_arg)
    st.caption(f"Device: {resolved_device}")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "prediction_image" not in st.session_state:
    st.session_state.prediction_image = None
if "prediction_filename" not in st.session_state:
    st.session_state.prediction_filename = None
if "heatmap" not in st.session_state:
    st.session_state.heatmap = None
if "heatmap_error" not in st.session_state:
    st.session_state.heatmap_error = None
if "prediction_dataset_name" not in st.session_state:
    st.session_state.prediction_dataset_name = None
if "prediction_model_name" not in st.session_state:
    st.session_state.prediction_model_name = None
if "prediction_analysis_mode" not in st.session_state:
    st.session_state.prediction_analysis_mode = None
if "prediction_selected_backbone" not in st.session_state:
    st.session_state.prediction_selected_backbone = None
if "prediction_base_backbone_name" not in st.session_state:
    st.session_state.prediction_base_backbone_name = None
if "prediction_threshold" not in st.session_state:
    st.session_state.prediction_threshold = None
if "prediction_explanation_mode" not in st.session_state:
    st.session_state.prediction_explanation_mode = None
if "prediction_llm_provider" not in st.session_state:
    st.session_state.prediction_llm_provider = None

st.subheader("Image Prediction")
left, right = st.columns([0.95, 1.05])

with left:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    image = None
    if uploaded_file is not None:
        if (
            st.session_state.get("prediction") is not None
            and st.session_state.get("prediction_filename") != uploaded_file.name
        ):
            st.session_state.prediction = None
            st.session_state.prediction_image = None
            st.session_state.prediction_filename = None
            st.session_state.prediction_dataset_name = None
            st.session_state.prediction_model_name = None
            st.session_state.prediction_analysis_mode = None
            st.session_state.prediction_selected_backbone = None
            st.session_state.prediction_base_backbone_name = None
            st.session_state.prediction_threshold = None
            st.session_state.prediction_explanation_mode = None
            st.session_state.prediction_llm_provider = None
            st.session_state.heatmap = None
            st.session_state.heatmap_error = None
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")

with right:
    st.write("Choose a trained dataset/model pair, then run prediction.")
    st.caption(f"Analysis model: {model_name}")
    if model_name == "proposed_mnff_edl":
        st.caption(f"Base backbone: {effective_base_backbone_name}")
    predict_clicked = st.button("Predict", type="primary", disabled=image is None, use_container_width=True)

    if predict_clicked and image is not None and uploaded_file is not None:
        st.session_state.prediction = None
        st.session_state.prediction_image = image
        st.session_state.prediction_filename = uploaded_file.name
        st.session_state.prediction_dataset_name = dataset_name
        st.session_state.prediction_model_name = model_name
        st.session_state.prediction_analysis_mode = analysis_mode
        st.session_state.prediction_base_backbone_name = base_backbone_name
        st.session_state.prediction_selected_backbone = (
            effective_base_backbone_name if model_name == "proposed_mnff_edl" else model_name
        )
        st.session_state.prediction_threshold = uncertainty_threshold
        st.session_state.prediction_explanation_mode = explanation_mode
        st.session_state.prediction_llm_provider = llm_provider
        st.session_state.heatmap = None
        st.session_state.heatmap_error = None

        model = None
        try:
            with st.spinner("Loading model..."):
                model = load_cached_model(
                    dataset_name,
                    model_name,
                    image_size,
                    device_arg,
                    _weights_mtime(dataset_name, model_name),
                    base_backbone_name,
                    _best_backbone_config_mtime(),
                )
        except Exception as exc:
            st.error(
                "Model loading failed. Train the selected dataset/model first and confirm the checkpoint exists at "
                f"{weights_path(dataset_name, model_name)}. Details: {exc}"
            )

        if model is not None:
            try:
                with st.spinner("Running prediction..."):
                    st.session_state.prediction = predict_pil_image(
                        model,
                        image,
                        image_size=image_size,
                        device_arg=device_arg,
                        uncertainty_threshold=uncertainty_threshold,
                    )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

            if st.session_state.prediction and show_heatmap:
                try:
                    with st.spinner("Generating visual explanation..."):
                        st.session_state.heatmap = generate_heatmap(
                            model,
                            image,
                            image_size=image_size,
                            device_arg=device_arg,
                            class_index=int(st.session_state.prediction["class_index"]),
                        )
                except Exception as exc:
                    st.session_state.heatmap_error = str(exc)

prediction = st.session_state.get("prediction")
prediction_image = st.session_state.get("prediction_image")

if prediction:
    result_threshold = st.session_state.get("prediction_threshold")
    if result_threshold is None:
        result_threshold = uncertainty_threshold
    render_result_card(prediction, float(result_threshold))
    render_user_friendly_explanation(
        prediction,
        float(result_threshold),
        st.session_state.get("prediction_explanation_mode") or explanation_mode,
        st.session_state.get("prediction_llm_provider") or llm_provider,
        st.session_state.get("heatmap"),
        st.session_state.get("heatmap_error"),
        st.session_state.get("prediction_selected_backbone") or model_name,
    )

    if show_heatmap and prediction_image is not None:
        render_explanation_section(
            prediction_image,
            st.session_state.get("heatmap"),
            st.session_state.get("heatmap_error"),
        )
    elif not show_heatmap:
        st.info("Heatmap display is turned off in the sidebar.")

    if show_user_feedback:
        render_feedback_section()

if show_experiment_results:
    render_experiment_results()
