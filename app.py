import csv
import json
import os
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


APP_TITLE = "AI 생성 이미지 탐지 시스템"

ANALYSIS_MODE_LABELS = {
    "Auto Best Backbone": "자동 Best Backbone 사용",
    "Manual Model Selection": "수동 모델 선택",
}
EXPLANATION_MODE_LABELS = {
    "Rule-based": "규칙 기반 설명",
    "LLM-assisted": "LLM 보조 설명",
}
LLM_PROVIDER_LABELS = {
    "none": "사용 안 함",
    "openai": "OpenAI API",
    "ollama": "Ollama 로컬",
}


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
        st.error(f"{path} 파일을 불러오지 못했습니다: {exc}")
        return None


def read_best_backbone_config_safely() -> Optional[Dict[str, object]]:
    try:
        if not BEST_BACKBONE_CONFIG_JSON.exists():
            return None
        return json.loads(BEST_BACKBONE_CONFIG_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"{BEST_BACKBONE_CONFIG_JSON} 파일을 불러오지 못했습니다: {exc}")
        return None


def format_score(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def llm_status_text(provider: str) -> str:
    if provider == "openai":
        model_name = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        if os.getenv("OPENAI_API_KEY"):
            return f"OpenAI API 연결 준비됨 · 모델: {model_name}"
        return f"OPENAI_API_KEY 미설정 · 키를 설정하면 OpenAI API로 설명을 생성합니다. 기본 모델: {model_name}"
    if provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return f"Ollama 로컬 연결 사용 · {host} · 모델: {model_name}"
    if provider == "gemini":
        return "Gemini는 UI 옵션만 준비되어 있고, 현재 구현은 규칙 기반 설명으로 대체됩니다."
    return "규칙 기반 설명만 사용합니다."


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


def display_label(result: Dict[str, object]) -> str:
    return "AI 생성 이미지" if int(result["class_index"]) == 1 else "실제/사람 생성 이미지"


def display_action(action: object) -> str:
    if str(action) == "auto_decision":
        return "자동 판정 가능"
    if str(action) == "manual_review":
        return "수동 검토 권장"
    return str(action)


def action_message(result: Dict[str, object], threshold: float) -> None:
    uncertainty = float(result["uncertainty"])
    if uncertainty <= threshold:
        st.success("현재 불확실성이 기준 이하이므로 자동 판정이 가능합니다.")
    else:
        st.warning("불확실성이 높아 사람이 추가로 검토하는 것을 권장합니다.")


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
    label = display_label(result)
    label_help = (
        "모델은 이 이미지가 AI로 생성되었을 가능성이 높다고 추정합니다."
        if int(result["class_index"]) == 1
        else "모델은 이 이미지가 실제 이미지 또는 사람이 만든 이미지일 가능성이 높다고 추정합니다."
    )

    with st.container(border=True):
        st.subheader("예측 결과")
        st.metric("예측 라벨", label)
        st.caption(label_help)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AI 확률", f"{float(result['prob_ai']):.2%}")
        m2.metric("실제 이미지 확률", f"{float(result['prob_real']):.2%}")
        m3.metric("신뢰도", f"{float(result['confidence']):.2%}")
        m4.metric("불확실성", f"{float(result['uncertainty']):.2%}")

        p1, p2 = st.columns(2)
        with p1:
            display_progress("AI 확률", float(result["prob_ai"]))
            display_progress("신뢰도", float(result["confidence"]))
        with p2:
            display_progress("실제 이미지 확률", float(result["prob_real"]))
            display_progress("불확실성", float(result["uncertainty"]))

        st.metric("권장 조치", display_action(result["action"]))
        action_message(result, threshold)


def render_explanation_section(image: Image.Image, heatmap: Optional[Image.Image], heatmap_error: Optional[str]) -> None:
    st.subheader("시각적 설명")
    c1, c2 = st.columns(2)
    c1.image(image, caption="원본 이미지", use_container_width=True)
    if heatmap is not None:
        c2.image(heatmap, caption="Grad-CAM / Saliency Heatmap", use_container_width=True)
    elif heatmap_error:
        c2.info(f"이 모델/이미지에서는 heatmap을 생성하지 못했습니다. {heatmap_error}")
    else:
        c2.info("heatmap은 생성되지 않았지만 예측은 정상적으로 완료되었습니다.")


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
    st.subheader("사용자 친화적 설명")
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
    st.caption("이 설명은 구조화된 예측 메타데이터만 사용합니다. 업로드한 이미지 자체는 LLM으로 전송되지 않습니다.")


def render_experiment_results() -> None:
    st.divider()
    st.subheader("실험 결과")

    results_df = read_csv_safely(RESULTS_CSV)
    if results_df is None:
        st.info(f"아직 모델 성능표가 없습니다. 학습 또는 평가를 실행하면 {RESULTS_CSV} 파일이 생성됩니다.")
    elif results_df.empty:
        st.info(f"{RESULTS_CSV} 파일은 있지만 아직 기록된 행이 없습니다.")
    else:
        st.write("모델 성능 비교표")
        st.dataframe(results_df, use_container_width=True)

    report_df = read_csv_safely(IMPROVEMENT_REPORT_CSV)
    if report_df is None:
        st.info(f"아직 개선율 리포트가 없습니다. `python compare_results.py`를 실행하면 {IMPROVEMENT_REPORT_CSV} 파일이 생성됩니다.")
        return
    if report_df.empty:
        st.info(f"{IMPROVEMENT_REPORT_CSV} 파일은 있지만 아직 기록된 행이 없습니다.")
        return

    st.write("데이터셋별 성능 개선율")
    st.dataframe(report_df, use_container_width=True)

    average_rows = report_df[report_df["dataset_name"] == "AVERAGE"] if "dataset_name" in report_df.columns else pd.DataFrame()
    if average_rows.empty or "improvement_percent" not in report_df.columns:
        st.info("리포트에서 평균 개선율 행을 찾을 수 없습니다.")
        return

    try:
        average = float(average_rows.iloc[0]["improvement_percent"])
        st.metric("평균 성능 개선율", f"{average:.2f}%")
        c1, c2, c3 = st.columns(3)
        c1.metric("5% 기준 통과", "예" if average >= 5.0 else "아니오")
        c2.metric("10% 기준 통과", "예" if average >= 10.0 else "아니오")
        c3.metric("20% 기준 통과", "예" if average >= 20.0 else "아니오")
    except Exception as exc:
        st.info(f"평균 개선율을 해석하지 못했습니다: {exc}")


def render_feedback_section() -> None:
    prediction = st.session_state.get("prediction")
    filename = st.session_state.get("prediction_filename")
    dataset_name = st.session_state.get("prediction_dataset_name")
    model_name = st.session_state.get("prediction_model_name")
    if not prediction or not filename or not dataset_name or not model_name:
        return

    st.divider()
    st.subheader("사용자 피드백")
    st.write("이 예측이 맞다고 생각하시나요?")
    st.caption("업로드한 이미지 자체는 저장하지 않습니다. 예측 메타데이터와 피드백만 저장됩니다.")

    b1, b2, b3 = st.columns(3)
    feedback_clicked = None
    if b1.button("맞음", use_container_width=True):
        feedback_clicked = "Correct"
    if b2.button("틀림", use_container_width=True):
        feedback_clicked = "Wrong"
    if b3.button("잘 모르겠음", use_container_width=True):
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
            st.success(f"피드백이 {FEEDBACK_CSV}에 저장되었습니다.")
        except Exception as exc:
            st.error(f"피드백을 저장하지 못했습니다: {exc}")


st.title(APP_TITLE)

st.info(
    "이 시스템은 AI 생성 이미지와 실제/사람 생성 이미지를 구분하는 시스템입니다."
)

with st.sidebar:
    st.header("데모 설정")
    dataset_name = st.selectbox("데이터셋", SUPPORTED_DATASETS, index=0)
    analysis_mode = st.selectbox(
        "분석 모드",
        ["Auto Best Backbone", "Manual Model Selection"],
        index=0,
        format_func=lambda value: ANALYSIS_MODE_LABELS[value],
    )
    best_backbone_config = None
    if analysis_mode == "Auto Best Backbone":
        best_backbone_config = read_best_backbone_config_safely()
        if best_backbone_config is None:
            st.warning("아직 최적 Backbone이 선택되지 않았습니다. 먼저 `select_best_backbone.py`를 실행하세요.")
            model_name = st.selectbox("대체 모델", SUPPORTED_MODELS, index=3, key="fallback_model_name")
        else:
            selected_best_model = str(best_backbone_config.get("best_model_name", ""))
            if selected_best_model not in SUPPORTED_MODELS:
                st.warning(
                    "최적 Backbone 설정 파일에 지원되지 않는 모델이 들어 있습니다. "
                    "`select_best_backbone.py`를 다시 실행하거나 수동 모델 선택을 사용하세요."
                )
                model_name = st.selectbox("대체 모델", SUPPORTED_MODELS, index=3, key="invalid_config_fallback_model")
            else:
                model_name = selected_best_model
                st.success(f"선택된 최적 Backbone: {model_name}")
                st.write(f"선택 기준 지표: `{best_backbone_config.get('selection_metric', 'n/a')}`")
                st.write(f"선택 점수: `{format_score(best_backbone_config.get('selected_score'))}`")
    else:
        model_name = st.selectbox("모델", SUPPORTED_MODELS, index=3, key="manual_model_name")
    base_backbone_name = "auto"
    effective_base_backbone_name = model_name
    if model_name == "proposed_mnff_edl":
        if best_backbone_config is None:
            best_backbone_config = read_best_backbone_config_safely()
        base_backbone_name = st.selectbox(
            "기반 Backbone",
            ["auto", "dinov3_linear", "siglip2_linear", "aimv2_linear"],
            index=0,
        )
        effective_base_backbone_name = resolve_base_backbone_for_ui(base_backbone_name, best_backbone_config)
        if base_backbone_name == "auto":
            if effective_base_backbone_name == "auto":
                st.warning(
                    "자동 Base Backbone을 사용하려면 `results/best_backbone_config.json`이 필요합니다. "
                    "먼저 `select_best_backbone.py`를 실행하거나 base backbone을 직접 선택하세요."
                )
            else:
                st.info(f"제안 모델 기반 Backbone: {effective_base_backbone_name}")
        else:
            st.info(f"제안 모델 기반 Backbone: {base_backbone_name}")
    image_size = st.selectbox("이미지 크기", [224, 256, 384], index=[224, 256, 384].index(DEFAULT_IMAGE_SIZE))
    uncertainty_threshold = st.slider(
        "불확실성 기준값",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_UNCERTAINTY_THRESHOLD,
        step=0.01,
    )
    show_heatmap = st.checkbox("Heatmap 표시", value=True)
    show_experiment_results = st.checkbox("실험 결과 표시", value=True)
    show_user_feedback = st.checkbox("사용자 피드백 섹션 표시", value=True)
    provider_options = ["openai", "ollama", "none"]
    explanation_mode = st.selectbox(
        "설명 방식",
        ["Rule-based", "LLM-assisted"],
        index=1,
        format_func=lambda value: EXPLANATION_MODE_LABELS[value],
    )
    llm_provider = st.selectbox(
        "LLM 제공자",
        provider_options,
        index=0,
        format_func=lambda value: LLM_PROVIDER_LABELS[value],
    )
    if llm_provider == "openai":
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value="",
            placeholder="환경변수로 설정했으면 비워두세요",
        )
        openai_model = st.text_input("OpenAI 모델", value=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
        if openai_api_key.strip():
            os.environ["OPENAI_API_KEY"] = openai_api_key.strip()
        if openai_model.strip():
            os.environ["OPENAI_MODEL"] = openai_model.strip()
    if llm_provider == "ollama":
        ollama_host = st.text_input("Ollama Host", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        ollama_model = st.text_input("Ollama 모델", value=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
        if ollama_host.strip():
            os.environ["OLLAMA_HOST"] = ollama_host.strip()
        if ollama_model.strip():
            os.environ["OLLAMA_MODEL"] = ollama_model.strip()
    st.caption(f"LLM 상태: {llm_status_text(llm_provider)}")
    if explanation_mode == "LLM-assisted":
        if llm_provider == "none":
            st.warning("LLM 보조 설명을 쓰려면 LLM 제공자를 OpenAI API 또는 Ollama 로컬로 선택하세요.")
        else:
            st.caption("LLM에는 예측 수치와 heatmap 요약만 전달되며, 업로드한 이미지 원본은 전송하지 않습니다.")
    device_arg = "auto"
    resolved_device = get_device(device_arg)
    st.caption(f"사용 장치: {resolved_device}")

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

st.subheader("이미지 예측")
left, right = st.columns([0.95, 1.05])

with left:
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "webp", "bmp"])
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
            st.image(image, caption="업로드한 이미지", use_container_width=True)
        except Exception as exc:
            st.error(f"업로드한 이미지를 읽지 못했습니다: {exc}")

with right:
    st.write("학습된 데이터셋/모델 조합을 선택한 뒤 예측을 실행하세요.")
    st.caption(f"분석 모델: {model_name}")
    if model_name == "proposed_mnff_edl":
        st.caption(f"기반 Backbone: {effective_base_backbone_name}")
    predict_clicked = st.button("예측 실행", type="primary", disabled=image is None, use_container_width=True)

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
            with st.spinner("모델을 불러오는 중..."):
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
                "모델을 불러오지 못했습니다. 선택한 데이터셋/모델을 먼저 학습했는지 확인하고, checkpoint가 존재하는지 확인하세요: "
                f"{weights_path(dataset_name, model_name)}. 상세 정보: {exc}"
            )

        if model is not None:
            try:
                with st.spinner("예측을 실행하는 중..."):
                    st.session_state.prediction = predict_pil_image(
                        model,
                        image,
                        image_size=image_size,
                        device_arg=device_arg,
                        uncertainty_threshold=uncertainty_threshold,
                    )
            except Exception as exc:
                st.error(f"예측에 실패했습니다: {exc}")

            if st.session_state.prediction and show_heatmap:
                try:
                    with st.spinner("시각적 설명을 생성하는 중..."):
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
        st.info("사이드바에서 Heatmap 표시가 꺼져 있습니다.")

    if show_user_feedback:
        render_feedback_section()

if show_experiment_results:
    render_experiment_results()
