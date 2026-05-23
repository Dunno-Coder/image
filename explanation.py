import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


SUPPORTED_LLM_PROVIDERS = ("none", "openai", "gemini", "ollama")
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


class ExplanationProviderUnavailable(RuntimeError):
    """Raised when an external LLM provider cannot be used safely."""


def _pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _user_label(predicted_label: str) -> str:
    normalized = predicted_label.lower()
    if "ai" in normalized:
        return "AI 생성 이미지"
    return "실제 이미지"


def _action_text(action: str, is_high_uncertainty: bool) -> str:
    if action == "manual_review" or is_high_uncertainty:
        return "사람이 한 번 더 확인하는 수동 검토 권장"
    if action == "auto_decision":
        return "현재 기준에서는 자동 판정 가능"
    return str(action)


def build_evidence_summary(
    predicted_label: str,
    prob_real: float,
    prob_ai: float,
    confidence: float,
    uncertainty: float,
    action: str,
    threshold: float,
    heatmap_summary: str | None = None,
    selected_backbone: str | None = None,
) -> dict:
    is_high_uncertainty = float(uncertainty) > float(threshold)
    return {
        "predicted_label": predicted_label,
        "display_label": _user_label(predicted_label),
        "prob_real": float(prob_real),
        "prob_ai": float(prob_ai),
        "prob_real_percent": _pct(prob_real),
        "prob_ai_percent": _pct(prob_ai),
        "confidence": float(confidence),
        "confidence_percent": _pct(confidence),
        "uncertainty": float(uncertainty),
        "uncertainty_percent": _pct(uncertainty),
        "uncertainty_threshold": float(threshold),
        "uncertainty_threshold_percent": _pct(threshold),
        "is_high_uncertainty": is_high_uncertainty,
        "action": action,
        "recommended_action_text": _action_text(action, is_high_uncertainty),
        "heatmap_summary": heatmap_summary,
        "selected_backbone": selected_backbone,
        "model_name": selected_backbone,
        "safety_note": (
            "이 결과는 모델 기반 추정이며 100% 확정 판정이 아닙니다. "
            "중요한 판단에는 사람이 함께 검토해야 합니다."
        ),
    }


def build_llm_prompt(evidence: dict) -> str:
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    return f"""
아래는 AI 생성 이미지 탐지 모델이 만든 구조화된 XAI 결과입니다.
업로드된 이미지 원본은 너에게 제공되지 않습니다. 따라서 이미지에 보이는 사물,
장면, 사람, 텍스트를 새로 추측하지 말고, 제공된 수치와 heatmap 요약만 근거로 설명하세요.

목표:
- 일반 사용자가 이해할 수 있는 쉬운 한국어 설명을 작성합니다.
- 모델이 왜 이런 판단을 했는지 prob_ai, prob_real, confidence, uncertainty, heatmap 요약을 연결해 설명합니다.
- 결과가 확정 판정이 아니라 모델 기반 추정임을 분명히 말합니다.
- 불확실성이 높으면 왜 수동 검토가 필요한지 차분하게 설명합니다.

출력 형식:
1. 결과 요약
2. 수치 근거
3. 판단 이유
4. 시각적 근거
5. 권장 조치
6. 주의할 점

문체:
- 비전문가에게 설명하듯 친절하고 짧은 문장으로 작성합니다.
- 과장하지 말고 "가능성이 높다", "모델은 ...로 본다" 같은 조심스러운 표현을 씁니다.
- "확실히 AI", "100% 실제" 같은 단정 표현은 쓰지 않습니다.

XAI evidence:
{evidence_json}
""".strip()


def generate_rule_based_explanation(evidence: dict) -> str:
    label = evidence.get("display_label", evidence.get("predicted_label", "알 수 없음"))
    prob_ai = evidence.get("prob_ai_percent", _pct(evidence.get("prob_ai", 0.0)))
    prob_real = evidence.get("prob_real_percent", _pct(evidence.get("prob_real", 0.0)))
    confidence = evidence.get("confidence_percent", _pct(evidence.get("confidence", 0.0)))
    uncertainty = evidence.get("uncertainty_percent", _pct(evidence.get("uncertainty", 0.0)))
    threshold = evidence.get("uncertainty_threshold_percent", _pct(evidence.get("uncertainty_threshold", 0.0)))
    high_uncertainty = bool(evidence.get("is_high_uncertainty"))
    heatmap_summary = evidence.get("heatmap_summary")
    selected_backbone = evidence.get("selected_backbone") or evidence.get("model_name")
    action_text = evidence.get("recommended_action_text") or _action_text(
        str(evidence.get("action", "")),
        high_uncertainty,
    )

    backbone_sentence = (
        f"분석에는 `{selected_backbone}` 모델 결과가 사용되었습니다."
        if selected_backbone
        else "분석에는 현재 선택된 이미지 분류 모델 결과가 사용되었습니다."
    )

    prob_ai_value = float(evidence.get("prob_ai", 0.0))
    prob_real_value = float(evidence.get("prob_real", 0.0))
    if prob_ai_value >= prob_real_value + 0.20:
        reasoning_sentence = (
            "AI 확률이 실제 이미지 확률보다 뚜렷하게 높아서, 모델은 AI 생성 이미지 쪽 근거를 더 강하게 본 것으로 해석됩니다."
        )
    elif prob_real_value >= prob_ai_value + 0.20:
        reasoning_sentence = (
            "실제 이미지 확률이 AI 확률보다 뚜렷하게 높아서, 모델은 실제 이미지 쪽 근거를 더 강하게 본 것으로 해석됩니다."
        )
    else:
        reasoning_sentence = (
            "두 확률의 차이가 크지 않아, 이 판단은 비교적 조심스럽게 보는 것이 좋습니다."
        )

    if high_uncertainty:
        reasoning_sentence += (
            f" 또한 불확실성이 기준값 {threshold}보다 높아 모델 내부 증거가 충분히 안정적이지 않을 수 있습니다."
        )

    if heatmap_summary:
        visual_sentence = (
            f"시각적 설명에서는 {heatmap_summary} 이 영역은 모델 예측에 상대적으로 민감하게 작용한 부분으로 볼 수 있습니다. "
            "다만 heatmap은 단서 위치를 보여주는 보조 자료이며, 그 자체만으로 판정을 확정하지는 않습니다."
        )
    else:
        visual_sentence = (
            "이번 예측에서는 사용할 수 있는 heatmap 요약이 없습니다. 따라서 시각적 근거보다는 확률과 불확실성 수치를 중심으로 해석해야 합니다."
        )

    return (
        f"1. 결과 요약\n"
        f"모델은 이 이미지를 {label}일 가능성이 높다고 추정했습니다. {backbone_sentence}\n\n"
        f"2. 수치 근거\n"
        f"AI 이미지 확률은 {prob_ai}, 실제 이미지 확률은 {prob_real}입니다. "
        f"confidence는 {confidence}로 모델 판단의 강도를, uncertainty는 {uncertainty}로 판단의 불확실성을 나타냅니다.\n\n"
        f"3. 판단 이유\n"
        f"{reasoning_sentence}\n\n"
        f"4. 시각적 근거\n"
        f"{visual_sentence}\n\n"
        f"5. 권장 조치\n"
        f"현재 권장 조치는 `{action_text}`입니다. "
        f"불확실성 기준값은 {threshold}이며, 이 기준을 넘으면 사람이 추가로 확인하는 편이 안전합니다.\n\n"
        f"6. 주의할 점\n"
        f"{evidence.get('safety_note')}"
    )


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ExplanationProviderUnavailable(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ExplanationProviderUnavailable(str(exc.reason)) from exc


def _extract_openai_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    text = "\n".join(parts).strip()
    if not text:
        raise ExplanationProviderUnavailable("OpenAI 응답에서 설명 텍스트를 찾지 못했습니다.")
    return text


def _openai_explanation(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ExplanationProviderUnavailable("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    payload = {
        "model": model,
        "instructions": (
            "너는 이미지 판별 XAI 결과를 일반 사용자에게 설명하는 한국어 보조자입니다. "
            "이미지를 직접 보지 못한다는 한계를 지키고, 제공된 수치와 heatmap 요약만 근거로 설명하세요."
        ),
        "input": prompt,
        "max_output_tokens": 900,
        "store": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = _post_json(f"{base_url}/responses", payload, headers, timeout)
    return _extract_openai_text(response)


def _ollama_explanation(prompt: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    payload = {
        "model": model,
        "prompt": (
            "너는 이미지 판별 XAI 결과를 일반 사용자에게 설명하는 한국어 보조자입니다.\n"
            "이미지 원본은 보지 못하며, 제공된 수치와 heatmap 요약만 근거로 설명합니다.\n\n"
            f"{prompt}"
        ),
        "stream": False,
        "options": {"temperature": 0.2},
    }
    response = _post_json(
        f"{host}/api/generate",
        payload,
        {"Content-Type": "application/json"},
        timeout,
    )
    text = response.get("response")
    if not isinstance(text, str) or not text.strip():
        raise ExplanationProviderUnavailable("Ollama 응답에서 설명 텍스트를 찾지 못했습니다.")
    return text.strip()


def provider_configuration_status(provider: str) -> Tuple[bool, Optional[str]]:
    provider = provider.lower()
    if provider == "none":
        return True, None
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY가 설정되어 있지 않아 규칙 기반 설명을 사용합니다."
        return True, None
    if provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            return False, "GEMINI_API_KEY가 설정되어 있지 않아 규칙 기반 설명을 사용합니다."
        return False, "Gemini 연결은 아직 구현되지 않았습니다. 현재는 규칙 기반 설명을 사용합니다."
    if provider == "ollama":
        return True, None
    return False, f"지원하지 않는 LLM 제공자입니다: {provider}. 규칙 기반 설명을 사용합니다."


def generate_llm_explanation(evidence: dict, provider: str = "none") -> str:
    provider = provider.lower()
    prompt = build_llm_prompt(evidence)

    if provider == "none":
        return generate_rule_based_explanation(evidence)
    if provider == "openai":
        return _openai_explanation(prompt)
    if provider == "ollama":
        return _ollama_explanation(prompt)
    if provider == "gemini":
        raise ExplanationProviderUnavailable("Gemini 연결은 아직 구현되지 않았습니다.")
    raise ExplanationProviderUnavailable(f"지원하지 않는 provider입니다: {provider}")


def generate_explanation_with_status(evidence: Dict[str, Any], provider: str = "none") -> Tuple[str, Optional[str]]:
    configured, warning = provider_configuration_status(provider)
    if provider == "none" or configured:
        try:
            return generate_llm_explanation(evidence, provider), warning
        except Exception as exc:
            fallback = generate_rule_based_explanation(evidence)
            return fallback, f"LLM 설명 생성에 실패하여 규칙 기반 설명으로 대체했습니다: {exc}"
    return generate_rule_based_explanation(evidence), warning
