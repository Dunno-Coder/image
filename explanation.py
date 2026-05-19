import os
from typing import Any, Dict, Optional, Tuple


SUPPORTED_LLM_PROVIDERS = ("none", "openai", "gemini", "ollama")


class ExplanationProviderUnavailable(RuntimeError):
    """Raised when an external LLM provider cannot be used safely."""


def _pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _user_label(predicted_label: str) -> str:
    normalized = predicted_label.lower()
    if "ai" in normalized:
        return "AI 생성 이미지"
    return "실제 이미지"


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
        "recommended_action_text": (
            "Manual review is recommended because uncertainty is high."
            if is_high_uncertainty
            else "Auto decision is allowed."
        ),
        "heatmap_summary": heatmap_summary,
        "selected_backbone": selected_backbone,
        "model_name": selected_backbone,
        "safety_note": (
            "This is a model-based estimate, not a 100% certain judgment. "
            "Important decisions should include human review."
        ),
    }


def build_llm_prompt(evidence: dict) -> str:
    return f"""
당신은 AI 생성 이미지 탐지 시스템의 설명 보조자입니다.
중요: 당신은 이미지 분류기가 아닙니다. 아래 구조화된 모델 결과만 사용해서 설명하세요.
이미지 자체는 제공되지 않았으며, 새로운 시각적 사실을 추측하면 안 됩니다.

반드시 아래 6개 섹션 구조를 따르세요.

1. 결과 요약
- 이미지가 AI 생성 이미지 또는 실제 이미지 중 무엇으로 예측되었는지 말하세요.

2. 수치 근거
- prob_ai, prob_real, confidence, uncertainty를 모두 퍼센트로 설명하세요.
- confidence는 "모델 판단의 강도", uncertainty는 "판단의 불확실성"처럼 쉽게 설명하세요.

3. 판단 이유
- 숫자가 예측을 어떻게 뒷받침하는지 설명하세요.
- prob_ai가 prob_real보다 훨씬 높으면 "AI 생성 패턴에 대한 근거가 더 강하게 나타난 것으로 보입니다"라고 조심스럽게 말하세요.
- uncertainty가 높으면 자동 판정에는 충분히 확신하기 어렵다고 설명하세요.

4. 시각적 근거
- heatmap_summary가 있으면 모델이 어느 영역에 상대적으로 주목한 것으로 보이는지 설명하세요.
- heatmap_summary가 없으면 시각적 설명은 제공되지 않았다고 간단히 말하세요.
- 항상 조심스럽게 표현하세요.

5. 권장 조치
- action이 auto_decision이면 현재 threshold 기준에서 자동 판정이 가능하다고 설명하세요.
- action이 manual_review이면 사람이 추가로 검토하는 것이 좋다고 설명하세요.

6. 주의
- 항상 이 결과가 완벽한 포렌식 판정이 아니라 모델 기반 추정이라고 말하세요.
- "확실히 AI 생성", "확실히 실제" 같은 단정 표현을 쓰지 마세요.

문체:
- 한국어
- 명확하고 비전문가에게 친절한 표현
- 과도한 AI 전문용어 금지
- 사용된 selected_backbone을 짧고 비전문적인 문장으로 언급
- "시스템은 ...로 추정합니다", "모델은 ...에 주목한 것으로 보입니다", "이는 ...일 수 있습니다" 같은 신중한 표현 사용

구조화된 evidence:
- predicted_label: {evidence.get("display_label", evidence.get("predicted_label"))}
- prob_ai: {evidence.get("prob_ai_percent")}
- prob_real: {evidence.get("prob_real_percent")}
- confidence: {evidence.get("confidence_percent")}
- uncertainty: {evidence.get("uncertainty_percent")}
- uncertainty_threshold: {evidence.get("uncertainty_threshold_percent")}
- action: {evidence.get("action")}
- recommended_action_text: {evidence.get("recommended_action_text")}
- heatmap_summary: {evidence.get("heatmap_summary") or "없음"}
- selected_backbone: {evidence.get("selected_backbone") or evidence.get("model_name") or "없음"}
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
    action = evidence.get("action", "")
    selected_backbone = evidence.get("selected_backbone") or evidence.get("model_name")
    backbone_sentence = (
        f"이 분석은 사전에 평가된 모델 중 선택된 {selected_backbone} 백본을 사용해 수행되었습니다."
        if selected_backbone
        else "이 분석에는 현재 선택된 이미지 분류 모델이 사용되었습니다."
    )

    if float(evidence.get("prob_ai", 0.0)) >= float(evidence.get("prob_real", 0.0)) + 0.20:
        reasoning_sentence = "AI 이미지 확률이 실제 이미지 확률보다 훨씬 높아, 모델은 AI 생성 패턴에 대한 근거를 더 강하게 본 것으로 해석할 수 있습니다."
    elif float(evidence.get("prob_real", 0.0)) >= float(evidence.get("prob_ai", 0.0)) + 0.20:
        reasoning_sentence = "실제 이미지 확률이 AI 이미지 확률보다 훨씬 높아, 모델은 실제 이미지에 가까운 근거를 더 강하게 본 것으로 해석할 수 있습니다."
    else:
        reasoning_sentence = "두 확률의 차이가 크지 않기 때문에, 모델 판단은 비교적 조심스럽게 해석하는 것이 좋습니다."

    if high_uncertainty:
        reasoning_sentence += " 또한 uncertainty가 높아 자동 판정을 내리기에는 모델이 충분히 안정적이지 않을 수 있습니다."

    visual_sentence = (
        f"heatmap 또는 saliency 결과에서는 {heatmap_summary} 이 영역은 모델이 상대적으로 민감하게 반응한 부분으로 볼 수 있지만, 이것만으로 원인을 단정할 수는 없습니다."
        if heatmap_summary
        else "이번 예측에서는 사용할 수 있는 heatmap 요약이 없어서, 시각적 근거는 별도로 제시되지 않았습니다."
    )

    action_sentence = (
        f"현재 uncertainty가 설정 기준({threshold})보다 높기 때문에, 시스템은 사람이 추가로 확인하는 수동 검토를 권장합니다."
        if high_uncertainty
        else f"현재 uncertainty가 설정 기준({threshold}) 이하이므로, 시스템은 이 threshold 기준에서 자동 판정이 가능하다고 판단했습니다."
    )
    if action == "manual_review":
        action_sentence = f"시스템의 권장 조치는 manual_review입니다. {action_sentence}"
    elif action == "auto_decision":
        action_sentence = f"시스템의 권장 조치는 auto_decision입니다. {action_sentence}"

    return (
        f"1. 결과 요약\n"
        f"시스템은 이 이미지를 {label}로 추정합니다. {backbone_sentence}\n\n"
        f"2. 수치 근거\n"
        f"AI 이미지 확률은 {prob_ai}, 실제 이미지 확률은 {prob_real}입니다. "
        f"confidence는 {confidence}로 모델 판단의 강도를 나타내고, uncertainty는 {uncertainty}로 판단의 불확실성을 의미합니다.\n\n"
        f"3. 판단 이유\n"
        f"{reasoning_sentence}\n\n"
        f"4. 시각적 근거\n"
        f"{visual_sentence}\n\n"
        f"5. 권장 조치\n"
        f"{action_sentence}\n\n"
        f"6. 주의\n"
        "이 결과는 완벽한 포렌식 판정이 아니라 모델 기반 추정입니다. "
        "따라서 중요한 상황에서는 추가 검토가 필요하며, 이 설명을 확정적인 사실로 받아들이면 안 됩니다."
    )


def provider_configuration_status(provider: str) -> Tuple[bool, Optional[str]]:
    provider = provider.lower()
    if provider == "none":
        return True, None
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not configured. Using the rule-based explanation."
        return False, "OpenAI wrapper is currently a safe placeholder. No external request was sent."
    if provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            return False, "GEMINI_API_KEY is not configured. Using the rule-based explanation."
        return False, "Gemini wrapper is currently a safe placeholder. No external request was sent."
    if provider == "ollama":
        if not os.getenv("OLLAMA_HOST"):
            return False, "OLLAMA_HOST is not configured. Using the rule-based explanation."
        return False, "Ollama wrapper is currently a safe placeholder. No external request was sent."
    return False, f"Unsupported LLM provider '{provider}'. Using the rule-based explanation."


def _openai_placeholder(prompt: str) -> str:
    del prompt
    if not os.getenv("OPENAI_API_KEY"):
        raise ExplanationProviderUnavailable("OPENAI_API_KEY is not configured.")
    raise ExplanationProviderUnavailable("OpenAI provider is a placeholder and did not call an external API.")


def _gemini_placeholder(prompt: str) -> str:
    del prompt
    if not os.getenv("GEMINI_API_KEY"):
        raise ExplanationProviderUnavailable("GEMINI_API_KEY is not configured.")
    raise ExplanationProviderUnavailable("Gemini provider is a placeholder and did not call an external API.")


def _ollama_placeholder(prompt: str) -> str:
    del prompt
    if not os.getenv("OLLAMA_HOST"):
        raise ExplanationProviderUnavailable("OLLAMA_HOST is not configured.")
    raise ExplanationProviderUnavailable("Ollama provider is a placeholder and did not call an external API.")


def generate_llm_explanation(evidence: dict, provider: str = "none") -> str:
    provider = provider.lower()
    prompt = build_llm_prompt(evidence)

    if provider == "none":
        return generate_rule_based_explanation(evidence)

    try:
        if provider == "openai":
            return _openai_placeholder(prompt)
        if provider == "gemini":
            return _gemini_placeholder(prompt)
        if provider == "ollama":
            return _ollama_placeholder(prompt)
        raise ExplanationProviderUnavailable(f"Unsupported provider: {provider}")
    except Exception:
        return generate_rule_based_explanation(evidence)


def generate_explanation_with_status(evidence: Dict[str, Any], provider: str = "none") -> Tuple[str, Optional[str]]:
    configured, warning = provider_configuration_status(provider)
    if provider == "none" or configured:
        try:
            return generate_llm_explanation(evidence, provider), warning
        except Exception as exc:
            return generate_rule_based_explanation(evidence), f"LLM explanation failed: {exc}"
    return generate_rule_based_explanation(evidence), warning
