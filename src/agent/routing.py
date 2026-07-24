"""Deterministic prompt-complexity classification and model selection."""

from dataclasses import dataclass
from typing import Literal

from ..config import Settings, settings

Complexity = Literal["simple", "medium", "complex"]


@dataclass(frozen=True)
class RoutingDecision:
    """Stable routing metadata shared by the API and downstream cost services."""

    complexity: Complexity
    model_profile: str
    display_name: str
    provider_model_id: str
    reason: str
    matched_rules: list[str]
    prompt_length: int


COMPLEX_RULES = {
    "architecture": ("system architecture", "technical design", "rag architecture"),
    "security": ("security analysis", "secure", "rbac", "threat model"),
    "scalability": ("scalable", "scalability", "infrastructure", "monitoring"),
    "complex_debugging": ("debug a complex", "distributed system", "root cause"),
    "multiple_integrations": ("multiple integrations", "multiple data sources"),
    "multi_step_planning": ("multi-step", "step-by-step plan", "implementation plan"),
}
MEDIUM_RULES = {
    "comparison": ("compare", "comparison", "versus", " vs "),
    "analysis": ("analyze", "analysis", "evaluate", "trade-offs", "tradeoffs"),
    "explanation": ("explain", "multiple points", "advantages", "disadvantages"),
    "structured_content": ("create a table", "structured", "outline", "bullet points"),
    "moderate_coding": ("write a function", "code example", "implement"),
    "multiple_conditions": ("requirements", "conditions", "criteria"),
}
SIMPLE_RULES = {
    "translation": ("translate", "translation"),
    "rewrite": ("rewrite", "rephrase", "paraphrase"),
    "grammar": ("grammar", "proofread", "correct this sentence"),
    "short_summary": ("summarize", "summary"),
    "formatting": ("format this", "capitalize", "convert to"),
}


def _matches(prompt: str, rules: dict[str, tuple[str, ...]]) -> list[str]:
    return [name for name, phrases in rules.items() if any(p in prompt for p in phrases)]


def route_prompt(prompt: str, config: Settings = settings) -> RoutingDecision:
    """Classify a prompt using priority-ordered, deterministic rules."""
    normalized = f" {prompt.casefold().strip()} "
    prompt_length = len(prompt)
    complex_matches = _matches(normalized, COMPLEX_RULES)
    medium_matches = _matches(normalized, MEDIUM_RULES)
    simple_matches = _matches(normalized, SIMPLE_RULES)

    if complex_matches:
        complexity: Complexity = "complex"
        matched = complex_matches + medium_matches + simple_matches
        reason = f"Complex technical signals matched: {', '.join(complex_matches)}."
    elif medium_matches:
        complexity = "medium"
        matched = medium_matches + simple_matches
        reason = f"Moderate reasoning signals matched: {', '.join(medium_matches)}."
    elif simple_matches:
        complexity = "simple"
        matched = simple_matches
        reason = f"Straightforward task signals matched: {', '.join(simple_matches)}."
    elif prompt_length >= config.routing_complex_length:
        complexity = "complex"
        matched = ["long_prompt"]
        reason = "The prompt is long enough to require the advanced profile."
    elif prompt_length >= config.routing_medium_length:
        complexity = "medium"
        matched = ["moderate_prompt_length"]
        reason = "The prompt length suggests a moderate processing workload."
    else:
        complexity = "simple"
        matched = ["safe_default"]
        reason = "No higher-complexity rule matched; using the cost-efficient default."

    model = config.routing_models[complexity]
    return RoutingDecision(
        complexity=complexity,
        model_profile=model["profile"],
        display_name=model["display_name"],
        provider_model_id=model["provider_model_id"],
        reason=reason,
        matched_rules=matched,
        prompt_length=prompt_length,
    )
