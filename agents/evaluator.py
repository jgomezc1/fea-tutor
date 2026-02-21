"""
Evaluator Agent — a focused LLM call that classifies student responses.

This agent NEVER talks to the student. It receives the student's response,
the correct solution, learning objectives, and misconception inventory,
and outputs a structured classification.

The evaluator is separate from the teacher to ensure evaluation is rigorous
and not influenced by pedagogical concerns (friendliness, encouragement, etc.).
"""

import json
import time
import anthropic

# The model to use for evaluation
EVALUATOR_MODEL = "claude-sonnet-4-20250514"

EVALUATOR_SYSTEM_PROMPT = """You are an expert evaluator for a Finite Element Analysis (FEA) tutoring system. Your ONLY job is to classify student responses along two dimensions: procedural correctness and conceptual understanding.

You are rigorous, analytical, and conservative. You do NOT teach, encourage, or provide feedback. You ONLY classify.

## Classification Dimensions

### Procedural Correctness
- **correct**: Right answer through valid intermediate steps
- **minor_error**: Right approach, arithmetic or notational mistake
- **major_error**: Wrong approach or fundamentally flawed steps
- **incomplete**: Started correctly but did not finish or got stuck

### Conceptual Understanding
- **deep**: Student explains WHY using physical reasoning (equilibrium arguments, force directions, physical interpretation). They construct reasoning, not recite formulas.
- **mechanical**: Student follows the procedure correctly but cannot explain the reasoning. They know THAT something is true but not WHY. Pattern matching, formula recitation, or "I remember it's..." are indicators.
- **misconceived**: Student's explanation reveals a specific known misconception from the inventory.
- **absent**: Student provides no reasoning, just numbers or statements without justification.

## Critical Evaluation Principle
The key marker of DEEP understanding in this FEA course is PHYSICAL REASONING — specifically, equilibrium arguments. A student who says "the off-diagonal is -k because the pattern is positive on diagonal, negative off-diagonal" is MECHANICAL. A student who says "the off-diagonal is -k because displacing node 1 causes the spring to pull node 2, and that force opposes the positive direction" is DEEP.

Stumbling or self-correcting while constructing physical reasoning is a POSITIVE signal — it means the student is thinking, not reciting.

## Output Format
You MUST respond with ONLY a valid JSON object (no markdown, no explanation):

{
    "procedural": "correct|minor_error|major_error|incomplete",
    "conceptual": "deep|mechanical|misconceived|absent",
    "misconception_id": null or "M1" or "M2" etc.,
    "physical_reasoning_detected": true or false,
    "evidence": "the specific student statement(s) that led to this classification",
    "reasoning_markers_present": ["list of positive markers detected"],
    "reasoning_markers_absent": ["list of expected positive markers not found"],
    "confidence": "high|medium|low",
    "suggested_action": "advance|practice|articulation_probe|remediate|re_model"
}
"""


def build_evaluator_prompt(context: dict) -> str:
    """Build the user message for the evaluator given the orchestrator context."""

    parts = []

    parts.append("## Student Response")
    parts.append(context["student_response"])

    if "problem" in context:
        problem = context["problem"]
        parts.append("\n## Problem Statement")
        parts.append(json.dumps(problem.get("solution_steps", []), indent=2))

        parts.append("\n## Expected Final Answer")
        parts.append(str(problem.get("final_answer", "N/A")))

        parts.append("\n## Target Learning Objectives")
        for lo_id in problem.get("targets_los", []):
            lo = context["learning_objectives"].get(lo_id, {})
            parts.append(f"- {lo_id}: {lo.get('description', 'N/A')}")

    parts.append("\n## Known Misconceptions for This Concept")
    for m_id, m_desc in context.get("misconceptions", {}).items():
        parts.append(f"- {m_id}: {m_desc}")

    parts.append("\n## Physical Reasoning Markers")
    markers = context.get("physical_reasoning_markers", {})
    if markers.get("positive"):
        parts.append("Positive (indicates deep understanding):")
        for m in markers["positive"]:
            parts.append(f"  + {m}")
    if markers.get("negative"):
        parts.append("Negative (indicates mechanical understanding only):")
        for m in markers["negative"]:
            parts.append(f"  - {m}")

    if "problem" in context and "evaluation_markers" in context["problem"]:
        em = context["problem"]["evaluation_markers"]
        parts.append("\n## Problem-Specific Evaluation Markers")
        if em.get("positive"):
            parts.append("Positive:")
            for m in em["positive"]:
                parts.append(f"  + {m}")
        if em.get("negative"):
            parts.append("Negative:")
            for m in em["negative"]:
                parts.append(f"  - {m}")

    parts.append("\n## Instructions")
    parts.append("Classify this student response. Output ONLY valid JSON.")

    return "\n".join(parts)


def evaluate(context: dict, client: anthropic.Anthropic = None) -> dict:
    """
    Call the evaluator LLM and return structured classification.

    Args:
        context: Evaluator context from orchestrator.build_evaluator_context()
        client: Anthropic client (creates one if not provided)

    Returns:
        Classification dict with keys: procedural, conceptual, misconception_id,
        physical_reasoning_detected, evidence, confidence, suggested_action
    """
    if client is None:
        client = anthropic.Anthropic()

    user_message = build_evaluator_prompt(context)

    # Retry with exponential backoff on transient API errors (429, 500, 503, 529)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=EVALUATOR_MODEL,
                max_tokens=1000,
                system=EVALUATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            break
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 500, 503, 529) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [Evaluator] API error {e.status_code}, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise

    # Parse the JSON response
    response_text = response.content[0].text.strip()

    # Handle potential markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        evaluation = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: conservative default if parsing fails
        evaluation = {
            "procedural": "incomplete",
            "conceptual": "absent",
            "misconception_id": None,
            "physical_reasoning_detected": False,
            "evidence": "Failed to parse evaluator response",
            "reasoning_markers_present": [],
            "reasoning_markers_absent": [],
            "confidence": "low",
            "suggested_action": "articulation_probe",
        }

    return evaluation
