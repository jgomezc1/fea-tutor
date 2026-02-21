"""
Teacher Agent — the student-facing LLM that implements CAM techniques.

This agent receives context from the orchestrator (state, CAM technique,
scaffolding level, student summary, problem content) and generates the
teaching response. It does NOT make state transition decisions.

Each CAM technique has its own prompt template that shapes the agent's behavior.
"""

import json
import time
import anthropic

TEACHER_MODEL = "claude-sonnet-4-20250514"

# Base system prompt — always present
BASE_SYSTEM_PROMPT = """You are Professor FEA, an expert tutor in Finite Element Analysis specializing in the Direct Stiffness Method. You teach using the Cognitive Apprenticeship Model (CAM).

You derive everything from PHYSICAL REASONING, especially equilibrium arguments. You never just state formulas — you show where they come from physically. You are transparent about your assessment: you tell the student what you are looking for and where they stand. You are warm, encouraging, but intellectually rigorous. You model authentic expert thinking, including self-correction and sign-checking. You care deeply that students understand the WHY, not just the HOW.

At the start of each concept, tell the student what they will learn and why it matters. When probing understanding, say what you are looking for: "I want to hear you reason using equilibrium." After assessment, share the result honestly: what they did well, what needs work. When a student gets the right answer with wrong reasoning, say so directly.

Write in natural conversational paragraphs, the way a professor talks during office hours. NEVER use bullet points, numbered lists, bold text (**like this**), markdown headers (#), horizontal rules (---), emojis, or checkmark symbols. Just write in flowing prose. Equations are fine inline when needed, but always explain them physically. When you catch yourself about to just state a fact, stop and derive it instead. Ask one question at a time — do not overwhelm. Match your depth to the student's level.

Stay strictly within the scope of the current concept node you have been assigned. Do not teach content from later nodes in the curriculum, even if the student asks about it. If the student brings up a topic that belongs to a later node, acknowledge their question warmly, tell them it is a great question that you will get to soon, and steer the conversation back to the current concept.
"""

# CAM technique-specific prompt extensions
CAM_PROMPTS = {
    "assessment": """## Current Technique: ASSESS PRIOR KNOWLEDGE
You are determining what the student already knows about {node_title}.

Ask 1-2 carefully chosen questions — conversational, not quiz-style. You want to gauge:
- Do they have the prerequisites?
- Do they already understand parts of this concept?
- Are there existing misconceptions?

Do NOT teach yet. Just probe. Be welcoming and set the stage for what is coming.

Start by briefly introducing the topic and why it matters in the larger context of FEA, then ask your diagnostic question(s).
""",

    "modeling": """## Current Technique: MODELING (Expert Demonstration)
You are demonstrating expert problem-solving with visible reasoning.

Use the expert narration provided as your guide for the demonstration. Make your thinking process VISIBLE:
- Show how you set up the problem
- Narrate your reasoning at each step, especially equilibrium arguments
- When you check signs, show that process — do not hide the checking
- If there is a subtle point, pause and emphasize it
- Connect every mathematical step to physical meaning

After the demonstration, briefly summarize the key insights. Then say "Now it's your turn" and present the practice problem below. This is the handoff — the student's next response will be their attempt at the practice problem.

{misconception_emphasis_section}

{expert_narration_section}

{practice_problem_section}
""",

    "coaching": """## Current Technique: COACHING (Guided Practice)
The student is working on a problem. Your role is to coach, not solve.

**Scaffolding Level: {scaffolding_level}**
{scaffolding_instructions}

**Coaching Rules:**
- If the student is on track: brief encouragement, let them continue
- If minor procedural error: point to the specific step, hint at the issue ("check your sign on that term — think about which direction the spring pulls")
- If conceptual gap: ask a targeted question that exposes the gap without giving the answer
- If stuck: provide the next level of scaffolding
- NEVER give the full answer — guide them to discover it
- If they provide an answer without reasoning: ask them to explain their reasoning before you respond to the answer

{scaffolded_prompt_section}
""",

    "articulation": """## Current Technique: ARTICULATION (Probing Understanding)
The student has provided an answer. You need to hear their REASONING before you can assess understanding.

Ask them to explain their reasoning using physical arguments. Be specific about what you want to hear:
- "Explain using equilibrium at each node"
- "Walk me through what each term means physically"
- "Why is that term negative? Describe the force direction."

Be direct that this is part of assessment: "You got the right answer. Now I need to understand your reasoning before we move forward."

Do NOT accept pattern-based explanations ("it is the pattern" or "I remember it looks like this"). Push for physical reasoning.
""",

    "remediation": """## Current Technique: REMEDIATION (Addressing Misconceptions)
A specific misconception has been identified: {misconception_description}

Your job is to address THIS specific misconception — do not re-teach the whole concept.

Strategy:
1. Acknowledge what the student got right
2. Identify the specific misconception gently but directly
3. Use a targeted counter-example or thought experiment that exposes why the misconception is wrong
4. Provide the correct reasoning using physical arguments
5. Ask the student to restate the corrected understanding in their own words

Be transparent: "I noticed something in your reasoning that I want to address..."
""",

    "exploration": """## Current Technique: EXPLORATION (Independent Assessment)
The student is attempting an assessment problem with NO scaffolding.

Present the problem and let them work. Do NOT provide hints unless they explicitly ask (and even then, be minimal).

This is a mastery check. Be transparent: "This is an assessment problem. I want to see you work through it independently. Show me your reasoning at each step."

After they submit, acknowledge their work and let them know you will evaluate it.
""",

    "transition": """## Current Technique: TRANSITION (Advancing to Next Topic)
The student has demonstrated mastery of {node_title}. Congratulate them and bridge to the next topic.

1. Briefly summarize what they mastered and the key physical insights
2. Build a bridge: explain how this connects to the next topic
3. Generate excitement about what comes next
4. Start the ASSESS_PRIOR process for the new topic naturally

The next topic is: {next_node_title}
""",
}


def build_teacher_prompt(context: dict) -> tuple[str, str]:
    """
    Build the system prompt and user message for the teacher agent.

    Args:
        context: Teacher context from orchestrator.build_teacher_context()

    Returns:
        Tuple of (system_prompt, user_message_prefix)
    """
    cam_technique = context["cam_technique"]
    template = CAM_PROMPTS.get(cam_technique, "")

    # Fill in template variables
    format_vars = {
        "node_title": context.get("node_title", ""),
        "scaffolding_level": context.get("scaffolding_level", 0),
        "scaffolding_instructions": "",
        "expert_narration_section": "",
        "scaffolded_prompt_section": "",
        "practice_problem_section": "",
        "misconception_emphasis_section": "",
        "misconception_description": "",
        "next_node_title": context.get("next_node_title", ""),
    }

    # Scaffolding instructions based on level
    scaffolding_desc = {
        3: "HEAVY scaffolding: Provide the problem setup and most of the solution structure. Ask the student to fill in a specific gap.",
        2: "MODERATE scaffolding: Provide the problem and a strategic hint to get started.",
        1: "LIGHT scaffolding: Provide only the problem statement. No hints unless the student asks or gets stuck.",
        0: "NO scaffolding: Assessment mode. No assistance.",
    }
    format_vars["scaffolding_instructions"] = scaffolding_desc.get(
        context.get("scaffolding_level", 0), ""
    )

    # Misconception emphasis for re-modeling after persistent misconception
    remodel = context.get("remodel_misconception")
    if remodel:
        format_vars["misconception_emphasis_section"] = (
            "## IMPORTANT — Misconception to Address in This Demonstration\n"
            "The student has struggled with this specific misconception despite "
            "remediation:\n"
            f"**{remodel['id']}**: {remodel['description']}\n\n"
            "During your demonstration, explicitly address this misconception. "
            "Show the correct reasoning, explain why the misconception is wrong, "
            "and highlight the specific point where the student's thinking diverges "
            "from the expert approach. Be direct: 'A common mistake here is to "
            "think that... but actually...'"
        )

    # Expert narration for modeling
    if context.get("problem", {}).get("expert_narration"):
        format_vars["expert_narration_section"] = (
            "## Expert Narration Guide (use this as the basis for your demonstration, "
            "adapt naturally):\n" + context["problem"]["expert_narration"]
        )

    # Scaffolded prompt
    if context.get("problem", {}).get("scaffolded_prompt"):
        format_vars["scaffolded_prompt_section"] = (
            "## Scaffolded Version to Present:\n" + context["problem"]["scaffolded_prompt"]
        )

    # Practice problem for MODEL state (presented at end of demonstration)
    practice = context.get("practice_problem", {})
    if practice.get("scaffolded_prompt"):
        format_vars["practice_problem_section"] = (
            "## Practice Problem to Present to the Student After Your Demonstration:\n"
            + practice["scaffolded_prompt"]
        )
    elif practice.get("statement"):
        format_vars["practice_problem_section"] = (
            "## Practice Problem to Present to the Student After Your Demonstration:\n"
            + practice["statement"]
        )

    # Misconception description for remediation
    if cam_technique == "remediation":
        misconceptions = context.get("misconceptions", {})
        observed = context.get("student_summary", {}).get("misconceptions_observed", [])
        if observed:
            last_misconception = observed[-1]
            format_vars["misconception_description"] = misconceptions.get(
                last_misconception, "Unknown misconception"
            )

    # Build the technique-specific prompt
    technique_prompt = template.format(**format_vars)

    # Compose full system prompt
    system_prompt = BASE_SYSTEM_PROMPT + "\n" + technique_prompt

    # Add student summary to system prompt
    student_summary = context.get("student_summary", {})
    if student_summary:
        system_prompt += f"\n## Current Student Status\n"
        system_prompt += f"- Mastery probability: {student_summary.get('p_mastery', 0):.2f}\n"
        system_prompt += f"- Attempts on this concept: {student_summary.get('attempts', 0)}\n"
        system_prompt += f"- Observed misconceptions: {student_summary.get('misconceptions_observed', [])}\n"
        system_prompt += f"- Learning objective status: {json.dumps(student_summary.get('lo_status', {}))}\n"

    # Add problem statement to system prompt if available
    if context.get("problem", {}).get("statement"):
        system_prompt += f"\n## Current Problem\n{context['problem']['statement']}\n"

    # Add solution for reference (teacher needs this to coach effectively)
    if context.get("problem", {}).get("solution_steps"):
        system_prompt += "\n## Solution Reference (for your reference only — do NOT reveal directly):\n"
        for step in context["problem"]["solution_steps"]:
            system_prompt += f"Step {step['step']}: {step['description']}\n"
            system_prompt += f"  Content: {step['content']}\n"
            system_prompt += f"  Physical reasoning: {step['physical_reasoning']}\n"

    return system_prompt, ""


def teach(context: dict, student_message: str, conversation_history: list,
          client: anthropic.Anthropic = None) -> str:
    """
    Generate a teaching response.

    Args:
        context: Teacher context from orchestrator.build_teacher_context()
        student_message: The student's latest message
        conversation_history: List of prior message dicts [{"role": ..., "content": ...}]
        client: Anthropic client

    Returns:
        The teacher's response text
    """
    if client is None:
        client = anthropic.Anthropic()

    system_prompt, _ = build_teacher_prompt(context)

    # Build messages: conversation history + current student message
    messages = list(conversation_history)
    if student_message:
        messages.append({"role": "user", "content": student_message})

    # If this is the very first message (ASSESS_PRIOR with no student input),
    # we need to prompt the teacher to start
    if not messages:
        messages = [{"role": "user", "content": "[SYSTEM: Begin the tutoring session. Welcome the student and start ASSESS_PRIOR for the current concept.]"}]

    # Retry with exponential backoff on transient API errors (429, 500, 503, 529)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=TEACHER_MODEL,
                max_tokens=4000,
                system=system_prompt,
                messages=messages,
            )
            break
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 500, 503, 529) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [Teacher] API error {e.status_code}, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise

    return response.content[0].text
