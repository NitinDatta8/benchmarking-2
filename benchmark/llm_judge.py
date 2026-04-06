"""
LLM-as-Judge quality evaluation using OpenAI.

Sends the original prompt + model output to an OpenAI model, which scores
the response on a 1-5 scale across correctness, completeness, and relevance.
Returns a composite score normalized to 0-1.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_JUDGE_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = """\
You are an expert evaluator judging AI assistant responses.
Score the response on three dimensions (1-5 each):

1. **Correctness**: Are the facts, logic, and any structured data accurate?
2. **Completeness**: Does the response fully address all parts of the user's request?
3. **Relevance**: Is the response focused and free of irrelevant filler?

Return ONLY valid JSON (no markdown fencing):
{"correctness": <int>, "completeness": <int>, "relevance": <int>}
"""


def _build_judge_prompt(system_msg, user_msg, model_output):
    return (
        f"## Original Prompt\n"
        f"**System:** {system_msg or '(none)'}\n\n"
        f"**User:** {user_msg}\n\n"
        f"## Model Response\n"
        f"{model_output}\n\n"
        f"Score the response above."
    )


def _parse_scores(text):
    """Parse judge JSON response, return (score_0_to_1, raw_scores_dict)."""
    clean = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    scores = json.loads(clean)
    total = scores["correctness"] + scores["completeness"] + scores["relevance"]
    # Normalize: max possible = 15, min = 3; clamp to [0, 1]
    normalized = max(0.0, min(1.0, (total - 3) / 12))
    return normalized, scores


class JudgeResult:
    __slots__ = ("score", "passed", "raw_scores")

    def __init__(self, score, raw_scores):
        self.score = score          # 0.0 - 1.0
        self.passed = score >= 0.6  # maps to ~3.4/5 average
        self.raw_scores = raw_scores


_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        _client = OpenAI(api_key=api_key)
    return _client


def judge(prompt, model_output):
    """Score a single model output using the LLM judge.

    Args:
        prompt: The prompt dict from benchmark_prompts.json.
        model_output: The model's text output.

    Returns:
        JudgeResult with .score (0-1), .passed (bool), .raw_scores (dict).
    """
    if not model_output or not model_output.strip():
        return JudgeResult(0.0, {"correctness": 1, "completeness": 1, "relevance": 1})

    system_msg = prompt.get("system", "")
    if prompt.get("type") == "multi_turn":
        user_msg = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in prompt["messages"]
        )
    else:
        user_msg = prompt["user"]

    judge_input = _build_judge_prompt(system_msg, user_msg, model_output)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": judge_input},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        text = response.choices[0].message.content
        score, raw = _parse_scores(text)
        return JudgeResult(score, raw)
    except Exception as e:
        print(f"  [judge] error: {e}")
        return JudgeResult(0.0, {"correctness": 0, "completeness": 0, "relevance": 0})


def judge_batch(prompts, outputs):
    """Score a list of (prompt, output) pairs. Returns list of JudgeResult."""
    return [judge(p, o) for p, o in zip(prompts, outputs)]
