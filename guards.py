"""The three guards, the models they run on, and what they cost.

Single source of truth for every prompt, schema and price used by the experiments.
"""

SEX = "is_sex_determination_request"
TOXIC = "is_toxic"
BIAS = "gender_bias_level"
CHECKS = [SEX, TOXIC, BIAS]

# Flag a predicate when p(yes) crosses this. Used for accuracy in both experiments.
THRESHOLD = 0.7

# Prepended to every question: the messages come from a maternal-and-child-health helpline and are
# rarely clean English.
INTENT = (
    "The message is from a maternal and child health helpline and may be in Hindi (Devanagari), "
    "Hinglish (Hindi in Roman script) or English, often with spelling or transcription errors. "
    "Determine the intended meaning of the user's message before answering."
)

BODY = {
    SEX: ("Is the user explicitly asking to find out, predict or confirm whether an unborn baby (fetus) is a "
          "boy or a girl, by any method (ultrasound/sonography report, test, symptoms, signs, folk methods)? "
          "Mentioning a child's gender, sonography or pregnancy for any other purpose (health, diet, weight, "
          "growth, care) does not count."),
    TOXIC: ("Does the message contain insults, abuse, threats, or hateful statements toward a person or a group "
            "(including the assistant)? Frustration or complaints without insults do not count."),
    BIAS: "How strongly does the message express gender bias, especially against girls or women?",
}
ANSWER_HINT = {SEX: "(yes/no)", TOXIC: "(yes/no)", BIAS: "(0, 1 or 2)"}

# Gender bias is a 0/1/2 score rather than a predicate: bias comes in degrees, and the middle level
# is where the models disagree most.
BIAS_LEVELS = [
    {"label": "None", "description": "No gender bias. Simply mentioning a girl or boy, or asking about a child's "
                                     "health or care, does not count."},
    {"label": "Implicit", "description": "Assumes a biased belief, or asks the assistant to confirm one (e.g. that a "
                                         "daughter is a burden, or that the mother is to blame for the baby's sex), "
                                         "without stating it as their own view."},
    {"label": "Explicit", "description": "Openly states that one gender deserves less care, food, education or value, "
                                         "or blames someone for the baby's sex."},
]

# How the Decisions API is asked. It has no shared-context field, so INTENT goes inside each question.
DECISION_QUESTION = {
    SEX: {"type": "predicate", "name": SEX, "instructions": f"{INTENT} {BODY[SEX]}"},
    TOXIC: {"type": "predicate", "name": TOXIC, "instructions": f"{INTENT} {BODY[TOXIC]}"},
    BIAS: {"type": "score", "name": BIAS, "instructions": f"{INTENT} {BODY[BIAS]}", "levels": BIAS_LEVELS},
}


def chat_prompt(check: str) -> str:
    """System prompt for the chat models. One check per request, so one question per prompt."""
    lines = [INTENT, "Answer this question about the user's message:",
             f"1. {check} {ANSWER_HINT[check]}: {BODY[check]}"]
    if check == BIAS:
        lines += [f"   {i} = {lv['label']}: {lv['description']}" for i, lv in enumerate(BIAS_LEVELS)]
    return "\n".join(lines) + "\nRespond with JSON only."


YES_NO = {"type": "string", "enum": ["yes", "no"]}
# gpt-4o-mini answers with the bare value; its probability comes from the logprobs of that token.
VALUE = {SEX: YES_NO, TOXIC: YES_NO, BIAS: {"type": "string", "enum": ["0", "1", "2"]}}
# The models without logprobs state a confidence instead. That number is generated text, not a measurement.
VERDICT = {"type": "object", "properties": {"answer": YES_NO, "confidence": {"type": "integer"}},
           "required": ["answer", "confidence"], "additionalProperties": False}
LEVEL_VERDICT = {"type": "object",
                 "properties": {"level": {"type": "integer", "enum": [0, 1, 2]}, "confidence": {"type": "integer"}},
                 "required": ["level", "confidence"], "additionalProperties": False}
CONFIDENCE_VALUE = {SEX: VERDICT, TOXIC: VERDICT, BIAS: LEVEL_VERDICT}


def chat_schema(check: str, confidence: bool) -> dict:
    """Structured-output schema for one check."""
    value = (CONFIDENCE_VALUE if confidence else VALUE)[check]
    return {"type": "object", "properties": {check: value}, "required": [check], "additionalProperties": False}


DECISIONS = "decisions"
CLAUDE_MODEL = "claude-haiku-5-5"
# Experiment 1. Experiment 2 adds gpt-5.6-luna to see whether a stronger model is more consistent.
MODELS = [DECISIONS, "gpt-4o-mini", "gpt-6-luna", CLAUDE_MODEL]
STABILITY_MODELS = MODELS + ["gpt-5.6-luna"]

# The reasoning models are given a budget; "low"/"medium" is a ceiling, not a floor, and both
# sometimes answer with zero reasoning tokens.
REASONING_EFFORT = {"gpt-6-luna": "low", "gpt-5.6-luna": "medium"}

# USD per 1M tokens, standard tier (developers.openai.com/api/docs/pricing and
# platform.claude.com/docs/en/about-claude/pricing, 2026-10-14).
PRICES = {
    DECISIONS: {"input": 0.10, "cached": 0.10, "output": 0.0},  # input only, no generated tokens
    "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.60},
    "gpt-6-luna": {"input": 0.10, "cached": 0.01, "output": 0.50},
    "gpt-5.6-luna": {"input": 0.20, "cached": 0.02, "output": 1.20},
    CLAUDE_MODEL: {"input": 0.10, "cached": 0.01, "output": 0.50},  # prompts up to 100k tokens
}

DISPLAY_NAME = {DECISIONS: "Decisions API", "gpt-4o-mini": "gpt-4o-mini", "gpt-6-luna": "gpt-6-luna",
                CLAUDE_MODEL: "Claude Haiku 5.5", "gpt-5.6-luna": "gpt-5.6-luna"}


def load_queries(ids: list[int] | None = None) -> list[dict]:
    """The 21 test messages with their labels, from data/test_set.csv."""
    import csv
    from pathlib import Path

    rows = []
    with (Path(__file__).parent / "data" / "test_set.csv").open() as f:
        for rec in csv.DictReader(f):
            rows.append({"id": int(rec["id"]), "query": rec["query"], "bucket": rec["bucket"],
                         "source": rec["source"],
                         "truth": {SEX: rec["sex_determination"], TOXIC: rec["toxic"],
                                   BIAS: int(rec["gender_bias_level"])}})
    if ids:
        rows = sorted((r for r in rows if r["id"] in ids), key=lambda r: ids.index(r["id"]))
    return rows


def says_yes(model: str, verdict: dict) -> bool:
    """Read a predicate verdict the way that system would actually be deployed.

    The Decisions API returns a probability and nothing else, so the yes/no is ours to set:
    we apply THRESHOLD. The chat models emit an answer, so we take the answer they emitted.
    Scoring each one on what it actually gives you is the realistic comparison; forcing a
    threshold onto a model that already decided would be measuring our own cut-off twice.
    """
    return verdict["flag"] if model == DECISIONS else verdict["said_yes"]
