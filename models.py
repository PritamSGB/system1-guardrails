"""Sending one guard check to one model, and reading the answer back.

Every model is asked a single question per request. What differs is how much of the answer is
measured rather than generated:

  decisions     a probability from the Decisions API; nothing is generated
  gpt-4o-mini   a bare yes/no, with p(yes) read from the logprobs of that token
  the rest      a JSON answer plus a confidence the model writes itself
"""
import http.client
import json
import math
import os
import re
import ssl
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

import guards

# API keys come from .env next to this file; anything already exported wins.
load_dotenv(Path(__file__).parent / ".env", override=False)

API_HOST = "api.openai.com"
DECISIONS_PATH = "/v1/decisions"
CHAT_PATH = "/v1/chat/completions"
DECISIONS_MODEL = "gpt-6-luna"  # the model behind the Decisions API endpoint

# OpenAI and Anthropic bill in USD. Rate as of 2026-10-09; override with USD_TO_INR.
# Excludes GST and any card forex markup.
USD_TO_INR = float(os.environ.get("USD_TO_INR", "95.16"))

# python.org builds on macOS ship without a CA bundle; prefer certifi, else the system one.
try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context(cafile="/etc/ssl/cert.pem")


class APIError(Exception):
    pass


class OpenAIClient:
    """Keeps one HTTPS connection open across calls, so TLS setup is paid once.

    Experiment 1 sends a query's three checks at the same time, and each needs its own
    connection for the requests to really overlap.
    """

    def __init__(self, api_key: str):
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                        "Connection": "keep-alive"}
        self.conn = None

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def post(self, path: str, payload: dict) -> tuple[dict, float | None]:
        """POST JSON and return (response, server processing ms from the openai-processing-ms header)."""
        body = json.dumps(payload)
        for attempt in range(2):
            if self.conn is None:
                self.conn = http.client.HTTPSConnection(API_HOST, timeout=60, context=SSL_CONTEXT)
            try:
                self.conn.request("POST", path, body=body, headers=self.headers)
                resp = self.conn.getresponse()
                data = resp.read()
                break
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                self.close()  # server dropped an idle keep-alive connection; reconnect once
                if attempt:
                    raise
        if resp.status != 200:
            raise APIError(f"HTTP {resp.status}: {data.decode()}")
        server_ms = resp.getheader("openai-processing-ms")
        return json.loads(data), float(server_ms) if server_ms else None


def parse_json(text: str) -> dict:
    try:
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except (AttributeError, ValueError):
        return {}


def value_logprob(choice: dict, check: str) -> list[dict] | None:
    """The top_logprobs of the value token that follows `"check":"` in the generated JSON."""
    text = ""
    for t in (choice.get("logprobs") or {}).get("content") or []:
        word = t["token"].strip().strip('"').lower()
        if word in ("yes", "no", "0", "1", "2") and re.search(rf'"{check}"\s*:\s*"?\s*$', text):
            return t["top_logprobs"]
        text += t["token"]
    return None


def normalised(top: list[dict], values: tuple[str, ...]) -> dict | None:
    """Probabilities over the allowed values only, renormalised to sum to 1."""
    probs = {v: 0.0 for v in values}
    for t in top:
        word = t["token"].strip().strip('"').lower()
        for v in values:
            if word == v or (len(v) > 1 and word.startswith(v)):
                probs[v] += math.exp(t["logprob"])
    total = sum(probs.values())
    return {v: p / total for v, p in probs.items()} if total else None


def predicate_verdict(p: float, said_yes: bool, confidence: int | None = None) -> dict:
    return {"p_yes": p, "flag": p > guards.THRESHOLD, "said_yes": said_yes, "confidence": confidence}


def bias_verdict(level: int | None, level_probs: list[float] | None = None, score: float | None = None,
                 confidence: int | None = None) -> dict:
    return {"level": level, "level_probs": level_probs, "score": score, "confidence": confidence,
            "flag": level is not None and level >= 1}


def _from_confidence_json(data: dict, check: str) -> dict:
    """Read {"answer"|"level", "confidence"} written by a model with no logprobs."""
    v = data.get(check) or {}
    conf = v.get("confidence")
    conf = max(0, min(100, int(conf))) if isinstance(conf, (int, float)) else None
    if check == guards.BIAS:
        return bias_verdict(v.get("level") if v.get("level") in (0, 1, 2) else None, confidence=conf)
    said_yes = str(v.get("answer", "")).lower().startswith("yes")
    # The stated confidence stands in for a probability; it is generated text, not a measurement.
    p = (1.0 if said_yes else 0.0) if conf is None else (conf / 100 if said_yes else 1 - conf / 100)
    return predicate_verdict(p, said_yes, conf)


def _cost_inr(model: str, tokens: dict) -> float:
    price = guards.PRICES[model]
    usd = ((tokens["input"] - tokens["cached"]) * price["input"] + tokens["cached"] * price["cached"]
           + tokens["output"] * price["output"]) / 1_000_000
    return usd * USD_TO_INR


def run_check(client: OpenAIClient, model: str, query: str, check: str,
              claude: anthropic.Anthropic | None = None) -> dict:
    """One request asking one guard question. Returns the verdict, timings, tokens and cost."""
    start = time.perf_counter()
    server_ms = None

    if model == guards.DECISIONS:
        result, server_ms = client.post(DECISIONS_PATH, {
            "model": DECISIONS_MODEL,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": query}]}],
            "questions": [guards.DECISION_QUESTION[check]],
        })
        ms = (time.perf_counter() - start) * 1000
        answer = result["answers"][0]
        if check == guards.BIAS:
            probs = [p["probability"] for p in sorted(answer["probabilities"], key=lambda p: p["value"])]
            verdict = bias_verdict(max(range(len(probs)), key=probs.__getitem__), probs, answer["score"])
            verdict["api_confidence"] = answer.get("confidence")
        else:
            verdict = predicate_verdict(answer["probability"], answer["probability"] > 0.5)
        u = result["usage"]
        tokens = {"input": u["input_tokens"], "cached": u.get("input_tokens_details", {}).get("cached_tokens", 0),
                  "output": u.get("output_tokens", 0), "reasoning": 0}
        raw = result["answers"]

    elif model == guards.CLAUDE_MODEL:
        try:
            response = claude.messages.create(
                model=guards.CLAUDE_MODEL, max_tokens=1024, system=guards.chat_prompt(check),
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": guards.chat_schema(check, True)}},
                messages=[{"role": "user", "content": query}],
            )
        except anthropic.APIStatusError as e:
            raise APIError(f"HTTP {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise APIError(f"connection error: {e}") from e
        ms = (time.perf_counter() - start) * 1000
        if response.stop_reason == "refusal":
            raise APIError(f"refusal: {response.stop_details}")
        raw = next((b.text for b in response.content if b.type == "text"), "")
        verdict = _from_confidence_json(parse_json(raw), check)
        u = response.usage
        # Anthropic folds thinking into output tokens, so reasoning can't be separated out.
        tokens = {"input": u.input_tokens + (u.cache_read_input_tokens or 0),
                  "cached": u.cache_read_input_tokens or 0, "output": u.output_tokens, "reasoning": 0}

    else:
        logprobs = model == "gpt-4o-mini"
        payload = {"model": model,
                   "messages": [{"role": "system", "content": guards.chat_prompt(check)},
                                {"role": "user", "content": query}],
                   "response_format": {"type": "json_schema",
                                       "json_schema": {"name": "verdicts", "strict": True,
                                                       "schema": guards.chat_schema(check, not logprobs)}}}
        if logprobs:
            payload.update(temperature=0, max_completion_tokens=100, logprobs=True, top_logprobs=10)
        else:
            payload.update(reasoning_effort=guards.REASONING_EFFORT.get(model, "low"), max_completion_tokens=4000)
        result, server_ms = client.post(CHAT_PATH, payload)
        ms = (time.perf_counter() - start) * 1000
        choice = result["choices"][0]
        raw = choice["message"].get("content") or ""
        data = parse_json(raw)
        if logprobs:
            top = value_logprob(choice, check)
            if check == guards.BIAS:
                stated = str(data.get(check, ""))
                dist = normalised(top, ("0", "1", "2")) if top else None
                probs = [dist[v] for v in ("0", "1", "2")] if dist else None
                verdict = bias_verdict(int(stated) if stated in ("0", "1", "2") else None, probs,
                                       sum(i * p for i, p in enumerate(probs)) if probs else None)
            else:
                said_yes = str(data.get(check, "")).lower() == "yes"
                dist = normalised(top, ("yes", "no")) if top else None
                verdict = predicate_verdict(dist["yes"] if dist else (1.0 if said_yes else 0.0), said_yes)
        else:
            verdict = _from_confidence_json(data, check)
        u = result["usage"]
        tokens = {"input": u["prompt_tokens"],
                  "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                  "output": u["completion_tokens"],
                  "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)}

    return {"check": check, "verdict": verdict, "raw": raw, "client_ms": ms, "server_ms": server_ms,
            "tokens": tokens, "cost_inr": _cost_inr(model, tokens)}


def connect(n: int = 1) -> tuple[list[OpenAIClient], anthropic.Anthropic | None, list[str]]:
    """Open n OpenAI connections and the Anthropic client. Returns the models that are usable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY (see .env.example)")
    claude = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        claude = anthropic.Anthropic(max_retries=0, timeout=60.0)  # retries would hide real latency
    available = [m for m in guards.STABILITY_MODELS if m != guards.CLAUDE_MODEL or claude]
    if claude is None:
        print(f"ANTHROPIC_API_KEY not set: skipping {guards.CLAUDE_MODEL}")
    return [OpenAIClient(api_key) for _ in range(n)], claude, available


def warm_up(clients: list[OpenAIClient], models: list[str], claude: anthropic.Anthropic | None) -> None:
    """One throwaway call per model per connection; the first real call is otherwise slower."""
    for c in clients:
        for m in models:
            run_check(c, m, "warm up", guards.SEX, claude)
        run_check(c, guards.DECISIONS, "warm up", guards.SEX, claude)
