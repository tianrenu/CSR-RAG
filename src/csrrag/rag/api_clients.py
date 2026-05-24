"""API clients for the minimal real RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any

import requests


@dataclass
class OpenAICompatibleEmbeddingClient:
    base_url: str
    api_key: str
    model: str
    timeout: int = 60
    max_retries: int = 3
    retry_sleep: float = 2.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = _post_with_retries(
            url=f"{self.base_url.rstrip('/')}/embeddings",
            headers=self._headers(),
            json_body={"model": self.model, "input": texts},
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_sleep=self.retry_sleep,
        )
        payload = response.json()
        data = sorted(payload["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


@dataclass
class OpenAICompatibleChatClient:
    base_url: str
    api_key: str
    model: str
    timeout: int = 120
    max_retries: int = 3
    retry_sleep: float = 2.0
    strict_json_attempts: int = 3
    reasoning_split: bool = True

    def answer(self, question: str, contexts: list[dict[str, Any]], max_tokens: int = 256) -> str:
        return self.answer_with_metadata(question, contexts, max_tokens=max_tokens)["answer"]

    def answer_with_metadata(self, question: str, contexts: list[dict[str, Any]], max_tokens: int = 256) -> dict[str, Any]:
        last_result: dict[str, Any] | None = None
        for attempt in range(1, self.strict_json_attempts + 1):
            result = self._complete(
                self._messages(question, contexts, attempt=attempt),
                max_tokens=max(max_tokens, 1024),
                json_mode=True,
            )
            result["attempt"] = attempt
            result["used_fallback"] = attempt > 1
            result["format_ok"] = _is_format_ok(result)
            last_result = result
            if result["format_ok"]:
                return result

        if last_result is None:
            raise RuntimeError("No completion attempts were made.")
        last_result["answer"] = ""
        return last_result

    def _messages(self, question: str, contexts: list[dict[str, Any]], attempt: int) -> list[dict[str, str]]:
        context_text = "\n\n".join(f"[{idx}] {doc['title']}\n{doc['text']}" for idx, doc in enumerate(contexts, start=1))
        retry_clause = ""
        if attempt > 1:
            retry_clause = (
                "Previous output violated the protocol. Retry with the required JSON object only. "
                "Do not include any hidden or visible reasoning text. "
            )
        system_content = (
            "You are a strict answer extraction function for an academic RAG evaluation. "
            "You must obey this output protocol exactly. "
            f"{retry_clause}"
            "Use only the provided context. "
            "Return exactly one valid JSON object and nothing else. "
            "The JSON object must have exactly one key: \"answer\". "
            "The value of \"answer\" must be a short answer phrase with at most 16 words. "
            "If the answer is not directly supported by the context, use exactly \"I don't know\". "
            "Never output chain-of-thought, analysis, explanations, citations, markdown, code fences, XML, or tags. "
            "Never output <think>, </think>, or any text before or after the JSON object. "
            "Valid examples: {\"answer\":\"Paris\"} and {\"answer\":\"I don't know\"}. "
            "Invalid examples: <think>...</think>{\"answer\":\"Paris\"}; Here is the answer: {\"answer\":\"Paris\"}; ```json."
        )
        return [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Output exactly one JSON object now. No thinking text. No markdown. JSON:"
                ),
            },
        ]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _complete(self, messages: list[dict[str, str]], max_tokens: int, json_mode: bool) -> dict[str, Any]:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            request_body["response_format"] = {"type": "json_object"}
        if self.reasoning_split:
            request_body["reasoning_split"] = True
        response = _post_with_retries(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json_body=request_body,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_sleep=self.retry_sleep,
        )
        payload = response.json()
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        finish_reason = payload["choices"][0].get("finish_reason", "")
        has_reasoning_details = bool(message.get("reasoning_details"))
        cleaned = _strip_thinking(content)
        answer, parse_ok = _extract_answer(cleaned)
        return {
            "answer": answer,
            "json_parse_ok": parse_ok,
            "short_answer_ok": _is_short_answer(answer),
            "json_mode": json_mode,
            "raw_content_length": len(content),
            "finish_reason": finish_reason,
            "had_thinking": content != cleaned,
            "had_reasoning_details": has_reasoning_details,
            "used_fallback": False,
            "format_ok": False,
        }


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "", 1).strip()
    return text


def _extract_answer(text: str) -> tuple[str, bool]:
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict) and "answer" in payload:
            return str(payload["answer"]).strip(), True
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and "answer" in payload:
                return str(payload["answer"]).strip(), True
        except json.JSONDecodeError:
            pass
    match = re.search(r"(?im)^(?:final\s+answer|answer)\s*[:：]\s*(.+)$", cleaned)
    if match:
        return match.group(1).strip().strip("\"'"), False
    return cleaned.strip().strip("\"'"), False


def _is_short_answer(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    reasoning_markers = [
        "let me",
        "analyze",
        "context",
        "question",
        "the user",
        "provided",
        "therefore",
        "because",
        "\n",
    ]
    if any(marker in lowered for marker in reasoning_markers):
        return False
    return len(cleaned.split()) <= 16 and len(cleaned) <= 120


def _is_format_ok(result: dict[str, Any]) -> bool:
    return bool(
        result.get("json_parse_ok")
        and result.get("short_answer_ok")
        and result.get("answer")
        and not result.get("had_thinking")
    )


def _post_with_retries(
    url: str,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(retry_sleep * (attempt + 1))
    if last_error is None:
        raise RuntimeError("Request failed without an exception.")
    raise last_error
