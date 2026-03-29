from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass

from openai import OpenAI

from reflectrag.common.json_utils import parse_json_loose
from reflectrag.common.secrets import load_api_keys
from reflectrag.common.telemetry import ApiCallRecord, CostTracker


@dataclass
class ChatResult:
    text: str
    parsed: object


class OpenAICompatRunner:
    def __init__(
        self,
        base_url: str,
        model: str,
        keys_file: str | None = None,
        temperature: float = 0.0,
        timeout: int = 45,
        max_retries: int = 3,
        cost_tracker: CostTracker | None = None,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.keys = load_api_keys(keys_file)
        self.cost_tracker = cost_tracker
        self._key_index = random.randrange(0, len(self.keys)) if self.keys else 0
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {k: 0.0 for k in self.keys}

    def _next_key(self) -> str:
        if not self.keys:
            raise RuntimeError("OPENAI_API_KEYS is empty.")
        with self._lock:
            now = time.time()
            n = len(self.keys)
            for _ in range(n):
                key = self.keys[self._key_index % n]
                self._key_index += 1
                if self._cooldown_until.get(key, 0.0) <= now:
                    return key
            soonest = min(self._cooldown_until.values()) if self._cooldown_until else now
        sleep_for = max(0.05, min(1.0, soonest - now))
        time.sleep(sleep_for)
        with self._lock:
            key = self.keys[self._key_index % len(self.keys)]
            self._key_index += 1
        return key

    def _mark_key_cooldown(self, key: str, error: str) -> None:
        message = (error or "").lower()
        cooldown = 0.2
        if any(token in message for token in ("quota", "pre_consume_token_quota_failed", "insufficient_quota")):
            cooldown = 15.0
        elif any(token in message for token in ("rate limit", "429", "too many requests")):
            cooldown = 5.0
        elif any(token in message for token in ("timed out", "timeout", "connection error", "server disconnected", "remoteprotocolerror")):
            cooldown = 1.5
        cooldown += random.random() * 0.5
        with self._lock:
            self._cooldown_until[key] = max(self._cooldown_until.get(key, 0.0), time.time() + cooldown)

    def complete_json(self, prompt: str, stage: str, system_prompt: str | None = None) -> ChatResult:
        attempts = max(self.max_retries, len(self.keys) * self.max_retries)
        last_error = ""
        for _ in range(attempts):
            api_key = self._next_key()
            t0 = time.perf_counter()
            try:
                client = OpenAI(base_url=self.base_url, api_key=api_key)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    stream=False,
                )
                text = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                if self.cost_tracker:
                    self.cost_tracker.add(
                        ApiCallRecord(
                            stage=stage,
                            latency_s=time.perf_counter() - t0,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            model=self.model,
                            success=True,
                        )
                    )
                return ChatResult(text=text, parsed=parse_json_loose(text))
            except Exception as exc:
                last_error = str(exc)
                self._mark_key_cooldown(api_key, last_error)
                if self.cost_tracker:
                    self.cost_tracker.add(
                        ApiCallRecord(
                            stage=stage,
                            latency_s=time.perf_counter() - t0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            model=self.model,
                            success=False,
                            error=last_error,
                        )
                    )
                time.sleep(0.2 + random.random() * 0.4)
        raise RuntimeError(f"{stage} failed: {last_error}")
