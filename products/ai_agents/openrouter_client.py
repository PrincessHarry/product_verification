"""Thin client for calling free vision-capable LLMs on OpenRouter.

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible
`/chat/completions` endpoint in front of many providers, including a
rotating set of models that are free to use. We don't pin the app to a
single model: free models occasionally get rate-limited or retired, so
we try a short list of known-good free vision models in order and fall
back to the next one if a call fails.

No provider SDK is required, this just uses `requests`.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free, vision-capable models on OpenRouter, tried in order. If one is
# rate-limited, down, or removed, the client automatically tries the
# next one. Update this list from https://openrouter.ai/collections/free-models
# if a model stops working.
DEFAULT_VISION_MODELS: List[str] = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

REQUEST_TIMEOUT_SECONDS = 60


@dataclass
class OpenRouterResult:
    """Result of a chat completion call."""

    ok: bool
    text: str = ""
    model_used: str = ""
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    """Calls OpenRouter's chat completions endpoint with automatic
    fallback across a list of free models."""

    def __init__(self, models: Optional[List[str]] = None):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.models = models or self._models_from_env() or DEFAULT_VISION_MODELS
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME", "Product Verification System")

    @staticmethod
    def _models_from_env() -> Optional[List[str]]:
        raw = os.getenv("OPENROUTER_VISION_MODELS", "").strip()
        if not raw:
            return None
        return [m.strip() for m in raw.split(",") if m.strip()]

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # These two are optional but recommended by OpenRouter - they
            # help free-tier requests get attributed/prioritized properly.
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> OpenRouterResult:
        """Send a chat completion request, trying each configured model in
        turn until one returns a usable response."""
        if not self.is_configured:
            return OpenRouterResult(
                ok=False,
                error=(
                    "OPENROUTER_API_KEY is not set. Add it to your .env file - "
                    "get a free key at https://openrouter.ai/keys"
                ),
            )

        last_error = "No models available"
        for model in self.models:
            try:
                response = requests.post(
                    OPENROUTER_API_URL,
                    headers=self._headers(),
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                last_error = f"{model}: network error ({exc})"
                logger.warning("OpenRouter request failed for %s: %s", model, exc)
                continue

            if response.status_code == 401:
                # Bad API key - no point trying other models.
                return OpenRouterResult(ok=False, error="Invalid OPENROUTER_API_KEY.")

            if response.status_code == 429:
                last_error = f"{model}: rate limited, trying next free model"
                logger.info(last_error)
                continue

            if not response.ok:
                last_error = f"{model}: HTTP {response.status_code} - {response.text[:300]}"
                logger.warning(last_error)
                continue

            try:
                data = response.json()
                text = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError) as exc:
                last_error = f"{model}: unexpected response shape ({exc})"
                logger.warning(last_error)
                continue

            if not text or not text.strip():
                last_error = f"{model}: empty response"
                continue

            return OpenRouterResult(ok=True, text=text, model_used=model, raw=data)

        return OpenRouterResult(ok=False, error=last_error)

    def vision_json(
        self,
        prompt: str,
        image_data_url: str,
        temperature: float = 0.2,
        max_tokens: int = 1400,
    ) -> OpenRouterResult:
        """Ask a vision model a question about an image and parse the
        reply as JSON. Falls back gracefully if the model wraps the JSON
        in prose or a markdown code fence."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if not result.ok:
            return result

        parsed = extract_json(result.text)
        if parsed is None:
            return OpenRouterResult(
                ok=False,
                model_used=result.model_used,
                error="Model response was not valid JSON.",
                raw={"raw_text": result.text},
            )
        result.raw["parsed"] = parsed
        return result


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from an LLM's text reply."""
    text = text.strip()
    # Strip markdown code fences like ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else text

    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        pass

    # Fall back to grabbing the outermost { ... } block.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except (ValueError, TypeError):
            return None
    return None
