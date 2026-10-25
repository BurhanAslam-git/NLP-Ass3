"""
Ollama LLM engine with:
- Substring-based health_check (handles long tag variants like llama3.2:3b-instruct-q4_K_M).
- Single retry on mid-stream failure (G4).
- Shared singleton pattern (duplicate engine instance bug P9 fixed in main.py).
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

from backend.app.config import settings

logger = logging.getLogger(__name__)


class OllamaEngine:
    def __init__(self) -> None:
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming call to Ollama /api/chat with optional tool definitions."""
        url = f"{self._base}/api/chat"
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": 0.2,
                "num_ctx": settings.ollama_num_ctx,
            },
        }
        if tools:
            body["tools"] = tools

        timeout = httpx.Timeout(settings.request_timeout_seconds, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            return data.get("message", {})

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calls_out: list[dict[str, Any]] | None = None,
        attempt: int = 1,
    ) -> AsyncIterator[str]:
        """
        Stream chat tokens from Ollama /api/chat.

        If `tools` is given, the model may answer with tool calls instead of
        text; Ollama returns them in message.tool_calls. We append those to
        `tool_calls_out` (the orchestrator runs them) and keep yielding any
        text tokens as usual.

        On a mid-stream HTTP failure, retries once after 500 ms and yields a
        reconnecting notice token so the user knows what happened.
        """
        url = f"{self._base}/api/chat"
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "keep_alive": settings.ollama_keep_alive,  # keep the model in RAM between turns
            "options": {
                "temperature": 0.3,
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": settings.ollama_num_predict,  # never generate forever
                "repeat_penalty": 1.15,                      # discourage repetition loops
            },
        }
        if tools:
            body["tools"] = tools
        timeout = httpx.Timeout(settings.request_timeout_seconds, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=body) as response:
                    if response.status_code != 200:
                        text = await response.aread()
                        raise RuntimeError(
                            f"Ollama returned {response.status_code}: "
                            f"{text.decode(errors='replace')[:500]}"
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError("Ollama returned malformed stream data.") from exc
                        message = data.get("message") or {}
                        if message.get("tool_calls") and tool_calls_out is not None:
                            tool_calls_out.extend(message["tool_calls"])
                        token = message.get("content")
                        if token:
                            yield token
                        if data.get("done"):
                            # Ollama's own timing: how many prompt tokens it had to read
                            # (vs. reuse from its cache) and how fast it generated.
                            logger.info(
                                "Ollama: prompt %s tok in %.2f s, generated %s tok in %.2f s (tools=%s)",
                                data.get("prompt_eval_count"),
                                (data.get("prompt_eval_duration") or 0) / 1e9,
                                data.get("eval_count"),
                                (data.get("eval_duration") or 0) / 1e9,
                                bool(tools),
                            )
                            break
        except httpx.HTTPError as exc:
            if attempt == 1:
                logger.warning("Ollama stream interrupted — retrying in 500 ms. Error: %s", exc)
                yield " [Reconnecting…] "
                await asyncio.sleep(0.5)
                async for token in self.stream_chat(
                    messages, tools=tools, tool_calls_out=tool_calls_out, attempt=2
                ):
                    yield token
            else:
                raise RuntimeError(
                    "Cannot reach Ollama. Start it with: ollama serve"
                ) from exc

    async def warm_up(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> None:
        """
        Load the model into RAM at startup and let Ollama cache the system
        prompt + tool schemas (the fixed start of every request), so real
        turns only pay for the new tokens.
        """
        body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {"num_predict": 1, "num_ctx": settings.ollama_num_ctx},
        }
        if tools:
            body["tools"] = tools
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(f"{self._base}/api/chat", json=body)
            resp.raise_for_status()
        logger.info("LLM warm-up done (%s).", self._model)

    async def health_check(self) -> bool:
        """
        Check that Ollama is running and the configured model is available.

        Uses substring matching so e.g. 'llama3.2:3b' matches the stored tag
        'llama3.2:3b-instruct-q4_K_M'.
        """
        try:
            timeout = httpx.Timeout(5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{self._base}/api/tags")
                if r.status_code != 200:
                    return False
                payload = r.json()
                models = payload.get("models") or []
                configured = self._model.lower()
                for entry in models:
                    name = str(entry.get("name", "")).lower()
                    # Accept if the configured model name is a substring of the stored tag.
                    if configured in name or name in configured:
                        return True
                logger.warning(
                    "Health check: model '%s' not found in Ollama tags: %s",
                    self._model,
                    [e.get("name") for e in models],
                )
                return False
        except (httpx.HTTPError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Health check failed: %s", exc)
            return False
