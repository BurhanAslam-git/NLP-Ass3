"""
Tool orchestrator: LLM decides -> validate -> execute (async, timeout) -> LLM answers.

How the LLM picks a tool (Ollama native tool calling):
  1. We send /api/chat with the conversation AND a "tools" list (name,
     description, JSON schema for every tool — see domain/tools.py).
  2. The model's chat template shows it those definitions. The model then
     either writes a normal answer (we stream the tokens straight to the user)
     or emits a structured tool call, which Ollama parses and returns as
     message.tool_calls = [{"function": {"name": ..., "arguments": {...}}}].
  3. For each call we validate the arguments against the schema and run the
     handler with asyncio.wait_for(timeout=TOOL_TIMEOUT_SECONDS).
  4. The result (or a structured error) is appended as a {"role": "tool"}
     message and we call the LLM again, so it can explain the result in words.
  5. At most TOOL_MAX_ROUNDS tool rounds per turn; the final round is sent
     WITHOUT tools, which forces a plain-text answer (no endless loops).

Failures (unknown tool, bad arguments, timeout, exception) never crash the
turn: they become {"success": false, "error": "..."} and the LLM tells the
user something sensible, in text and (for voice turns) in speech.
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from backend.app.config import settings
from backend.app.domain.tools import TOOLS, tool_specs, validate_args
from backend.app.models.schemas import ToolCallFrame, ToolResultFrame

logger = logging.getLogger(__name__)


async def run_tool(name: str, raw_args: Any, defaults: dict[str, str] | None = None) -> tuple[dict, bool, float]:
    """Validate and run one tool call. Returns (result, ok, latency_ms)."""
    t0 = time.perf_counter()
    try:
        if isinstance(raw_args, str):           # some models send arguments as a JSON string
            raw_args = json.loads(raw_args or "{}")
        args = dict(raw_args or {})
        # Fill the patron's card from the conversation if the model left it out.
        for key, value in (defaults or {}).items():
            if value and key in TOOLS.get(name, {}).get("parameters", {}).get("properties", {}):
                args.setdefault(key, value)
        clean = validate_args(name, args)
        result = await asyncio.wait_for(TOOLS[name]["handler"](**clean), timeout=settings.tool_timeout_seconds)
    except KeyError as exc:
        result = {"success": False, "error": str(exc).strip("'\"")}
    except (ValueError, json.JSONDecodeError) as exc:
        result = {"success": False, "error": f"Invalid arguments for {name}: {exc}"}
    except asyncio.TimeoutError:
        result = {"success": False, "error": f"The {name} service took longer than "
                                              f"{settings.tool_timeout_seconds:g} s and was stopped. Please try again."}
    except Exception as exc:  # a bug or I/O error inside the tool
        logger.exception("Tool %s failed", name)
        result = {"success": False, "error": f"The {name} service failed: {exc}"}
    latency_ms = (time.perf_counter() - t0) * 1000
    ok = bool(result.get("success", True)) and "error" not in result
    logger.info("Tool %s ok=%s %.0f ms", name, ok, latency_ms)
    return result, ok, latency_ms


async def chat_with_tools(llm, payload: list[dict], session_id: str,
                          defaults: dict[str, str] | None = None) -> AsyncIterator[tuple[str, Any]]:
    """
    Run the LLM with tools and yield events for the WebSocket layer:
      ("token", {"content": str}), ("tool_call", ToolCallFrame),
      ("tool_result", ToolResultFrame), ("timing", {...}).
    `payload` is extended in place with the tool messages of this turn only
    (the stored session history keeps just user/assistant text).
    """
    t_start = time.perf_counter()
    tool_ms = 0.0
    first_token = True
    for round_no in range(settings.tool_max_rounds + 1):
        offer_tools = round_no < settings.tool_max_rounds   # last round: answer in text
        tool_calls: list[dict] = []
        async for token in llm.stream_chat(payload, tools=tool_specs() if offer_tools else None,
                                           tool_calls_out=tool_calls):
            if first_token:
                first_token = False
                # Time spent in the LLM before the first word (tool execution excluded).
                llm_ms = (time.perf_counter() - t_start) * 1000 - tool_ms
                yield ("timing", {"llm_first_token_ms": round(llm_ms, 1)})
            yield ("token", {"content": token})
        if not tool_calls:
            break

        payload.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for call in tool_calls:
            fn = call.get("function", {})
            name, args = fn.get("name", ""), fn.get("arguments") or {}
            yield ("tool_call", ToolCallFrame(session_id=session_id, name=name,
                                              arguments=args if isinstance(args, dict) else {"raw": str(args)}))
            result, ok, latency_ms = await run_tool(name, args, defaults)
            tool_ms += latency_ms
            yield ("tool_result", ToolResultFrame(session_id=session_id, name=name, ok=ok,
                                                  latency_ms=round(latency_ms, 1)))
            payload.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})

    yield ("timing", {"tool_ms": round(tool_ms, 1) if tool_ms else None})
