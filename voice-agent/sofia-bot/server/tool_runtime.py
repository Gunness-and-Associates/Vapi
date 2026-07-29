"""Shared HTTP tool execution used by the dashboard tester and live voice agent."""

import asyncio
import json
from typing import Any

import aiohttp


ALLOWED_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def normalize_http_method(method: str | None) -> str:
    normalized = (method or "POST").strip().upper()
    if normalized not in ALLOWED_HTTP_METHODS:
        raise ValueError(f"Unsupported HTTP method '{normalized}'.")
    return normalized


async def execute_http_tool(
    *,
    url: str,
    arguments: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    method: str = "POST",
    timeout_secs: int = 15,
    retries: int = 0,
) -> tuple[int, str]:
    """Execute a configured HTTP tool and return its final status and response text."""
    http_method = normalize_http_method(method)
    payload = arguments or {}
    attempts = max(1, min(int(retries) + 1, 3))
    timeout = aiohttp.ClientTimeout(total=max(1, min(int(timeout_secs), 60)))
    last_error: Exception | None = None

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(attempts):
            try:
                request_data: dict[str, Any] = {
                    "method": http_method,
                    "url": url,
                    "headers": headers or {},
                }
                if http_method == "GET":
                    request_data["params"] = payload
                else:
                    request_data["json"] = payload
                async with session.request(**request_data) as response:
                    text = await response.text()
                    if 200 <= response.status < 300 or attempt + 1 >= attempts:
                        return response.status, text
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    raise
            await asyncio.sleep(0.25 * (attempt + 1))

    if last_error:
        raise last_error
    raise RuntimeError("Tool request did not complete.")


def extract_tool_result(response_text: str, response_path: str = "") -> str:
    """Extract a useful model-facing value from common API response shapes."""
    try:
        data = json.loads(response_text)
    except (TypeError, json.JSONDecodeError):
        return response_text

    value: Any = data
    path = (response_path or "").strip()
    if path:
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                index = int(part)
                value = value[index] if index < len(value) else None
            else:
                value = None
            if value is None:
                break
        if value is not None:
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            value = results[0].get("result")
            if value is not None:
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        for key in ("result", "message", "data"):
            if data.get(key) is not None:
                value = data[key]
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
