"""A small OpenAI-compatible client with retries and prompt-keyed caching."""
import hashlib
import json
import os
import time
from pathlib import Path
import httpx


class IncompleteResponse(Exception):
    """The completion hit max_tokens before it produced a complete JSON object."""


class ChatClient:
    """Call a locally hosted or remote Chat Completions endpoint."""

    def __init__(self, base_url, model, cache, context_budget=65000, max_output=4096,
                 json_mode=True, timeout=180, strict_length=False):
        """Store endpoint settings without writing API credentials to disk."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.context_budget = context_budget
        self.max_output = max_output
        self.json_mode = json_mode
        self.timeout = timeout
        self.strict_length = strict_length

    def complete(self, messages):
        """Return a parsed JSON response and usage metadata, or fail explicitly."""
        # UTF-8 bytes are a conservative proxy for common byte-based tokenizers.
        # This is not an exact token counter for arbitrary hosted models.
        estimated = sum(len(m["content"].encode("utf-8")) + 64 for m in messages)
        if estimated + self.max_output > self.context_budget:
            raise ValueError(
                "Context guard exceeded: raise budget/model context or shorten chunks/state "
                f"(input_byte_estimate={estimated}, max_output={self.max_output}, "
                f"context_budget={self.context_budget}, message_count={len(messages)})"
            )
        request = {"model": self.model, "messages": messages, "temperature": 0,
                   "max_tokens": self.max_output, "stream": False,
                   "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0, "stop": None  }
                   # Apply the thinking switch only to explicitly configured models.
        thinking_control_models = {
                 "qwen3.6-35b-a3b",
                "qwen3.8-27b",
        }
        if self.model in thinking_control_models:
            request["chat_template_kwargs"] = {
                "enable_thinking": False
            }

        if self.json_mode:
            request["response_format"] = {"type": "json_object"}
        cache_identity = [self.base_url, request] + (["strict-length-v1"] if self.strict_length else [])
        key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True,
                                       ensure_ascii=False).encode()).hexdigest()
        target = self.cache / (key + ".json")
        if target.exists():
            saved = json.loads(target.read_text())
            return saved["result"], {**saved["metadata"], "cache_hit": True}
        headers = {}
        if os.environ.get("LLM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["LLM_API_KEY"]
        started = time.monotonic()
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.base_url + "/chat/completions",
                                           json=request, headers=headers)
                    response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                reason = choice.get("finish_reason")
                content = (choice.get("message") or {}).get("content")
                if self.strict_length and reason == "length":
                    raise IncompleteResponse("Completion length cutoff (strict grouped mode)")
                try:
                    result = json.loads(content)
                except (TypeError, json.JSONDecodeError) as exc:
                    # Same prompt and temperature 0 will hit the same cap again.
                    if reason == "length":
                        raise IncompleteResponse(
                            f"Incomplete response: length (max_tokens={self.max_output})"
                        ) from exc
                    raise ValueError(f"Incomplete response: {reason}") from exc
                if reason not in ("stop", None, "length"):
                    raise ValueError(f"Incomplete response: {reason}")
                if (not isinstance(result, dict) or not {"add", "remove"} <= set(result)
                        or not isinstance(result.get("add"), list)
                        or not isinstance(result.get("remove"), list)):
                    if reason == "length":
                        raise IncompleteResponse(
                            f"Incomplete response: length (max_tokens={self.max_output})"
                        )
                    raise ValueError("Response must contain add and remove arrays")
                # Models sometimes add commentary keys; keep only the two operations.
                result = {"add": result["add"], "remove": result["remove"]}
                metadata = {"cache_hit": False, "elapsed_s": time.monotonic() - started,
                            "usage": body.get("usage", {}), "request_hash": key,
                            "input_byte_estimate": estimated, "attempts": attempt + 1}
                saved = {"result": result, "metadata": metadata}
                temporary = target.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
                temporary.replace(target)
                return result, metadata
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
            except (httpx.TransportError, ValueError, KeyError, IndexError):
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError("Unreachable retry state")
