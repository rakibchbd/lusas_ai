from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot answer."""


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result: dict[str, Any] = json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise OllamaError(
                "Could not reach Ollama. Start Ollama and verify ollama_url "
                "in config.json."
            ) from exc
        except TimeoutError as exc:
            raise OllamaError("Ollama did not respond before the timeout.") from exc

        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), str
        ):
            raise OllamaError("Ollama returned an unexpected response.")
        return message["content"]
