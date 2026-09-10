from __future__ import annotations

from pathlib import Path
from typing import Any


class LocalModelError(RuntimeError):
    """Raised when the local LUSAS model cannot answer."""


class LocalModel:
    def __init__(
        self,
        model_path: Path,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 8192,
        seed: int | None = 42,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.generation_options = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "num_ctx": num_ctx,
            "seed": seed,
        }
        self._runtime: tuple[Any, Any, Any, str] | None = None

    def _load(self) -> tuple[Any, Any, Any, str]:
        if self._runtime is None:
            if not self.model_path.is_dir():
                raise LocalModelError(
                    f"LUSAS model not found at {self.model_path}. "
                    "Train and promote a model before starting chat."
                )
            try:
                from training.run_model import load_model

                self._runtime = load_model(self.model_path)
            except (ImportError, OSError, RuntimeError) as exc:
                raise LocalModelError(
                    "Could not load the local LUSAS model. "
                    "Install training/requirements.txt and verify the model files."
                ) from exc
        return self._runtime

    def chat(self, prompt: str) -> str:
        try:
            from training.run_model import generate_loaded

            torch, tokenizer, model, device = self._load()
            return generate_loaded(
                torch,
                tokenizer,
                model,
                device,
                prompt,
                self.max_new_tokens,
                **self.generation_options,
            )
        except LocalModelError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise LocalModelError(
                f"The local LUSAS model failed while generating a response: {exc}"
            ) from exc
