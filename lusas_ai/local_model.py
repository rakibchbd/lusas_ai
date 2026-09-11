from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .model_registry import foundation_config, get_model_spec, stable_path


class LocalModelError(RuntimeError):
    """Raised when the selected official model is unavailable or cannot answer."""


class LocalModel:
    def __init__(
        self,
        settings: Settings,
        model_id: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        num_ctx: int = 8192,
        seed: int | None = 42,
    ) -> None:
        get_model_spec(model_id)
        self.settings = settings
        self.model_id = model_id
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

    @property
    def model_path(self) -> Path:
        return stable_path(self.settings, self.model_id)

    def switch(self, model_id: str) -> None:
        get_model_spec(model_id)
        if model_id != self.model_id:
            self.model_id = model_id
            self._runtime = None

    def _load(self) -> tuple[Any, Any, Any, str]:
        if self._runtime is None:
            if not self.model_path.is_dir():
                raise LocalModelError(
                    f"{get_model_spec(self.model_id).display_name} is not installed at "
                    f"{self.model_path}. Configure its foundation and deploy an approved model first."
                )
            foundation = foundation_config(self.settings, self.model_id)
            if not foundation["model"] and not foundation["path"]:
                spec = get_model_spec(self.model_id)
                raise LocalModelError(
                    f"{spec.display_name} has no configured foundation. Set "
                    f"{spec.foundation_model_env} or {spec.foundation_path_env}; no fallback model is used."
                )
            try:
                from training.run_model import load_model

                self._runtime = load_model(
                    self.model_id,
                    model_path=self.model_path,
                    settings=self.settings,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise LocalModelError(
                    f"Could not load {get_model_spec(self.model_id).display_name}. "
                    "Verify its explicitly configured foundation and model files."
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
                f"{get_model_spec(self.model_id).display_name} failed while generating a response: {exc}"
            ) from exc
