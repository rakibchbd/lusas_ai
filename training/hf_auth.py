"""Hugging Face authentication without storing credentials in the repository."""

from __future__ import annotations

import os


def auth_kwargs() -> dict[str, str]:
    """Return an explicit Hub token only when configured in the environment."""
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"token": token} if token else {}
