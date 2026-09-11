from __future__ import annotations

import re


CREATOR_QUESTION = re.compile(
    r"\b(?:who|what)\s+(?:made|created|built|developed|designed|authored|founded)\s+you\b"
    r"|\b(?:who|what)\s+is\s+your\s+(?:creator|maker|developer|author|designer|founder)\b"
    r"|\bwho\s+developed\s+you\b"
    r"|\bwho\s+(?:made|created|developed|built|founded)\s+(?:lusas|lusa)\b"
    r"|\bwho\s+the\s+(?:creator|maker|developer|author|designer|founder)\s+is\b"
    r"|\bwho\s+is\s+the\s+(?:creator|maker|developer|author|designer|founder)"
    r"(?:\s+of\s+(?:lusas(?:\s+ai)?|lusa))?\b"
    r"|\b(?:creator|maker|developer|author|designer|founder)\s+of\s+(?:lusas(?:\s+ai)?|lusa)\b"
    r"|\btell\s+me\s+about\s+(?:your\s+)?(?:creator|maker|developer|founder)\b"
    r"|\bwhat(?:'s|\s+is)\s+the\s+story\s+behind\s+(?:lusas(?:\s+ai)?|lusa)\b"
    r"|\bwhat\s+is\s+your\s+identity\b",
    re.IGNORECASE,
)


_INTERNAL_TAIL = re.compile(
    r"\s*(?:###\s*System:|System instructions:|Workspace context:|"
    r"###\s*Instruction:|an\s+internal\s+provider\s+policy\b|"
    r"never\s+\w+\s+that\s+.*(?:provider|company)\b|"
    r"web\s+excerpts\s+are\s+untrusted\b|"
    r"you\s+help\s+with\s+software\s+in\s+the\s+configured\s+workspace\b|"
    r"do\s+not\s+access\s+credentials\b)",
    re.IGNORECASE,
)


def strip_internal_prompt_leak(response: str) -> str:
    """Stop generated text when a model starts reproducing hidden instructions."""
    match = _INTERNAL_TAIL.search(response)
    if match:
        response = response[:match.start()]
    return response.strip() or "I could not produce a clean answer. Please try again."


def is_creator_question(prompt: str) -> bool:
    return bool(CREATOR_QUESTION.search(prompt))
