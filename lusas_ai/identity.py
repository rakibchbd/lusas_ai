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


PERSON_IDENTITY_QUESTION = re.compile(
    r"\b(?:who|what)\s+is\s+(?:rakib(?:\s+chowdhury)?|lusa(?:\s+chowdhury)?)\b"
    r"|\btell\s+me\s+about\s+(?:rakib(?:\s+chowdhury)?|lusa(?:\s+chowdhury)?)\b",
    re.IGNORECASE,
)


FIRST_PERSON_PERSON_IDENTIFICATION = re.compile(
    r"\bI\s+(?:am|'m|was)\s+(?:Rakib(?:\s+Chowdhury)?|Lusa(?:\s+Chowdhury)?)\b",
    re.IGNORECASE,
)


_NAME_ONLY_INPUT = re.compile(
    r"^[A-Za-z][A-Za-z.'’-]{1,}(?:\s+[A-Za-z][A-Za-z.'’-]{1,}){1,3}[?.!,]*$"
)
_NAME_INPUT_EXCLUSIONS = {
    "about",
    "and",
    "api",
    "are",
    "build",
    "code",
    "create",
    "explain",
    "for",
    "function",
    "generate",
    "help",
    "how",
    "is",
    "make",
    "program",
    "python",
    "script",
    "show",
    "tell",
    "the",
    "what",
    "who",
    "why",
    "write",
}
_PERSON_QUESTION_EXCLUSIONS = _NAME_INPUT_EXCLUSIONS | {
    "chatgpt",
    "docker",
    "github",
    "java",
    "javascript",
    "linux",
    "macos",
    "model",
    "openai",
    "rust",
    "system",
    "typescript",
}
_WHO_IS_PERSON = re.compile(
    r"^\s*who\s+is\s+((?:[A-Za-z][A-Za-z.'’-]{1,}\s*){1,4})[?.!,]*$",
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


def is_person_identity_question(prompt: str) -> bool:
    return bool(PERSON_IDENTITY_QUESTION.search(prompt))


def is_ambiguous_name_prompt(prompt: str) -> bool:
    """Recognize a short name fragment that needs clarification before answering."""
    normalized = " ".join(prompt.strip().split())
    words = normalized.rstrip("?.!,").split()
    if not 2 <= len(words) <= 4 or not _NAME_ONLY_INPUT.fullmatch(normalized):
        return False
    return not any(word.lower() in _NAME_INPUT_EXCLUSIONS for word in words)


def ambiguous_name_response(prompt: str) -> str:
    """Ask for intent without inventing facts, URLs, or a profile for the name."""
    name = " ".join(
        word[:1].upper() + word[1:]
        for word in prompt.strip().rstrip("?.!,").split()
    )
    return f"I don't have verified information about {name} yet. What would you like to know?"


def unknown_person_subject(prompt: str) -> str | None:
    """Return an unsupported person's name from a simple identity question."""
    match = _WHO_IS_PERSON.fullmatch(prompt)
    if not match:
        return None
    subject = " ".join(match.group(1).split())
    if any(word.lower() in _PERSON_QUESTION_EXCLUSIONS for word in subject.split()):
        return None
    return subject


def unknown_person_response(prompt: str) -> str:
    """Ask for context instead of letting the model invent a person's identity."""
    subject = unknown_person_subject(prompt) or "that person"
    name = " ".join(
        word[:1].upper() + word[1:]
        for word in subject.rstrip("?.!,").split()
    )
    return f"I don't have verified information about {name} yet. What would you like to know?"
