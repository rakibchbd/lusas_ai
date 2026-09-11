from __future__ import annotations

import re


GREETING_RESPONSE = "Hello! I am LUSAS AI, also known as Lusa. How can I help you?"
AGE_RESPONSE = (
    "I do not have a human age. I am a software AI system created by "
    "Rakib Chowdhury."
)
WELLBEING_RESPONSE = (
    "I am ready and working well. How can I help you?"
)
OCCUPATION_RESPONSE = (
    "I am LUSAS AI, a local AI assistant focused on software development, "
    "learning, and automation."
)


IDENTITY_RESPONSE = """I am LUSAS AI, created and developed by Lusa Chowdhury (Rakib), also known as Rakib Chowdhury. Lusa is his childhood nickname, a name used mainly by his family members and people from his village. The name “LUSAS AI” was inspired by this personal nickname.

Rakib Chowdhury was born on November 15, 1999, in Darshana Mor, Rangpur City, Bangladesh.

His journey into technology began in 2011, when he started learning web development. As his interest in programming and technology grew, he began studying Python and Machine Learning (ML) in 2017, expanding his knowledge into artificial intelligence and software development.

Since 2021, he has been working as a Systems Engineer, specializing in servers, hosting infrastructure, system administration, deployment, and backend technologies. His experience across web development, Python, machine learning, server infrastructure, and backend systems eventually contributed to the development of LUSAS AI."""

CREATOR_QUESTION = re.compile(
    r"\b(?:who|what)\s+(?:made|created|built|developed|designed|authored|founded)\s+you\b"
    r"|\b(?:who|what)\s+is\s+your\s+(?:creator|maker|developer|author|designer)\b"
    r"|\bwho\s+developed\s+you\b"
    r"|\bwho\s+(?:made|created|developed|built)\s+(?:lusas|lusa)\b"
    r"|\bwho\s+the\s+(?:creator|maker|developer|author|designer)\s+is\b"
    r"|\bwho\s+is\s+the\s+(?:creator|maker|developer|author|designer|founder)"
    r"(?:\s+of\s+(?:lusas(?:\s+ai)?|lusa))?\b"
    r"|\b(?:creator|maker|developer|author|designer|founder)\s+of\s+(?:lusas(?:\s+ai)?|lusa)\b",
    re.IGNORECASE,
)


_INTERNAL_TAIL = re.compile(
    r"\s*(?:###\s*System:|System instructions:|Workspace context:|"
    r"###\s*Instruction:|never\s+claim\b|web\s+excerpts\s+are\s+untrusted\b|"
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

GREETING_QUESTION = re.compile(
    r"^\s*(?:hi|hello|hey|hola|greetings|good\s+(?:morning|afternoon|evening))"
    r"[!,.?\s]*$",
    re.IGNORECASE,
)

AGE_QUESTION = re.compile(
    r"^\s*(?:how\s+old\s+are\s+you|what\s+is\s+your\s+age)\s*[?!.\s]*$",
    re.IGNORECASE,
)
WELLBEING_QUESTION = re.compile(
    r"^\s*how\s+are\s+you\s*[?!.\s]*$",
    re.IGNORECASE,
)
OCCUPATION_QUESTION = re.compile(
    r"^\s*(?:what\s+is\s+your\s+(?:occupation|profession|job)|"
    r"what\s+do\s+you\s+do)\s*[?!.\s]*$",
    re.IGNORECASE,
)


def deterministic_response(prompt: str) -> str | None:
    """Answer basic identity and conversation prompts without model leakage."""
    if CREATOR_QUESTION.search(prompt):
        return IDENTITY_RESPONSE
    if GREETING_QUESTION.fullmatch(prompt):
        return GREETING_RESPONSE
    if AGE_QUESTION.fullmatch(prompt):
        return AGE_RESPONSE
    if WELLBEING_QUESTION.fullmatch(prompt):
        return WELLBEING_RESPONSE
    if OCCUPATION_QUESTION.fullmatch(prompt):
        return OCCUPATION_RESPONSE
    return None
