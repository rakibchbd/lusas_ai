"""Small, reliable responses for routine conversational intents."""

from __future__ import annotations

import re


_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))(?:\s+there)?[!?.\s]*$",
    re.IGNORECASE,
)
_WELLBEING = re.compile(
    r"^\s*(?:how\s+are\s+you|how's\s+it\s+going|how\s+are\s+things)[!?.\s]*$",
    re.IGNORECASE,
)
_KNOWS_USER = re.compile(
    r"^\s*(?:do\s+you\s+know\s+me|do\s+you\s+remember\s+me)[!?.\s]*$",
    re.IGNORECASE,
)
_CAPABILITIES = re.compile(
    r"^\s*(?:what\s+(?:do\s+you\s+know|you\s+know)|"
    r"what(?:'s|\s+is)\s+your\s+(?:capabilities|knowledge)|"
    r"what\s+can\s+you\s+do|tell\s+me\s+(?:what\s+you\s+know|about\s+yourself))"
    r"[!?.\s]*$",
    re.IGNORECASE,
)
_PARAMETERS = re.compile(
    r"^\s*(?:how\s+many\s+(?:parameters?|peremeters?|perameters?)\b.*|"
    r"how\s+large\s+is\s+your\s+model)[!?.\s]*$",
    re.IGNORECASE,
)


def quick_response(prompt: str) -> str | None:
    """Return a stable response for a routine prompt, or None for model work."""
    if _GREETING.fullmatch(prompt):
        return "Hello! I'm LUSAS AI. How can I help?"
    if _WELLBEING.fullmatch(prompt):
        return "I'm ready to help with your local project. What would you like to work on?"
    if _KNOWS_USER.fullmatch(prompt):
        return (
            "Based on what you've shared locally, you are Rakib Chowdhury, also known "
            "as Lusa Chowdhury, the creator and developer of LUSAS AI. I only know "
            "information you choose to provide."
        )
    if _PARAMETERS.fullmatch(prompt):
        return (
            "The local base model is Qwen2.5-Coder-0.5B-Instruct, which has about "
            "0.5 billion parameters. LUSAS also uses a small LoRA adapter containing "
            "additional trained weights."
        )
    if _CAPABILITIES.fullmatch(prompt):
        return (
            "I know information stored in my local knowledge base: project facts, "
            "approved learning, configured web references, and workspace files. I "
            "don't know private details unless you provide them. Ask about a specific "
            "topic and I'll answer from the information available locally."
        )
    return None
