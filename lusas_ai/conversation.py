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
_CAPABILITIES = re.compile(
    r"^\s*(?:what\s+(?:do\s+you\s+know|you\s+know)|"
    r"what(?:'s|\s+is)\s+your\s+(?:capabilities|knowledge)|"
    r"what\s+can\s+you\s+do|tell\s+me\s+(?:what\s+you\s+know|about\s+yourself))"
    r"[!?.\s]*$",
    re.IGNORECASE,
)


def quick_response(prompt: str) -> str | None:
    """Return a stable response for a routine prompt, or None for model work."""
    if _GREETING.fullmatch(prompt):
        return "Hello! I'm LUSAS AI. How can I help?"
    if _WELLBEING.fullmatch(prompt):
        return "I'm ready to help with your local project. What would you like to work on?"
    if _CAPABILITIES.fullmatch(prompt):
        return (
            "I work from the local model, structured project knowledge, approved local "
            "learning, configured web references, and files in the local workspace. "
            "I can help with software development, explanations, testing, and guarded "
            "upgrades with backups and rollback."
        )
    return None
