"""Small, reliable responses for routine conversational intents."""

from __future__ import annotations

import re


_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))(?:\s+there)?"
    r"(?:[,!?.\s]+(?:how\s+are\s+you|how's\s+it\s+going))?[!?.\s]*$",
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
_THANKS = re.compile(
    r"^\s*(?:thanks?|thank\s+you)(?:\s+(?:so\s+much|very\s+much|for\s+your\s+help))?"
    r"[!?.\s]*$",
    re.IGNORECASE,
)
_CONFUSION = re.compile(
    r"^\s*(?:i\s+don't\s+understand|i\s+do\s+not\s+understand|"
    r"(?:please\s+)?explain\s+(?:that|this)\s+(?:again|more\s+simply)|"
    r"what\s+do(?:es)?\s+that\s+mean)[!?.\s]*$",
    re.IGNORECASE,
)
_REPEAT = re.compile(
    r"^\s*(?:can|could|would)\s+you\s+(?:please\s+)?repeat(?:\s+that|\s+it)?"
    r"[!?.\s]*$",
    re.IGNORECASE,
)
_WHO_ARE_YOU = re.compile(
    r"^\s*(?:who\s+are\s+you|what(?:'s|\s+is)\s+your\s+name|"
    r"what\s+should\s+i\s+call\s+you)[!?.\s]*$",
    re.IGNORECASE,
)
_HELP = re.compile(
    r"^\s*(?:(?:can|could|would)\s+you\s+(?:please\s+)?help\s+me|"
    r"i\s+(?:have|'ve\s+got)\s+a\s+question)[!?.\s]*$",
    re.IGNORECASE,
)
_WHAT_DOING = re.compile(
    r"^\s*(?:what\s+are\s+you\s+doing|are\s+you\s+(?:there|available|listening))"
    r"[!?.\s]*$",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT = re.compile(
    r"^\s*(?:okay|ok|alright|got\s+it|that\s+makes\s+sense|understood)"
    r"[!?.\s]*$",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"^\s*(?:no,?\s+)?(?:that(?:'s|\s+is)\s+not\s+what\s+i\s+meant|"
    r"you\s+misunderstood|let\s+me\s+clarify)[!?.\s]*$",
    re.IGNORECASE,
)
_BENGALI_GREETING = re.compile(r"^\s*(?:হ্যালো|হাই|নমস্কার)[!?.।\s]*$", re.IGNORECASE)
_BENGALI_IDENTITY = re.compile(r"^\s*(?:তুমি কে|আপনার নাম কী|তোমার নাম কী)[!?.।\s]*$", re.IGNORECASE)


def quick_response(prompt: str) -> str | None:
    """Return a stable response for a routine prompt, or None for model work."""
    if _BENGALI_GREETING.fullmatch(prompt):
        return "হ্যালো! আমি LUSAS AI। কীভাবে সাহায্য করতে পারি?"
    if _BENGALI_IDENTITY.fullmatch(prompt):
        return "আমি LUSAS AI, আপনার স্থানীয় সহকারী।"
    if _GREETING.fullmatch(prompt):
        return "Hello! I'm LUSAS AI. How can I help?"
    if _WELLBEING.fullmatch(prompt):
        return "I'm ready to help with your local project. What would you like to work on?"
    if _THANKS.fullmatch(prompt):
        return "You're welcome! Let me know what you would like to do next."
    if _CONFUSION.fullmatch(prompt):
        return "No problem. Tell me which part is unclear and I'll explain it more simply."
    if _REPEAT.fullmatch(prompt):
        return "Sure. Tell me which part you want me to repeat."
    if _WHO_ARE_YOU.fullmatch(prompt):
        return (
            "I am LUSAS AI, a local assistant for software development, learning, "
            "and project work."
        )
    if _HELP.fullmatch(prompt):
        return "Yes—tell me what you would like help with."
    if _WHAT_DOING.fullmatch(prompt):
        return "I'm here and ready to help with your question or project."
    if _ACKNOWLEDGEMENT.fullmatch(prompt):
        return "Great. If you want, we can continue with a question or a small example."
    if _CORRECTION.fullmatch(prompt):
        return "Thanks for correcting me. Please restate the goal in your own words."
    if _KNOWS_USER.fullmatch(prompt):
        return (
            "Based on what you've shared locally, you are Rakib Chowdhury, also known "
            "as Lusa Chowdhury, the creator and developer of LUSAS AI. I only know "
            "information you choose to provide."
        )
    if _PARAMETERS.fullmatch(prompt):
        return (
            "Sara 1.0 and Lira 1.0 are the official LUSAS AI model names. Their "
            "parameter counts depend on the explicitly configured foundation for "
            "each model; LUSAS does not silently substitute another model."
        )
    if _CAPABILITIES.fullmatch(prompt):
        return (
            "I know information stored in my local knowledge base: project facts, "
            "approved learning, configured web references, and workspace files. I "
            "don't know private details unless you provide them. Ask about a specific "
            "topic and I'll answer from the information available locally."
        )
    return None
