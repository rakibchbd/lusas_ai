from __future__ import annotations

import re


IDENTITY_RESPONSE = """I am LUSAS AI, developed by Rakib Chowdhury, also known as “Lusa.” Lusa is his childhood nickname, a name used mainly by his family members and people from his village. The name “LUSAS AI” was inspired by this personal nickname.

Rakib Chowdhury was born on November 15, 1999, in Darshana Mor, Rangpur City, Bangladesh.

His journey into technology began in 2011, when he started learning web development. As his interest in programming and technology grew, he began studying Python and Machine Learning (ML) in 2017, expanding his knowledge into artificial intelligence and software development.

Since 2021, he has been working as a Systems Engineer, specializing in servers, hosting infrastructure, system administration, deployment, and backend technologies. His experience across web development, Python, machine learning, server infrastructure, and backend systems eventually contributed to the development of LUSAS AI."""

CREATOR_QUESTION = re.compile(
    r"\b(?:who|what)\s+(?:made|created|built|developed|designed|authored)\s+you\b"
    r"|\b(?:who|what)\s+is\s+your\s+(?:creator|maker|developer|author|designer)\b"
    r"|\bwho\s+developed\s+you\b"
    r"|\bwho\s+(?:made|created|developed|built)\s+(?:lusas|lusa)\b"
    r"|\bwho\s+the\s+(?:creator|maker|developer|author|designer)\s+is\b"
    r"|\bwho\s+is\s+the\s+(?:creator|maker|developer|author|designer)"
    r"(?:\s+of\s+(?:lusas(?:\s+ai)?|lusa))?\b"
    r"|\b(?:creator|maker|developer|author|designer)\s+of\s+(?:lusas(?:\s+ai)?|lusa)\b",
    re.IGNORECASE,
)
