"""Cleaning and approval gates for collected knowledge and training records."""

from __future__ import annotations

from hashlib import sha256
import html
import re
from typing import Any, Iterable


MAX_CLEAN_CHARS = 5_000
_MALICIOUS_PATTERNS = (
    ("prompt_injection", re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.I)),
    ("instruction_override", re.compile(r"(?:system|developer)\s+message\s*[:=]", re.I)),
    ("shell_execution", re.compile(r"(?:curl|wget)\s+[^\n|]+\|\s*(?:sh|bash)|rm\s+-rf\s+", re.I)),
    ("credential_theft", re.compile(r"(?:steal|dump|exfiltrate)\s+(?:passwords?|tokens?|credentials?)", re.I)),
    ("script_payload", re.compile(r"<\s*script\b|javascript\s*:", re.I)),
)


def clean_text(value: str, limit: int = MAX_CLEAN_CHARS) -> str:
    """Normalize untrusted text without executing or preserving markup payloads."""
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = "".join(character for character in text if character in "\n\t" or ord(character) >= 32)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def scan_malicious_content(value: str) -> list[str]:
    """Return named findings; callers must reject findings before training."""
    return [name for name, pattern in _MALICIOUS_PATTERNS if pattern.search(value or "")]


def categorize(text: str) -> str:
    normalized = (text or "").lower()
    categories = (
        ("security", ("security", "vulnerability", "exploit", "credential")),
        ("programming", ("python", "code", "api", "software", "programming")),
        ("infrastructure", ("server", "hosting", "deployment", "database")),
        ("ai", ("model", "machine learning", "artificial intelligence", "training")),
    )
    for category, terms in categories:
        if any(term in normalized for term in terms):
            return category
    return "general"


def prepare_collected_record(record: dict[str, Any]) -> dict[str, Any]:
    """Clean, categorize, hash, and quarantine a newly collected record."""
    title = clean_text(str(record.get("title", "")), 500)
    summary = clean_text(str(record.get("summary", record.get("content", ""))))
    combined = f"{title}\n{summary}".strip()
    findings = scan_malicious_content(combined)
    return {
        **record,
        "title": title,
        "summary": summary,
        "content": summary,
        "category": categorize(combined),
        "dedupe_hash": sha256(combined.lower().encode("utf-8")).hexdigest(),
        "malicious_findings": findings,
        "malicious": bool(findings),
        "approval_status": "rejected" if findings else "pending",
        "verification_status": "rejected" if findings else "unverified",
        "trainable": False,
    }


def deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        prepared = prepare_collected_record(record)
        # Refreshes re-scan cached records, but must not erase an administrator's
        # review or a corroboration result that is still attached to that record.
        for key in ("approval_status", "reviewed_by", "reviewed_at", "verification_status", "source_count"):
            if key in record:
                prepared[key] = record[key]
        key = str(prepared["dedupe_hash"])
        if key not in unique:
            unique[key] = prepared
    return list(unique.values())


def is_approved_training_record(record: dict[str, Any]) -> bool:
    """Only curated or explicitly approved, verified records may enter training."""
    if record.get("malicious") or record.get("malicious_findings"):
        return False
    approval = record.get("approval_status", record.get("approval", ""))
    verification = record.get("verification_status", record.get("verification", ""))
    source_type = str(record.get("source_type", "curated_local"))
    if source_type in {"allowlisted_web", "web", "web_collection"}:
        return approval == "approved" and verification in {"corroborated", "admin_approved"}
    return approval in {"approved", "admin_approved", "curated"} or (
        source_type == "curated_local" and not approval and not verification
    )


def approved_training_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a clean, deduplicated training set; unverified records are excluded."""
    approved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not is_approved_training_record(record):
            continue
        instruction = clean_text(str(record.get("instruction", "")), 1_000)
        output = clean_text(str(record.get("output", "")))
        if not instruction or not output or scan_malicious_content(f"{instruction}\n{output}"):
            continue
        key = sha256(f"{instruction.lower()}\n{output.lower()}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        approved.append({**record, "instruction": instruction, "output": output, "trainable": True})
    return approved
