"""Structured, local knowledge records derived from bounded web evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .config import Settings


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid knowledge record on line {line_number}.")
        records.append(payload)
    return records


def _write(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _domain(text: str) -> str:
    labels = (
        ("ai", (" ai ", "machine learning", "transformer", "model")),
        ("programming", ("python", "programming", "code", "software")),
        ("security", ("security", "vulnerability", "exploit")),
        ("performance", ("performance", "latency", "memory", "speed")),
        ("infrastructure", ("cloud", "server", "deployment", "database")),
    )
    normalized = f" {text.lower()} "
    for name, terms in labels:
        if any(term in normalized for term in terms):
            return name
    return "general"


def _confidence(verification_status: str) -> float:
    return {
        "corroborated": 0.8,
        "unverified": 0.5,
        "conflicting": 0.25,
    }.get(verification_status, 0.25)


def _concepts(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9_-]{3,}", text.lower())
    return sorted(set(words))[:20]


def upsert_web(settings: Settings, articles: list[dict[str, str]]) -> int:
    """Persist source, confidence, freshness, and graph-friendly concepts."""
    existing = {
        record.get("id"): record
        for record in _read(settings.knowledge_path)
        if record.get("id")
    }
    now = datetime.now(timezone.utc)
    for article in articles:
        article_id = article.get("id")
        if not article_id:
            continue
        verification = article.get("verification_status", "unverified")
        observed = article.get("fetched_at") or now.isoformat()
        previous = existing.get(article_id)
        previous_observed = previous.get("observed_at") if previous else None
        expires_at = (
            previous.get("expires_at")
            if previous and previous_observed == observed
            else (now + timedelta(days=30)).isoformat()
        )
        existing[article_id] = {
            "id": article_id,
            "kind": "web_evidence",
            "claim": article.get("summary", ""),
            "title": article.get("title", ""),
            "source": article.get("url", ""),
            "source_feed": article.get("source", ""),
            "observed_at": observed,
            "checked_at": now.isoformat(),
            "expires_at": expires_at,
            "verification": verification,
            "confidence": _confidence(verification),
            "domain": _domain(
                f"{article.get('title', '')} {article.get('summary', '')}"
            ),
            "concepts": _concepts(
                f"{article.get('title', '')} {article.get('summary', '')}"
            ),
        }
    records = list(existing.values())[-max(1, settings.web_max_items) :]
    _write(settings.knowledge_path, records)
    return len(records)


def records(settings: Settings) -> list[dict[str, Any]]:
    return _read(settings.knowledge_path)


def is_outdated(record: dict[str, Any], now: datetime | None = None) -> bool:
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, str):
        return False
    try:
        return (now or datetime.now(timezone.utc)) >= datetime.fromisoformat(expires_at)
    except ValueError:
        return True


def graph(settings: Settings) -> dict[str, list[dict[str, Any]]]:
    """Return a small derived knowledge graph without a database dependency."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for record in records(settings):
        node_id = str(record.get("id", ""))
        nodes.append(
            {
                "id": node_id,
                "label": record.get("title", ""),
                "domain": record.get("domain", "general"),
                "verification": record.get("verification", "unverified"),
                "outdated": is_outdated(record),
            }
        )
        for concept in record.get("concepts", [])[:20]:
            edges.append({"from": node_id, "to": concept, "type": "mentions"})
    return {"nodes": nodes, "edges": edges}
