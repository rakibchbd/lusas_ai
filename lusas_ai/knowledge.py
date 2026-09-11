"""Structured, local knowledge records derived from bounded web evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .config import Settings


SEED_FACTS: tuple[dict[str, Any], ...] = (
    {
        "knowledge_id": "lusas-founder",
        "topic": "LUSAS AI founder and creator",
        "content": "LUSAS AI was created and developed by Lusa Chowdhury (Rakib), also known as Rakib Chowdhury.",
        "facts": [
            {
                "subject": "LUSAS AI",
                "relation": "creator",
                "object": "Lusa Chowdhury (Rakib), also known as Rakib Chowdhury",
            }
        ],
        "related_topics": ["founder", "creator", "Lusa", "Rakib Chowdhury"],
    },
    {
        "knowledge_id": "lusas-name-origin",
        "topic": "LUSAS AI name origin",
        "content": "The name LUSAS AI was inspired by Lusa, Rakib Chowdhury's childhood nickname used by family and people from his village.",
        "facts": [
            {
                "subject": "LUSAS AI",
                "relation": "name_origin",
                "object": "Lusa, a childhood nickname of Rakib Chowdhury",
            }
        ],
        "related_topics": ["origin", "nickname", "history"],
    },
    {
        "knowledge_id": "lusas-founder-background",
        "topic": "Rakib Chowdhury technical background",
        "content": "Rakib Chowdhury began learning web development in 2011, studied Python and machine learning from 2017, and has worked as a Systems Engineer since 2021 in servers, hosting, administration, deployment, and backend technologies.",
        "facts": [
            {
                "subject": "Rakib Chowdhury",
                "relation": "technical_background",
                "object": "web development, Python, machine learning, systems engineering, servers, hosting, deployment, and backend technologies",
            }
        ],
        "related_topics": ["background", "Python", "machine learning", "Systems Engineer"],
    },
    {
        "knowledge_id": "lusas-founder-profile",
        "topic": "Rakib Chowdhury personal profile",
        "content": "Rakib Chowdhury was born on November 15, 1999, in Darshana Mor, Rangpur City, Bangladesh. His work includes web development, Python, machine learning, systems engineering, servers, hosting, deployment, and backend technologies.",
        "facts": [
            {
                "subject": "Rakib Chowdhury",
                "relation": "birth_date",
                "object": "November 15, 1999",
            },
            {
                "subject": "Rakib Chowdhury",
                "relation": "birth_place",
                "object": "Darshana Mor, Rangpur City, Bangladesh",
            },
        ],
        "related_topics": [
            "profile",
            "born",
            "birth",
            "November 15, 1999",
            "Darshana Mor",
            "Rangpur City",
            "Bangladesh",
        ],
    },
)


_RETRIEVAL_CACHE: dict[Path, tuple[int, list[dict[str, Any]]]] = {}
_RESEARCH_TERMS = {
    "changelog",
    "changed",
    "changes",
    "current",
    "documentation",
    "docs",
    "latest",
    "news",
    "recent",
    "recently",
    "release",
    "releases",
    "update",
    "updates",
    "version",
}
_UNSUPPORTED_PERSONAL_CLAIMS = (
    "born",
    "birth",
    "city",
    "country",
    "former",
    "his name is",
    "her name is",
    "known for",
    "lives in",
    "startup",
    "based in",
    "based",
    "headquartered",
    "located in",
    "company",
    "platform",
    "known for",
    "founded in",
    "established in",
    "organization",
)
_CONTRADICTORY_IDENTITY_CLAIMS = (
    "he was created",
    "she was created",
    "he is an ai",
    "she is an ai",
)
_INCOMPLETE_ENDINGS = re.compile(
    r"\b(?:also|and|or|the|a|an|to|of|in|for|with|has|have)$",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_BIRTH_LOCATION = re.compile(
    r"\b(?:born|birth)\b.*?\bin\s+([^.!?]+)",
    re.IGNORECASE,
)


def _identity_sentence_is_supported(sentence: str, supplied_context: str) -> bool:
    """Reject unsupported dates, places, and organization claims in biographies."""
    sentence_lower = sentence.lower()
    context_lower = supplied_context.lower()
    sentence_years = set(_YEAR.findall(sentence))
    context_years = set(_YEAR.findall(supplied_context))
    if sentence_years and not sentence_years.issubset(context_years):
        return False

    if "born" in sentence_lower or "birth" in sentence_lower:
        sentence_location = _BIRTH_LOCATION.search(sentence)
        context_location = _BIRTH_LOCATION.search(supplied_context)
        if sentence_location and context_location:
            sentence_tokens = set(_terms(sentence_location.group(1)))
            context_tokens = set(_terms(context_location.group(1)))
            if len(sentence_tokens.intersection(context_tokens)) < 2:
                return False

    for marker in (
        "based in",
        "headquartered",
        "located in",
        "company",
        "platform",
        "known for",
    ):
        if marker in sentence_lower and marker not in context_lower:
            return False
    return True


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


def _terms(text: str) -> set[str]:
    aliases = {
        "made": "creator",
        "created": "creator",
        "developed": "creator",
        "developer": "creator",
        "founded": "creator",
        "founder": "creator",
        "maker": "creator",
        "built": "creator",
        "story": "origin",
        "history": "origin",
    }
    return {
        aliases.get(term, term)
        for term in re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower())
    }


def _confidence(verification_status: str) -> float:
    return {
        "corroborated": 0.8,
        "unverified": 0.5,
        "conflicting": 0.25,
    }.get(verification_status, 0.25)


def _concepts(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9_-]{3,}", text.lower())
    return sorted(set(words))[:20]


def _seed_record(seed: dict[str, Any], now: str) -> dict[str, Any]:
    index_text = " ".join(
        [seed["topic"], seed["content"], *seed.get("related_topics", [])]
    )
    return {
        "id": seed["knowledge_id"],
        "knowledge_id": seed["knowledge_id"],
        "kind": "fact",
        "topic": seed["topic"],
        "content": seed["content"],
        "facts": seed["facts"],
        "source": "user-provided project introduction",
        "source_type": "user_introduction",
        "created_at": now,
        "updated_at": now,
        "confidence": 1.0,
        "verification_status": "user_provided",
        "verification": "user_provided",
        "related_topics": seed.get("related_topics", []),
        "dependencies": [],
        "embeddings": None,
        "usage_count": 0,
        "last_used": None,
        "index_terms": sorted(_terms(index_text)),
        "scope": "personal_identity",
        "domain": "general",
        "concepts": _concepts(index_text),
    }


def ensure_seed(settings: Settings) -> list[dict[str, Any]]:
    """Store the introduction as separate facts, not as a response template."""
    current = _read(settings.knowledge_path)
    by_id = {item.get("knowledge_id", item.get("id")): item for item in current}
    now = datetime.now(timezone.utc).isoformat()
    for seed in SEED_FACTS:
        if seed["knowledge_id"] not in by_id:
            by_id[seed["knowledge_id"]] = _seed_record(seed, now)
    result = list(by_id.values())
    if result != current:
        _write(settings.knowledge_path, result)
    _RETRIEVAL_CACHE.pop(settings.knowledge_path, None)
    return result


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
        approval_status = article.get(
            "approval_status",
            previous.get("approval_status", "pending") if previous else "pending",
        )
        previous_observed = previous.get("observed_at") if previous else None
        expires_at = (
            previous.get("expires_at")
            if previous and previous_observed == observed
            else (now + timedelta(days=30)).isoformat()
        )
        index_text = f"{article.get('title', '')} {article.get('summary', '')}"
        existing[article_id] = {
            "id": article_id,
            "knowledge_id": article_id,
            "kind": "web_evidence",
            "topic": article.get("title", ""),
            "content": article.get("summary", ""),
            "facts": [article.get("summary", "")],
            "claim": article.get("summary", ""),
            "title": article.get("title", ""),
            "source": article.get("url", ""),
            "source_feed": article.get("source", ""),
            "source_type": "allowlisted_web",
            "created_at": previous.get("created_at", observed) if previous else observed,
            "updated_at": now.isoformat(),
            "observed_at": observed,
            "checked_at": now.isoformat(),
            "expires_at": expires_at,
            "verification": verification,
            "verification_status": verification,
            "approval_status": approval_status,
            "reviewed_by": article.get("reviewed_by", previous.get("reviewed_by") if previous else None),
            "reviewed_at": article.get("reviewed_at", previous.get("reviewed_at") if previous else None),
            "category": article.get("category", "general"),
            "malicious_findings": article.get("malicious_findings", []),
            "malicious": bool(article.get("malicious", False)),
            "trainable": False,
            "confidence": _confidence(verification),
            "related_topics": [],
            "dependencies": [],
            "embeddings": None,
            "usage_count": previous.get("usage_count", 0) if previous else 0,
            "last_used": previous.get("last_used") if previous else None,
            "index_terms": sorted(_terms(index_text)),
            "scope": "general",
            "domain": _domain(
                f"{article.get('title', '')} {article.get('summary', '')}"
            ),
            "concepts": _concepts(
                f"{article.get('title', '')} {article.get('summary', '')}"
            ),
        }
    seed_records = [item for item in existing.values() if item.get("kind") == "fact"]
    web_records = [item for item in existing.values() if item.get("kind") != "fact"]
    records = seed_records + web_records[-max(1, settings.web_max_items) :]
    _write(settings.knowledge_path, records)
    _RETRIEVAL_CACHE.pop(settings.knowledge_path, None)
    return len(records)


def ingest_learning(
    settings: Settings,
    instruction: str,
    output: str,
    *,
    source_type: str = "approved_learning",
) -> dict[str, Any]:
    """Normalize an approved example into searchable knowledge alongside training data."""
    content = re.sub(r"\s+", " ", output).strip()[:5_000]
    instruction = re.sub(r"\s+", " ", instruction).strip()[:1_000]
    if not instruction or not content:
        raise ValueError("Learning content cannot be empty.")
    knowledge_id = sha256(f"{instruction}\n{content}".encode("utf-8")).hexdigest()
    current = _read(settings.knowledge_path)
    previous = next((item for item in current if item.get("knowledge_id") == knowledge_id), None)
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id": knowledge_id,
        "knowledge_id": knowledge_id,
        "kind": "learned_fact",
        "topic": instruction,
        "content": content,
        "facts": [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", content) if sentence.strip()][:20],
        "source": "approved local learning example",
        "source_type": source_type,
        "created_at": previous.get("created_at", now) if previous else now,
        "updated_at": now,
        "confidence": 0.7,
        "verification_status": "user_provided",
        "verification": "user_provided",
        "related_topics": sorted(_terms(f"{instruction} {content}"))[:20],
        "dependencies": [],
        "embeddings": None,
        "usage_count": previous.get("usage_count", 0) if previous else 0,
        "last_used": previous.get("last_used") if previous else None,
        "index_terms": sorted(_terms(f"{instruction} {content}")),
        "scope": "general",
        "domain": _domain(f"{instruction} {content}"),
        "concepts": _concepts(f"{instruction} {content}"),
    }
    updated = [item for item in current if item.get("knowledge_id") != knowledge_id]
    updated.append(record)
    _write(settings.knowledge_path, updated)
    _RETRIEVAL_CACHE.pop(settings.knowledge_path, None)
    return record


def records(settings: Settings) -> list[dict[str, Any]]:
    return _read(settings.knowledge_path)


def _cached_records(settings: Settings) -> list[dict[str, Any]]:
    path = settings.knowledge_path
    try:
        signature = path.stat().st_mtime_ns
    except OSError:
        return []
    cached = _RETRIEVAL_CACHE.get(path)
    if cached and cached[0] == signature:
        return cached[1]
    loaded = _read(path)
    _RETRIEVAL_CACHE[path] = (signature, loaded)
    return loaded


def _retrieve_records(
    query: str, source_records: list[dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    """Rank normalized records without turning their prose into instructions."""
    query_terms = _terms(query)
    if not query_terms:
        return []
    identity_query = bool(
        query_terms.intersection({"creator", "lusas", "lusa", "rakib", "origin"})
    )
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in source_records:
        if (
            item.get("kind") == "web_evidence"
            and (
                not query_terms.intersection(_RESEARCH_TERMS)
                or item.get("approval_status") != "approved"
                or item.get("verification_status") not in {"corroborated", "admin_approved"}
            )
        ):
            continue
        if item.get("scope") == "personal_identity" and not identity_query:
            continue
        indexed = set(item.get("index_terms", []))
        if not indexed:
            indexed = _terms(
                f"{item.get('topic', '')} {item.get('content', '')} "
                f"{' '.join(str(item.get('related_topics', [])))}"
            )
        score = len(query_terms.intersection(indexed))
        if score:
            item["usage_count"] = int(item.get("usage_count", 0)) + 1
            item["last_used"] = datetime.now(timezone.utc).isoformat()
            scored.append((score, item))
    scored.sort(
        key=lambda pair: (pair[0], str(pair[1].get("updated_at", ""))),
        reverse=True,
    )
    return [item for _, item in scored[: max(1, limit)]]


def retrieve(settings: Settings, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant facts with normalized terms and an in-process index cache."""
    return _retrieve_records(query, _cached_records(settings), limit=limit)


def _format_context(matches: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in matches:
        facts = item.get("facts", [])
        fact_text = "; ".join(
            f"{fact.get('subject')} {fact.get('relation')} {fact.get('object')}"
            if isinstance(fact, dict)
            else str(fact)
            for fact in facts[:10]
        )
        chunks.append(
            f"Topic: {item.get('topic', '')}\n"
            f"Knowledge status: {item.get('verification_status', 'uncertain')}\n"
            f"Facts: {fact_text or item.get('content', item.get('claim', ''))}"
        )
    return "\n\n".join(chunks)


def context(settings: Settings, query: str, limit: int = 5) -> str:
    """Format only relevant structured knowledge for the model's context layer."""
    return _format_context(retrieve(settings, query, limit=limit))


def seed_context(query: str, limit: int = 5) -> str:
    """Return relevant built-in facts for direct model runs without a project path."""
    now = datetime.now(timezone.utc).isoformat()
    seed_records = [_seed_record(seed, now) for seed in SEED_FACTS]
    return _format_context(_retrieve_records(query, seed_records, limit=limit))


def ground_response(query: str, response: str, supplied_context: str) -> str:
    """Drop unsupported biography claims while preserving the model's wording."""
    from .identity import (
        FIRST_PERSON_PERSON_IDENTIFICATION,
        is_creator_question,
        is_person_identity_question,
    )

    identity_query = is_creator_question(query) or is_person_identity_question(query)
    if not supplied_context or not identity_query:
        return response.strip()
    if not response.strip() or response.strip().lower() == "i could not produce a clean answer. please try again.":
        if is_person_identity_question(query):
            return _person_identity_fallback(query, supplied_context)
        return "I don't know that detail yet."
    context_terms = _terms(supplied_context)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", response)
        if sentence.strip()
    ]
    grounded: list[str] = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if (
            is_person_identity_question(query)
            and FIRST_PERSON_PERSON_IDENTIFICATION.search(sentence)
        ):
            continue
        if is_person_identity_question(query) and not _identity_sentence_is_supported(
            sentence, supplied_context
        ):
            continue
        if any(
            marker in sentence_lower and marker not in supplied_context.lower()
            for marker in _UNSUPPORTED_PERSONAL_CLAIMS
        ) or _INCOMPLETE_ENDINGS.search(sentence):
            continue
        if any(marker in sentence_lower for marker in _CONTRADICTORY_IDENTITY_CLAIMS):
            continue
        sentence_terms = _terms(sentence)
        unknown_terms = sentence_terms - context_terms
        # A few connective words are expected in a natural answer. A sentence
        # dominated by terms absent from the supplied facts is likely a model
        # hallucination, so omit it instead of presenting it as known.
        if len(unknown_terms) <= max(4, len(sentence_terms) // 2):
            grounded.append(sentence)
    result = " ".join(grounded).strip()
    if result:
        return result
    if is_person_identity_question(query):
        return _person_identity_fallback(query, supplied_context)
    return "I don't know that detail yet."


def _person_identity_fallback(query: str, supplied_context: str) -> str:
    """Render a concise answer from the structured facts if generation fails."""
    query_lower = query.lower()
    if "rakib" in query_lower:
        person = "Rakib Chowdhury"
        pronoun = "his"
    else:
        person = "Lusa Chowdhury (Rakib)"
        pronoun = "his"

    context_lower = supplied_context.lower()
    parts: list[str] = []
    if "lusas ai creator" in context_lower:
        parts.append(f"{person} is the creator and developer of LUSAS AI.")
    if "childhood nickname" in context_lower:
        parts.append(f"Lusa is {pronoun} childhood nickname.")
    asks_about_background = bool(
        _terms(query).intersection(
            {"background", "experience", "profession", "work", "career", "technical"}
        )
    )
    if asks_about_background and "technical background" in context_lower:
        parts.append(
            "His background includes web development, Python, machine learning, "
            "systems engineering, servers, hosting, deployment, and backend technologies."
        )
    return " ".join(parts) or "I don't know that detail yet."


def known_person_response(query: str, supplied_context: str) -> str:
    """Answer a known personal-identity question from structured facts only."""
    return _person_identity_fallback(query, supplied_context)


def needs_response_repair(
    query: str, response: str, supplied_context: str = ""
) -> bool:
    """Detect a likely unrelated project-identity tangent in a model answer."""
    from .identity import FIRST_PERSON_PERSON_IDENTIFICATION, is_person_identity_question

    if supplied_context:
        if not is_person_identity_question(query):
            return False
        response_lower = response.lower()
        wrong_perspective = FIRST_PERSON_PERSON_IDENTIFICATION.search(response)
        missing_project_relationship = not (
            "lusas" in response_lower
            and any(
                role in response_lower
                for role in ("creator", "created", "developer", "developed", "founder")
            )
        )
        return bool(wrong_perspective or missing_project_relationship)
    query_terms = _terms(query)
    if query_terms.intersection({"lusa", "lusas", "rakib"}):
        return False
    return bool(_terms(response).intersection({"lusa", "lusas", "rakib"}))


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
