CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_catalog (
    model_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    foundation_model TEXT,
    foundation_path TEXT,
    status TEXT NOT NULL DEFAULT 'not_installed',
    stable_version TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approved_sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    domain TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    admin_approved INTEGER NOT NULL DEFAULT 0,
    approved_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL,
    malicious_findings TEXT NOT NULL DEFAULT '[]',
    reviewed_by TEXT,
    reviewed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,
    dataset_count INTEGER NOT NULL DEFAULT 0,
    candidate_path TEXT,
    metrics TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deployment_approvals (
    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    candidate_path TEXT NOT NULL,
    evaluation TEXT NOT NULL,
    status TEXT NOT NULL,
    approved_by TEXT,
    created_at TEXT NOT NULL,
    approved_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    subject TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL
);
