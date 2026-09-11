CREATE TABLE IF NOT EXISTS interaction_events (
    interaction_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    knowledge_used INTEGER NOT NULL DEFAULT 0,
    web_search_used INTEGER NOT NULL DEFAULT 0,
    sources TEXT NOT NULL DEFAULT '[]',
    confidence REAL,
    answer_quality TEXT NOT NULL DEFAULT 'unrated',
    feedback TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS practice_runs (
    run_id TEXT PRIMARY KEY,
    knowledge_id TEXT NOT NULL,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    evidence TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_events (
    research_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    sources_selected TEXT NOT NULL DEFAULT '[]',
    source_quality TEXT NOT NULL DEFAULT '{}',
    elapsed_ms REAL,
    information_found INTEGER NOT NULL DEFAULT 0,
    verification_success INTEGER NOT NULL DEFAULT 0,
    usefulness TEXT NOT NULL DEFAULT 'unrated',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_tasks (
    task_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0.5,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    run_after TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interaction_status ON interaction_events(status, created_at);
CREATE INDEX IF NOT EXISTS idx_practice_knowledge ON practice_runs(knowledge_id, created_at);
CREATE INDEX IF NOT EXISTS idx_evolution_task_status ON evolution_tasks(status, priority DESC, created_at);
