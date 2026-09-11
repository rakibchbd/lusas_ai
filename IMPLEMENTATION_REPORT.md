# LUSAS AI model architecture and implementation report

## Result

The public model catalog now contains exactly:

| Public name | Stable ID | Artifact root |
| --- | --- | --- |
| Sara 1.0 | `sara-1.0` | `models/sara-1.0/` |
| Lira 1.0 | `lira-1.0` | `models/lira-1.0/` |

The former single-model artifact tree was removed. `models/` now contains only
`.gitkeep`; official model artifacts are created only after explicit foundation
configuration and approved training.

Removed legacy components include the former single-model weights and
production/candidate/backup layout, legacy model configuration and loading
logic, implicit fallback behavior, obsolete training dependencies and command
flags, old UI/documentation labels, and their regression-test fixtures. A
repository audit contains no legacy model identifiers or production artifact
paths.

Routine interactions do not require model weights. The local conversation
fast path answers greetings, identity/capability questions, and safe basic
arithmetic (for example, `what is 67+87`), so an uninstalled model cannot turn
a deterministic question into a misleading installation error.

## Architecture

`model_registry.py` is the single source of truth for public names, IDs,
model-specific stable/candidate/backup paths, and foundation configuration.
`LocalModel` accepts an official ID and loads only that model's stable adapter.
There is no cross-model or implicit foundation fallback.

The web console loads `/api/models`, displays the active model selector, stores
the selected ID locally and server-side, and includes `model_id` in every chat
request. `/admin.html` is protected by `LUSAS_ADMIN_TOKEN` and exposes model
status, approved sources, knowledge reviews, candidate comparison, training
jobs, deployment approval, rollback, and audit history.

The lifecycle is:

1. An administrator approves HTTPS sources and/or local training examples.
2. Collected records are cleaned, categorized, hashed, deduplicated, and
   scanned for prompt injection, shell payloads, credential theft, and scripts.
3. New web records remain pending and untrainable until approved and
   corroborated.
4. Training creates a model-specific candidate with explicit foundation
   metadata.
5. Evaluation requires accuracy, hallucination, safety, regression, English,
   Bengali, latency, and resource gates.
6. A passing candidate is staged. It cannot deploy until an administrator
   approves the exact candidate and evaluation report.
7. Promotion creates a model-specific backup. Stable artifacts can be restored
   only from the matching model's backup directory.

Automatic worker cycles can gather approved-source data, train, evaluate, and
stage candidates. They never promote candidates. Source-code evolution remains
separately bounded and tested.

When a non-routine question has no local answer, the agent can perform a
bounded fallback research pass over configured administrator-approved HTTPS
sources. It prefers cached approved evidence, refreshes within the configured
interval, preserves source URLs and verification status, and treats fetched
content as reference data rather than instructions. New material remains
pending until administrator review and corroboration.

## Created and modified files

Created:

- `lusas_ai/model_registry.py`
- `lusas_ai/data_pipeline.py`
- `lusas_ai/admin_db.py`
- `migrations/001_admin.sql`
- `web/admin.html`
- `web/admin.js`
- `tests/test_model_registry.py`
- `tests/test_data_pipeline.py`
- `.env.example`
- `IMPLEMENTATION_REPORT.md`

Updated:

- `config.json`, `README.md`, `training/README.md`, and training datasets
- `lusas_ai/config.py`, `agent.py`, `conversation.py`, `knowledge.py`,
  `local_model.py`, `monitor.py`, `web_learning.py`, and `web_ui.py`
- `training/train_lora.py`, `evaluate_model.py`, `run_model.py`,
  `model_lifecycle.py`, and `auto_upgrade.py`
- `web/index.html`, `web/app.js`, and `web/styles.css`
- the affected regression and web API tests

## Environment variables

- `LUSAS_SARA_FOUNDATION_MODEL` or `LUSAS_SARA_FOUNDATION_PATH`
- `LUSAS_LIRA_FOUNDATION_MODEL` or `LUSAS_LIRA_FOUNDATION_PATH`
- `LUSAS_ADMIN_TOKEN` for the local administration API
- `HF_TOKEN` only when the explicitly configured foundation needs authenticated
  model downloads

`.env.example` lists these variables without values. Real `.env` files are
ignored by Git.

Foundation settings are intentionally blank in `config.json`; no foundation
or model weights are bundled or selected silently.

## Database migration

`migrations/001_admin.sql` creates `schema_migrations`, `model_catalog`,
`approved_sources`, `data_reviews`, `training_jobs`, `deployment_approvals`,
and `audit_events`. The server applies pending migrations to
`.lusas/admin.sqlite3` and synchronizes the two official model records.

## Setup and operation

Install dependencies with:

~~~text
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r training/requirements.txt
export LUSAS_SARA_FOUNDATION_MODEL="approved/foundation-for-sara"
export LUSAS_LIRA_FOUNDATION_MODEL="approved/foundation-for-lira"
export LUSAS_ADMIN_TOKEN="local-admin-secret"
~~~

Train and evaluate a candidate:

~~~text
python3 training/train_lora.py --model-id sara-1.0 \
  --data training/data/examples.jsonl \
  --output models/sara-1.0/candidates/first \
  --foundation-model "$LUSAS_SARA_FOUNDATION_MODEL"
python3 training/evaluate_model.py --model-id sara-1.0 \
  --model-path models/sara-1.0/candidates/first \
  --data training/data/eval.jsonl
~~~

Run interactive mode with:

~~~text
python3 training/run_model.py --model-id sara-1.0 --interactive
~~~

Run the web UI with `python3 -m lusas_ai web`, then open
`http://127.0.0.1:8765/`. The administration page is at `/admin.html`.

## Verification

- Python unit/integration suite: **99 tests passed**.
- Python bytecode compilation: passed.
- `web/app.js` and `web/admin.js` syntax checks: passed.
- Config and training JSON/JSONL validation: passed.
- Localhost smoke test: model catalog returned both official IDs; chat included
  the selected ID; known founder questions returned grounded third-person
  answers; unknown people did not trigger scraping or code generation; both
  model selections answered `67+87` locally as `154` without installed weights.
- Visual browser check: selector, active-model label, Admin link, and redesigned
  dark console rendered correctly.

## Known limitations

- Sara 1.0 and Lira 1.0 are model-adapter tracks, not foundation models trained
  from zero. They are unavailable until explicit foundations and local model
  artifacts are provided.
- Real training and evaluation require the optional ML dependencies and enough
  CPU/GPU/MPS memory.
- Automatic learning stages candidates but requires an administrator for the
  final deployment approval, by design.
- Web learning and fallback research use only configured and approved HTTPS
  sources; they do not browse arbitrary sites or train directly from newly
  fetched content. A general search provider is intentionally not enabled
  without an administrator-approved endpoint and domain policy.
