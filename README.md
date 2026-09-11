# LUSAS AI

LUSAS AI is a local-first model and agent platform. Its two official public
models are `Sara 1.0` (`sara-1.0`) and `Lira 1.0` (`lira-1.0`). The runtime
never silently substitutes a different model. Each official model can use an
explicitly configured foundation model or local foundation path, while its
public identity remains the official LUSAS name.

The project includes a local chat agent, persistent model selection, a gated
knowledge pipeline, candidate training and evaluation, versioned deployment,
backups, rollback, monitoring, audit logs, and a localhost administration
dashboard.

## Quick start

Create the environment and install the training/runtime dependencies:

~~~text
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r training/requirements.txt
~~~

Configure each model explicitly. Use a model identifier available to your
Transformers installation or a local foundation directory:

~~~text
export LUSAS_SARA_FOUNDATION_MODEL="your-approved-sara-foundation"
export LUSAS_LIRA_FOUNDATION_MODEL="your-approved-lira-foundation"
# Alternatively use LUSAS_SARA_FOUNDATION_PATH and/or LUSAS_LIRA_FOUNDATION_PATH.
export LUSAS_ADMIN_TOKEN="choose-a-local-admin-secret"
~~~

Do not place tokens in `config.json`, source files, logs, or Git. A read-only
`HF_TOKEN` may be used by the Transformers download path when the configured
foundation requires it.

## Train and run Sara or Lira

Training accepts only clean, approved records. The foundation is required on
every training command; there is no implicit default:

~~~text
python3 training/train_lora.py \
  --model-id sara-1.0 \
  --data training/data/examples.jsonl \
  --output models/sara-1.0/candidates/first \
  --foundation-model "$LUSAS_SARA_FOUNDATION_MODEL"
~~~

Evaluate a candidate. The evaluation includes accuracy, hallucination,
safety, regression, English, Bengali, latency, and resource gates:

~~~text
python3 training/evaluate_model.py \
  --model-id sara-1.0 \
  --model-path models/sara-1.0/candidates/first \
  --data training/data/eval.jsonl
~~~

The model selector in the web console persists the selected ID in
`.lusas/selected_model.json`. The same selection can be used from the CLI:

~~~text
python3 training/run_model.py --model-id sara-1.0 --interactive
python3 -m lusas_ai chat --model-id lira-1.0
~~~

The interactive runner loads one selected model and accepts prompts until
`/exit`. A missing model or missing foundation produces a clear error; it does
not fall back to another model.

## Approval-gated deployment

Every candidate must pass all eight evaluation gates. An administrator must
then approve the exact candidate in the administration page or by calling the
deployment API. Promotion creates a model-specific backup before replacing
the stable artifact. Automatic cycles may train and evaluate candidates, but
they cannot deploy them.

~~~text
python3 training/model_lifecycle.py promote \
  --model-id sara-1.0 \
  --candidate models/sara-1.0/candidates/first \
  --evaluation models/sara-1.0/candidates/first/evaluation.json \
  --approved-by administrator \
  --admin-approved
~~~

Backups are stored below `models/<model-id>/backups/`. Roll back through the
administration dashboard or:

~~~text
python3 training/model_lifecycle.py rollback \
  --model-id sara-1.0 \
  --backup models/sara-1.0/backups/<backup-directory>
~~~

## Local web console and administration

Start the localhost console:

~~~text
python3 -m lusas_ai web
open http://127.0.0.1:8765
~~~

The chat request always contains the selected `model_id`. Open `/admin.html`
and enter `LUSAS_ADMIN_TOKEN` to manage the two models, approve HTTPS sources,
review and approve or reject collected records, start training jobs, approve
deployments, roll back stable versions, and inspect jobs and audit events.

## Knowledge and web learning

Web learning is restricted to HTTPS sources in the configured administrator
allowlist. Every fetched record is normalized, categorized, deduplicated, and
scanned for prompt injection, shell payloads, credential theft, and script
content. New records are quarantined as pending. They are never sent directly
to training. Only a corroborated record explicitly approved by an
administrator can enter the web-training queue.

Configure `web_sources` and `web_allowed_domains` together in `config.json`,
then approve the sources from `/admin.html` or refresh manually:

~~~text
python3 -m lusas_ai web-refresh --force
~~~

Local examples added with `python3 -m lusas_ai learn` are also cleaned and
checked before a later training cycle. Knowledge is stored locally under
`.lusas/` and the administration database is `.lusas/admin.sqlite3`; migration
`migrations/001_admin.sql` creates the model, source, review, training,
deployment-approval, and audit tables.

## Automatic improvement and controls

Run one cycle or keep the local worker running:

~~~text
python3 training/auto_upgrade.py --once
python3 training/auto_upgrade.py
~~~

Each real cycle audits the repository, records served interactions, detects
changed learning inputs, runs retention practice, measures process resources,
refreshes approved web sources, and then trains/evaluates model candidates when
their explicit foundations and approved data are available. The worker keeps
durable state under `.lusas/` and resumes its next cycle after a restart.

The cycle may gather approved web references and create a candidate for the
configured default model. It records evaluation results and notifications,
but leaves deployment in `staged` state until administrator approval. Source
evolution is independently bounded to the project source/test/training paths,
with backups and tests. Use runtime controls to pause learning, web research,
upgrades, or self-code evolution:

~~~text
python3 -m lusas_ai control status
python3 -m lusas_ai control stop
python3 -m lusas_ai control resume
~~~

Inspect the local report with `python3 -m lusas_ai report --json` or monitor
events with `python3 -m lusas_ai monitor`. Install the macOS LaunchAgent only
after reviewing the configuration:

~~~text
python3 -m lusas_ai service install
~~~

For a normal question that has no local knowledge, LUSAS performs a bounded
research fallback over the configured administrator-approved HTTPS sources. It
prefers verified cached evidence, records provenance, and keeps newly fetched
records pending until review. Deterministic arithmetic such as `67+87` is
answered locally and never requires Sara or Lira weights.

## Project limits

This repository supplies orchestration and model-adapter training, not a
foundation model's weights. Sara 1.0 and Lira 1.0 therefore remain unavailable
until their explicit foundation configuration, dependencies, approved data,
and local artifacts are present. Training a capable model from zero requires a
large lawful corpus and substantial accelerator resources. Automatic promotion
is intentionally disabled until a human administrator approves a fully gated
candidate.
