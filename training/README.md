# Sara 1.0 and Lira 1.0 training

The training tools create a local adapter for one of the two official LUSAS AI
models. Use the stable internal IDs `sara-1.0` and `lira-1.0` on every command.
The foundation is an explicit dependency, never an implicit fallback.

## Install

~~~text
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r training/requirements.txt
~~~

## Train a candidate

~~~text
python3 training/train_lora.py \
  --model-id sara-1.0 \
  --data training/data/examples.jsonl \
  --output models/sara-1.0/candidates/first \
  --foundation-model "$LUSAS_SARA_FOUNDATION_MODEL"
~~~

Use `--foundation-path /path/to/a/local/foundation` instead when the approved
foundation is on disk. For Lira, use `--model-id lira-1.0` and its explicit
`LUSAS_LIRA_FOUNDATION_MODEL` or `LUSAS_LIRA_FOUNDATION_PATH` setting.

The input loader rejects unapproved, unverified, duplicate, or malicious
records. Web records must be administrator-approved and corroborated before
they become trainable.

## Evaluate

~~~text
python3 training/evaluate_model.py \
  --model-id sara-1.0 \
  --model-path models/sara-1.0/candidates/first \
  --data training/data/eval.jsonl
~~~

The result must report true for every required gate: accuracy, hallucination,
safety, regression, English, Bengali, latency, and resources. A passing score
alone is insufficient.

## Run interactively

~~~text
python3 training/run_model.py --model-id sara-1.0 --interactive
~~~

Or use the saved selector choice by omitting `--model-id`. Type `/exit` to
quit. If the selected model is not installed or its explicit foundation is
not configured, the command fails clearly and does not substitute another
model.

## Promote and roll back

Promotion requires a complete evaluation JSON and an explicit administrator
approval. The command creates a backup under the selected model's directory:

~~~text
python3 training/model_lifecycle.py promote \
  --model-id sara-1.0 \
  --candidate models/sara-1.0/candidates/first \
  --evaluation models/sara-1.0/candidates/first/evaluation.json \
  --approved-by administrator \
  --admin-approved
~~~

Restore a model-specific backup with:

~~~text
python3 training/model_lifecycle.py rollback \
  --model-id sara-1.0 \
  --backup models/sara-1.0/backups/<backup-directory>
~~~

The automatic worker can train and evaluate candidates, but deployment always
stops at `staged` until the administration dashboard or the approval API
records human approval.
