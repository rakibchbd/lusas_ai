# LUSAS model training

This folder creates a LUSAS model adapter from an open-weight coding model.
The resulting adapter is a new model artifact that can be evaluated and
promoted independently of the agent runtime.

## Prepare

~~~text
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r training/requirements.txt
~~~

## Train a candidate

~~~text
python3 training/train_lora.py --data training/data/examples.jsonl \
  --output models/candidates/first
~~~

Run the trained candidate directly:

~~~text
python3 training/run_model.py --model models/candidates/first \
  --prompt "Write a Python function that reverses a string."
~~~

Talk with the trained model interactively. The model loads once, then accepts
multiple prompts until you type `/exit`:

~~~text
python3 training/run_model.py --model models/production --interactive
~~~

The default base model is a small coding model suitable for a first local
prototype. Change --base-model after confirming your machine has enough
memory for a larger model.

## Promote a candidate

Promotion creates a backup of the current production model before replacing
it. Only promote after an evaluation job has passed:

~~~text
python3 training/model_lifecycle.py promote \
  --candidate models/candidates/first --evaluation-passed
~~~

The backup is kept under models/backups/. Model weights are intentionally
excluded from Git by the root ignore file.

## Automatic upgrades

The project config enables automatic model and allowlisted web-learning cycles
every five minutes. Each cycle refreshes the configured HTTPS feeds, trains a
candidate when new data arrives, evaluates it, backs up the active model, and
promotes it only if the candidate passes and scores better than the current
production model:

~~~text
python3 training/auto_upgrade.py
~~~

Use --once to test one complete cycle.
