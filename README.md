# LUSAS AI

Local-first coding agent with tested self-upgrades.

LUSAS AI is being built as its own local model project, with a small agent
runtime around it. The first practical model path is a LUSAS adapter trained
from an open-weight coding model. It keeps model upgrades versioned, tested,
and recoverable.

Included:

- bounded workspace for project files
- local Ollama chat
- local LoRA fine-tuning pipeline for a LUSAS model
- model candidate promotion with checkpoint backups
- staged self-upgrades
- pre-apply backups
- tests, rollback-ready artifacts, and notifications

The checked-in configuration enables automatic software and model upgrades.
Every model cycle trains a candidate, evaluates it, backs up the active model,
and promotes it only when the evaluation passes. Software self-upgrades remain
bounded to the agent, test, and training source directories.

Training a foundation model from zero requires a large curated dataset and
substantial accelerator compute. This project starts with a local fine-tuned
model so it can become yours on ordinary hardware, then leaves room for
continued pretraining later.

## Quick start

Install Ollama, start it, and pull a coding model:

~~~text
ollama pull qwen2.5-coder:7b
~~~

Then run:

~~~text
python3 -m lusas_ai status
python3 -m lusas_ai chat
~~~

To talk directly with the trained LUSAS model, first prepare and promote a
candidate, then start the interactive model runner:

~~~text
python3 training/run_model.py --model models/production --interactive
~~~

Prepare a self-upgrade:

~~~text
python3 -m lusas_ai self-upgrade --goal "Improve the CLI help text"
~~~

Apply a tested staged upgrade with the --apply flag.
Backups and staged candidates are stored under .lusas/.

Run one complete automatic model-upgrade cycle immediately with:

~~~text
python3 training/auto_upgrade.py --once
~~~

To restore one, pass its backup directory to:

~~~text
python3 -m lusas_ai rollback --backup .lusas/backups/<backup>
~~~
