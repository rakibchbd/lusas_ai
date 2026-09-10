# LUSAS AI

Local-first coding agent with tested self-upgrades.

LUSAS AI is being built as its own local model project, with a small agent
runtime around it. The first practical model path is a LUSAS adapter trained
from an open-weight coding model. It keeps model upgrades versioned, tested,
and recoverable.

The assistant identifies itself as **Lusa**, created by **Rakib Chowdhury**.
That identity is enforced in the runtime and included in the training data so
the model does not fall back to its foundation model's identity.

Included:

- bounded workspace for project files
- local LUSAS model chat
- approved-example learning and automatic retraining
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

Install the local model dependencies and train a LUSAS candidate:

~~~text
python3 -m pip install --upgrade pip
python3 -m pip install -r training/requirements.txt
python3 training/train_lora.py \
  --data training/data/examples.jsonl \
  --output models/candidates/first
python3 -m lusas_ai status
~~~

Promote a candidate after evaluation, then run:

~~~text
python3 training/run_model.py --model models/production --interactive
python3 -m lusas_ai status
python3 -m lusas_ai chat
~~~

Generation behavior is configurable in `config.json`: `temperature`, `top_p`,
`top_k`, `repeat_penalty`, `num_ctx`, `num_predict`, and `seed` are available
for the local model runtime. The defaults favor focused, repeatable coding
answers.

Teach LUSAS explicitly with approved examples. They are stored locally and
included in the next automatic training cycle:

~~~text
python3 -m lusas_ai learn \
  --instruction "Who made you?" \
  --output "I am Lusa, created by Rakib Chowdhury."
~~~

Monitor exactly what LUSAS has learned and when upgrades happen:

~~~text
python3 -m lusas_ai monitor
python3 -m lusas_ai monitor --follow
~~~

The monitor reads `.lusas/learned.jsonl` and `.lusas/notifications.jsonl`,
shows learned instruction previews, and reports model-upgrade events. Stop live
monitoring with `Ctrl-C`.

The automatic model loop merges the built-in and learned examples, trains a
candidate, evaluates it against the regression set, and promotes it only when
all evaluation cases pass. It never replaces the production model with an
untested candidate.

Collect additional official documentation samples for review:

~~~text
python3 -m lusas_ai collect-web
~~~

The collector is limited to official Python, MDN, and PyTorch documentation,
checks `robots.txt`, applies a 512 KB response limit and a five-second delay,
and writes samples to `.lusas/web_pending.jsonl`. Web content is not added to
training automatically; review it before approving it with `lusas learn`.

When `auto_publish_upgrades` is enabled, a passing upgrade is also recorded in
the tracked learning history, committed to a new `lusas/upgrade-*` branch,
pushed to `origin`, and submitted as a pull request with `gh`. It will not
publish if the working tree already contains local changes. Model weight files
remain local because they are excluded by `.gitignore`; the pull request
contains the learned data and upgrade history, not large model binaries.

The configured automatic model-upgrade interval is five minutes. Keep the
continuous worker running with `python3 training/auto_upgrade.py`; each cycle
still trains and evaluates a candidate before promotion.

To start the worker automatically when you log in on macOS, install the
LaunchAgent once:

~~~text
python3 -m lusas_ai service install
~~~

It starts at login, restarts after an exit, and uses `launchd`'s network-state
keep-alive. Logs are written to `.lusas/auto-upgrade.log` and
`.lusas/auto-upgrade-error.log`. Remove it with:

~~~text
python3 -m lusas_ai service remove
~~~

To talk directly with the trained LUSAS model, first prepare and promote a
candidate, then start the interactive model runner:

~~~text
python3 training/run_model.py --model models/production --interactive
~~~

LUSAS identifies itself as developed by Lusa Chowdhury (Rakib). The current
local model is based on an open-weight coding model and fine-tuned locally.

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
