# LUSAS AI

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
- approved-example and allowlisted web learning with automatic retraining
- local LoRA fine-tuning pipeline for a LUSAS model
- model candidate promotion with checkpoint backups
- staged self-upgrades
- pre-apply backups
- tests, rollback-ready artifacts, native desktop notifications, and an
  inspectable upgrade history

The checked-in configuration enables automatic model evaluation, promotion, and
guarded source evolution. Every model cycle trains a candidate, evaluates it,
backs up the active model, and promotes it only when the evaluation passes.
Software self-upgrades remain bounded to the agent, test, and training source
directories and are subject to static security checks and the full test suite.

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

Hugging Face access is read from the `HF_TOKEN` environment variable when it
is available. The token is never stored in `config.json`, source files, logs,
or Git. Public models work without a token, while a read-only token provides
higher Hub rate limits and access to models your account is allowed to use:

~~~text
read -s HF_TOKEN
export HF_TOKEN
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

LUSAS answers from its local model, approved local training data, configured
local workspace, and optionally cached excerpts from explicitly allowlisted
HTTPS feeds. The web collector runs on a schedule, stores only bounded text
locally, and never executes downloaded content or sends prompts, credentials,
or workspace files to a website. Hugging Face is used only to download model
files during setup or training; inference itself runs locally.

The default web sources are official release feeds for CPython and
Hugging Face Transformers. Change `web_sources` and
`web_allowed_domains` together in `config.json` to use other HTTPS feeds.
Trigger a refresh manually with:

~~~text
python3 -m lusas_ai web-refresh --force
~~~

Upgrades remain local by design. The model, source changes, learned records,
version state, and logs use the local filesystem. Web learning only reads the
explicitly configured feeds; it does not publish to GitHub or use an external
answer API, and it never sends prompts, credentials, or workspace files.

The configured automatic model-upgrade interval is five minutes. Keep the
continuous worker running with `python3 training/auto_upgrade.py`; each cycle
refreshes the allowlisted web sources, incorporates new local records, and
still trains and evaluates a candidate before promotion.

Source evolution records a permanent JSONL history entry for every proposal,
including its unified diff, quality metrics, and deployment result. Diffs are retained under
`.lusas/upgrade-diffs/`, and progress events are retained in
`.lusas/evolution-progress.jsonl`. Native notifications use `osascript` on
macOS, `notify-send` on Linux, and a PowerShell Windows toast when available;
missing desktop services are reported in the notification log without
pretending delivery succeeded.

Inspect the local, HTTP-free upgrade dashboard at any time:

~~~text
python3 -m lusas_ai report
python3 -m lusas_ai report --json
~~~

Generate a browsable local HTML control dashboard from the same records:

~~~text
python3 -m lusas_ai dashboard
open .lusas/dashboard.html
~~~

The dashboard shows version, policy, web-learning, lesson, regression, and
upgrade-history summaries. It does not expose hidden model reasoning or grant
new permissions.

The `evolve` command prints each gate as it runs. The automatic local worker
is configured to run model and source-code upgrades every five minutes after
the macOS LaunchAgent is installed.

Run the guarded code-evolution pipeline directly with:

~~~text
python3 -m lusas_ai evolve --goal "Improve parser reliability" --apply
~~~

With both `evolution_enabled` and `auto_code_upgrades` enabled, each passing
model cycle also asks the current local LUSAS model to propose a small
source-code improvement. The
guarded evolution pipeline analyzes the repository, validates protected paths,
creates an isolated candidate workspace, runs standard-library AST security
checks and the full unit-test suite, records quality metrics, and compares
the candidate before deployment. Invalid, unsafe, or failing proposals are
rejected and logged without stopping the next cycle. Candidates are limited to
`lusas_ai/`, `tests/`, and `training/`, and rollback backups remain available
under `.lusas/backups/`.

The worker fingerprints its training and evaluation inputs. If nothing changed
since the previous successful cycle, it records a `skipped` event instead of
retraining the same data, making the five-minute schedule faster and reducing
unnecessary heat, memory use, and model churn.

Training now uses Apple MPS when available, larger batches on GPU/MPS, and
disabled pinned memory on MPS where it is unsupported. A rejected candidate is
also recorded as the current input result, so the same failed dataset is not
retrained every five minutes until new data or evaluation inputs arrive.

Every successful promotion increments the local LUSAS version by one patch step:
`v0.001`, followed by `v0.002`, `v0.003`, and so on. Rejected candidates do not
increment the version, but every attempt is recorded with its baseline score,
candidate score, improvement, and learned-example count.

View versions and upgrade history with:

~~~text
python3 -m lusas_ai monitor
~~~

To start the worker automatically when you log in on macOS, install the
LaunchAgent once:

~~~text
python3 -m lusas_ai service install
~~~

It starts at login, restarts after an exit, and uses `launchd`'s network-state
keep-alive. Because macOS protects Documents from some background processes,
the LaunchAgent writes logs to `~/Library/Logs/LUSAS_AI/`. The install command
verifies that the worker is actually running; if macOS denies access to the
Documents folder, it reports the failure instead of claiming success. Remove it
with:

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
