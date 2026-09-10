#!/bin/sh
cd "/Users/rakib/Documents/LUSAS_AI" || exit 1
exec "/Users/rakib/Documents/LUSAS_AI/.venv/bin/python" \
  "/Users/rakib/Documents/LUSAS_AI/training/auto_upgrade.py" --once
