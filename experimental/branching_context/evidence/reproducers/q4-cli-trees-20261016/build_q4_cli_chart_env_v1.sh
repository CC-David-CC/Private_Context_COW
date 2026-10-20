#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1
[[ ! -e $ROOT/chart-env-v1 ]]
/usr/bin/python3 -m venv "$ROOT/chart-env-v1"
"$ROOT/chart-env-v1/bin/python" -m pip install matplotlib==3.10.7
"$ROOT/chart-env-v1/bin/python" -m pip freeze > "$ROOT/chart-requirements-v1.txt"
