#!/usr/bin/env bash
set -euo pipefail

echo "Do you have a virtual environment set up in ./venv? (y/n)"
read -r venv_setup
if [ "$venv_setup" = "n" ]; then
    echo "Please set up a virtual environment first (python -m venv venv && source venv/bin/activate)."
    exit 1
fi

source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH:-.}:."
exec streamlit run thermal_flow_cnf/app_streamlit.py