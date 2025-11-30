#!/usr/bin/env bash
keep_terminal_open() {
    echo "An unrecoverable exception occurred."
    exec "$SHELL"
}

echo "Are you using macOS with an M-series chip? (y/n) (e.g., M1, M2)"
read -r is_metal
if [ "$is_metal" = "y" ]; then
    echo "Setting up for Metal backend."
else
    echo "This application is unfortunately only optimized for macOS with M-series chips using the Metal backend. \n You may try to running it, but the dependencies may not be compatible."
    echo "Do you want to continue anyway? (y/n)"
    read -r continue_anyway
    if [ "$continue_anyway" != "y" ]; then
        keep_terminal_open
    fi
    echo "While you chose to continue, you can join us to help improve compatibility on other platforms!"
fi

echo "Do you have a virtual environment set up in ./venv? (y/n)"
read -r venv_setup
if [ "$venv_setup" = "n" ]; then
    echo "Please set up a virtual environment first (python -m venv venv && source venv/bin/activate)."
    keep_terminal_open
fi

source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

export PYTHONPATH="${PYTHONPATH:-.}:."

# the rest of code to run the application goes here
python -m src.main
if [[ -t 1 ]]; then
    echo "Flow exited. Keeping the terminal open (type exit to close)."
    exec "$SHELL"
fi