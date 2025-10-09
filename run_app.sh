echo "Do you have a virtual environment set up? (y/n)"
read venv_setup
if [ "$venv_setup" != "n" ]; then
    echo "Please set up a virtual environment first."
    exit 1
fi

source venv/bin/activate
pip install -r thermal_flow_cnf/requirements.txt
export PYTHONPATH=.
streamlit run thermal_flow_cnf/app_streamlit.py