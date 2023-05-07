#!/bin/bash

# activate virtual environment 
echo -e "[INFO:] Creating environment ..." # user msg
python3.9 -m venv polyenvfair
source ./polyenv/bin/activate

# install dev tools
echo -e "[INFO:] Installing DEV tools ..." # user msg
apt-get update && apt-get install -y apt-transport-https -y
apt-get install libicu-dev -y
apt-get install python3-dev -y

#install things in this order
echo -e "[INFO:] Installing packages ..." # user msg
pip install pycld2
pip install polyglot
pip install --no-binary=:pyicu: pyicu

# all other packages
pip install -r requirements/requirements-polyglot.txt

# for fairness metrics
pip install git+https://github.com/centre-for-humanities-computing/spacy_polyglot

echo -e "[INFO:] Setup complete ..." # user msg

# run polyglot model 
echo -e "[INFO:] Evaluating polyglot ..." # user msg

python3.9 src/fairness_polyglot.py

echo -e "[INFO:] Evaluation done! Results saved ..." # user msg

# deactivate virtual env 
deactivate
