#!/bin/bash

# install all necessary reqs to run all models except polyglot 

# create and activate env 
python3.9 -m venv env
source ./env/bin/activate

# install requirements
echo -e "[INFO:] Installing necessary requirements in virtual environment..." # user msg

pip install -r requirements/requirements.txt

echo -e "[INFO:] Setup complete!" # user msg 

# deactivate environment
deactivate