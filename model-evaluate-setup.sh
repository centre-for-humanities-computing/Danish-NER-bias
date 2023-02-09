

# make env

# virtualenvwrapper
#mkvirtual flair_env

# conda
# conda create ...

# activate
# virtualenvwrapper
#workon flair_env

# conda
# conda active flair_env

# setup req.
pip install -r requirements_flair.txt

python evaluate_flair.py

# deactivate env
deactivate

# new env
mkvirtual others
# ... do the same thing as above

## ross' suggestions 
python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python src/evaluate_dacy.py
deactivate
rm -rf env

