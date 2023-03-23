#!/bin/bash

# evaluate all models except 

# activate virtual environment (NB. will only work if you run 'bash setup.sh' first!!)
source ./env/bin/activate

# evaluating models 
echo -e "[INFO:] Evaluating SpaCY models ..." # user msg
python3.9 src/evaluate_spacy.py

echo -e "[INFO:] Evaluating DaCy models ..." # user msg
python3.9 src/evaluate_dacy.py

echo -e "[INFO:] Evaluating Scandi-NER ..." # user msg
python3.9 src/evaluate_scandi.py

echo -e "[INFO:] Evaluating flair ..." # user msg
python3.9 src/evaluate_flair.py

echo -e "[INFO:] Evaluating NERDA BERT ..." # user msg
python3.9 src/evaluate_nerda.py

echo -e "[INFO:] Evaluating DaNLP BERT ..." # user msg
python3.9 src/evaluate_danlp.py

# happy message ! 
echo -e "[INFO:] Evaluation of all models done! Results saved ..." # user msg