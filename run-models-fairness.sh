#!/bin/bash

# evaluate all models except 

# activate virtual environment (NB. will only work if you run 'bash setup.sh' first!!)
source ./env/bin/activate

# evaluating models 
echo -e "[INFO:] Evaluating SpaCy models ..." # user msg
python3.9 src/fairness_models.py -m spacy

echo -e "[INFO:] Evaluating DaCy models ..." # user msg
python3.9 src/fairness_models.py -m dacy

echo -e "[INFO:] Evaluating Scandi-NER ..." # user msg
python3.9 src/fairness_models.py -m scandi_ner

echo -e "[INFO:] Evaluating flair ..." # user msg
python3.9 src/fairness_models.py -m flair

echo -e "[INFO:] Evaluating DaNLP BERT ..." # user msg
python3.9 src/fairness_models.py -m danlp 

# happy message ! 
echo -e "[INFO:] Evaluation of all models done! Results saved ..." # user msg

# deactivate virtual env 
deactivate
