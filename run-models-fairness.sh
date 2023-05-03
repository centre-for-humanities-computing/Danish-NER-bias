#!/bin/bash

# evaluate all models except 

# activate virtual environment (NB. will only work if you run 'bash setup.sh' first!!)
source ./env/bin/activate

# evaluating models 
echo -e "[INFO:] Evaluating SpaCY models ..." # user msg
python3.9 src/evaluate_models.py -m spacy -e fairness

echo -e "[INFO:] Evaluating DaCy models ..." # user msg
python3.9 src/evaluate_models.py -m dacy -e fairness

echo -e "[INFO:] Evaluating Scandi-NER ..." # user msg
python3.9 src/evaluate_models.py -m scandi_ner -e fairness

echo -e "[INFO:] Evaluating flair ..." # user msg
python3.9 src/evaluate_models.py -m flair -e fairness

echo -e "[INFO:] Evaluating NERDA BERT ..." # user msg
python3.9 src/evaluate_models.py -m nerda -e fairness

echo -e "[INFO:] Evaluating DaNLP BERT ..." # user msg
python3.9 src/evaluate_models.py -m danlp -e fairness

# happy message ! 
echo -e "[INFO:] Evaluation of all models done! Results saved ..." # user msg

# deactivate virtual env 
deactivate