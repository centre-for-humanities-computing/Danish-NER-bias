import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from transformers import pipeline
import pandas as pd

#scandi_ner = pipeline(task='ner', model='saattrupdan/nbailab-base-ner-scandi',  aggregation_strategy='first', device = 1, framework = "pt")

# load ScandiNER (https://huggingface.co/saattrupdan/nbailab-base-ner-scandi) using Dacy (https://centre-for-humanities-computing.github.io/DaCy/using_dacy.getting_started.html#named-entity-recognition)
import spacy, dacy

scandi_ner = spacy.blank("da")
scandi_ner.add_pipe("dacy/ner")
