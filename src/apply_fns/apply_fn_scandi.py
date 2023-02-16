# setup certificate to load model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# load ScandiNER (https://huggingface.co/saattrupdan/nbailab-base-ner-scandi) using dacy>2.31 (https://centre-for-humanities-computing.github.io/DaCy/using_dacy.getting_started.html#named-entity-recognition)
import spacy, dacy
scandi_ner = spacy.blank("da")
scandi_ner.add_pipe("dacy/ner")