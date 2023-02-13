import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from transformers import pipeline
import pandas as pd

scandi_ner = pipeline(task='ner', 
               model='saattrupdan/nbailab-base-ner-scandi', 
               aggregation_strategy='first', 
               device = 1, #gpu?
               framework = "pt"
               )