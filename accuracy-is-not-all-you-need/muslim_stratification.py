from pathlib import Path

import pandas as pd
import spacy
import spacy_stanza
import stanza
from spacy.training.augment import dont_augment

import augmenty
import dacy

from dacy.datasets import dane, danish_names, female_names, male_names, muslim_names, load_names
from dacy.score import n_sents_score, score

import apply_fns
#from apply_fns.apply_fn_danlp import apply_danlp_bert
#from apply_fns.apply_fn_nerda import apply_nerda

#from apply_fns.apply_fn_flair import apply_flair
#from apply_fns.apply_fn_polyglot import apply_polyglot

# Dataset
test = dane(splits=["test"])

# Augmenters - Create a list of augmenters we wish to apply to our model.

# randomly augment names
dk_name_dict = danish_names()
muslim_name_dict = muslim_names()
f_name_dict = female_names()
m_name_dict = male_names()

# define muslim male and female names (not helper functions yet)
muslim_m_dict = load_names(ethnicity="muslim", gender="male", min_prop_gender=0.5)
muslim_f_dict = load_names(ethnicity="muslim", gender="female", min_prop_gender=0.5)

patterns = [["first_name"], ["first_name", "last_name"],
            ["first_name", "last_name", "last_name"]]

person_tag = "PER" # define person tag for augmenters (person_tag = PERSON for spacy models)

dk_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = dk_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

f_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = f_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

m_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = m_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = muslim_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_f_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = muslim_f_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_m_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = muslim_m_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

n = 1 #should be 20
# augmenter   name               n rep
augmenters = [
    #(dont_augment, "No augmentation", 1),
    (dk_aug, "Danish names", n),
    (muslim_aug, "Muslim names", n),
    (f_aug, "Female names", n),
    (m_aug, "Male names", n),
    (muslim_f_aug, "Muslim female names", n),
    (muslim_m_aug, "Muslim male names", n)
]

# Apply functions and models
# Loading application functions for necessary models. No need to create one for SpaCy pipelines.

model_dict = {
    #"stanza": "da",
    #"spacy_small": "da_core_news_sm",
    #"spacy_medium": "da_core_news_md",
    #"spacy_large": "da_core_news_lg",
    #"dacy_small": "da_dacy_small_trf-0.1.0",
    #"dacy_medium": "da_dacy_medium_trf-0.1.0",
    "dacy_large": "da_dacy_large_trf-0.1.0",
    #"flair": apply_flair,
    #"polyglot": apply_polyglot,
    #"danlp_bert": apply_danlp_bert,
    #"nerda_bert": apply_nerda,
}

# # Performance

Path("robustness").mkdir(parents=True, exist_ok=True)

for mdl in model_dict:
    print(f"[INFO]: Scoring model '{mdl}' using DaCy")

    # load model
    if "dacy" in mdl:
        apply_fn = dacy.load(model_dict[mdl])
    elif "spacy" in mdl:
        apply_fn = spacy.load(model_dict[mdl])
        spacy.prefer_gpu()
    elif "stanza" in mdl:
        stanza.download(model_dict[mdl])
        # Initialize the pipeline
        apply_fn = spacy_stanza.load_pipeline(model_dict[mdl])
    else:
        apply_fn = model_dict[mdl]

    i = 0
    scores = []
    for aug, nam, k in augmenters:
        print(f"\t Running augmenter: {nam}")

        scores_ = score(corpus=test, apply_fn=apply_fn, augmenters=aug, k=k)
        scores_["model"] = mdl
        scores_["augmenter"] = nam
        scores_["i"] = i
        scores.append(scores_)

        i += 1

    for n in [5, 10]:
        scores_ = n_sents_score(n_sents=n, apply_fn=apply_fn)
        scores_["model"] = mdl
        scores_["augmenter"] = f"Input size augmentation {n} sentences"
        scores_["i"] = i + 1
        scores.append(scores_)

    scores = pd.concat(scores)

    scores.to_csv(f"robustness/{mdl}_augmentation_performance2.csv")