from dacy.score import n_sents_score, score
from pathlib import Path
import spacy
import dacy
import os 
import pandas as pd

def eval_model_augmentation(model_dict, augmenters, dataset):
    '''
    Return CSV file of model performance on NER task with different name augmentations
    using DACY score function

    input:
    - model_dict : dictionary of models to be run
    - dataset : test dataset for model eval
    - augmenters 

    output: 
    - CSV file in folder "robustness" (creates directory if it does not exist)
    '''

    # define output path
    output_path = os.path.join("..", "robustness")

    Path("../robustness").mkdir(parents=True, exist_ok=True)
    
    # loop over all models in model_dict 
    for mdl in model_dict:
        print(f"[INFO]: Scoring model '{mdl}' using DaCy")

        # load model depending on model name (different pipelines)
        if "dacy" in mdl:
            apply_fn = dacy.load(model_dict[mdl])
        elif "spacy" in mdl:
            apply_fn = spacy.load(model_dict[mdl])
            spacy.prefer_gpu()
        else:
            apply_fn = model_dict[mdl]

        i = 0
        scores = []
        for aug, nam, k in augmenters:
            print(f"\t Running augmenter: {nam} | Amount of times: {k}")

            scores_ = score(corpus=dataset, apply_fn=apply_fn, augmenters=aug, k=k)
            scores_["model"] = mdl
            scores_["augmenter"] = nam
            scores_["i"] = i
            scores.append(scores_)

            i += 1

        scores = pd.concat(scores)

        scores.to_csv(f"{output_path}/{mdl}_augmentation_performance.csv")