'''
Script to get fairness metrics (TP/TN/FP/PRECISION/RECALL/F1_SCORE) on all models (except polyglot) in a NER task 
on the DaNE test dataset when running several data augmentations on first & last names [PER] (e.g., majority vs minority names)

Run script in terminal by typing: 
    python src/evaluate_models.py -m chosen_model

For the optional arguments, you can write the following to chose between model frameworks:
    -m: 'spacy', 'dacy', 'scandi_ner', 'flair', 'danlp' 

Where -m is to choose between model frameworks
'''

# import ssl certificate to download models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# import packages
import spacy
import dacy
from dacy.datasets import dane

# utils
import argparse
import pathlib 

# add custom modules 
import sys
module_path = pathlib.Path(__file__).parents[0] / "evaluate_fns"
sys.path.append(module_path)

# import augmenters, performance function
from evaluate_fns.augmentation import dk_aug, muslim_aug, f_aug, m_aug, muslim_f_aug, muslim_m_aug, unisex_aug
from evaluate_fns.fairness_metrics import eval_fairness_metrics

# import polyglot wrapper
from spacy_polyglot import PolyglotComponent 

def main():
    # paths
    path = pathlib.Path(__file__)

    # fairness metrics paths 
    outfolder_PER = path.parents[1] / "results_DSH" / "PER"
    outfolder_ALL = path.parents[1] / "results_DSH" / "ALL_EXCL_MISC"

    # import data set 
    testdata = dane(splits=["test"], redownload=True, open_unverified_connected=True)

    # define augmenters: augmenter, name, n repetitions 
    n = 20
    augmenters = [
        (dk_aug, "Danish names", n),
        (muslim_aug, "Muslim names", n),
        (f_aug, "Female names", n),
        (m_aug, "Male names", n),
        (muslim_f_aug, "Muslim female names", n),
        (muslim_m_aug, "Muslim male names", n),
        (unisex_aug, "Unisex names", n),
    ]

    # initialize nlp object, add polyglot
    nlp = spacy.blank("da")
    nlp.add_pipe("polyglot", last=True)
    
    # define model dict
    model_dict = {"polyglot":nlp}

    # run for only PER entity
    eval_fairness_metrics(model_dict=model_dict, augmenters=augmenters, dataset=testdata, ents_to_keep=["PER"], outfolder=outfolder_PER, filename="PER")

    # run for all ents excl. MISC 
    eval_fairness_metrics(model_dict=model_dict, augmenters=augmenters, dataset=testdata, ents_to_keep=["PER", "LOC", "ORG"], outfolder=outfolder_ALL, filename="ALL_EXCL_MISC")

# run script 
if __name__ == "__main__":
    main()