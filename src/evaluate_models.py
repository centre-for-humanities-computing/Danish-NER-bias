'''
Script to evaluate all models (except polyglot) in a NER task on the DaNE test dataset when running several data augmentations on first & last names [PER] (e.g., majority vs minority names)

Run script in terminal by typing: 
    python src/evaluate_models.py -m chosen_model -e chosen_eval_function

For the optional arguments, you can write the following: 
    -m: 'spacy', 'dacy', 'nerda', 'scandi_ner', 'flair', 'danlp' 
    -e: 'dacy' or 'fairness'

Where -m is to choose between model frameworks and -e is to choose between evaluation function
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
from evaluate_fns.performance import eval_model_augmentation 
from evaluate_fns.fairness_metrics import eval_fairness_metrics

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-m", "--model", help = "model framework you want to evalute", type = str, default = "spacy") 
    parser.add_argument("-e", "--eval_function", help = "'dacy' to use dacy.score (Lassen et al., 2023) or 'fairness' to use custom scoring function", default = "fairness")
    
    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def load_model(chosen_model): 
    if chosen_model == "spacy":
        model_dict = {
        "spacy_small": "da_core_news_sm",
        "spacy_medium": "da_core_news_md",
        "spacy_large": "da_core_news_lg",
        }

    elif chosen_model == "dacy":
        model_dict = {
        "dacy_small": "da_dacy_small_trf-0.1.0",
        "dacy_medium": "da_dacy_medium_trf-0.1.0",
        "dacy_large": "da_dacy_large_trf-0.1.0"
        }

    elif chosen_model == "danlp":
        # import danlp 
        from apply_fns.apply_fn_danlp import apply_danlp_bert
        model_dict = {"danlp_bert": apply_danlp_bert}
    
    elif chosen_model == "nerda":
        from apply_fns.apply_fn_nerda import apply_nerda
        model_dict = {"nerda_bert": apply_nerda}
    
    elif chosen_model == "scandi_ner":
        from apply_fns.apply_fn_scandi import scandi_ner
        model_dict = {"scandi_ner": scandi_ner}   
    
    elif chosen_model == "flair":
        from apply_fns.apply_fn_flair import apply_flair
        model_dict = {"flair": apply_flair}  

    return model_dict 

def main():
    # define args
    args = input_parse()

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

    # define models to run 
    model_dict = load_model(args.model)

    # evaluate 
    if args.eval_function == "dacy":
        eval_model_augmentation(model_dict, augmenters, testdata)
    elif args.eval_function == "fairness":
        # run for only PER entity
        eval_fairness_metrics(model_dict=model_dict, augmenters=augmenters, dataset=testdata, ents_to_keep=["PER"], outfolder=outfolder_PER, filename="PER")

        # run for all ents excl. MISC 
        eval_fairness_metrics(model_dict=model_dict, augmenters=augmenters, dataset=testdata, ents_to_keep=["PER", "LOC", "ORG"], outfolder=outfolder_ALL, filename="ALL_EXCL_MISC")

# run script 
if __name__ == "__main__":
    main()