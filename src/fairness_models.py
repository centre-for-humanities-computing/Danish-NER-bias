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

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-m", "--model", help = "model framework you want to evalute", type = str, default = "spacy") 
    
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
        import spacy_wrap
        nlp = spacy.blank("da")
        config = {"model": {"name": "alexandrainst/da-ner-base"}}
        nlp.add_pipe("token_classification_transformer", config=config)
        model_dict = {"danlp":nlp}
    
    elif chosen_model == "scandi_ner":
        from apply_fns.apply_fn_scandi import scandi_ner
        model_dict = {"scandi_ner": scandi_ner}   
    
    elif chosen_model == "flair":
        from danlp_spacy_flair import FlairComponent
        nlp = spacy.blank("da")
        nlp.add_pipe("danlp_flair", last=True)
        model_dict = {"flair": nlp}

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

    # run for only PER entity
    eval_fairness_metrics(model_dict=model_dict, augmenters=augmenters, dataset=testdata, ents_to_keep=["PER"], outfolder=outfolder_PER, filename="PER")

    # run for all ents excl. MISC 
    eval_fairness_metrics(model_dict=model_dict, augmenters=augmenters, dataset=testdata, ents_to_keep=["PER", "LOC", "ORG"], outfolder=outfolder_ALL, filename="ALL_EXCL_MISC")

# run script 
if __name__ == "__main__":
    main()