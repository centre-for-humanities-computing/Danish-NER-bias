'''
Script to evaluate Polyglot in a NER task on the DaNE test dataset when running several data augmentations on first & last names [PER] (e.g., majority vs minority names)

Run script in terminal by typing: 
    python src/evaluate_polyglot.py  -e chosen_eval_function

For the optional arguments you can write the following to choose between evaluation function:
    -e: 'dacy' or 'fairness'
'''

# import polyglot
import apply_fns
from apply_fns.apply_fn_polyglot import apply_polyglot

# add custom modules 
import sys
from pathlib import Path
module_path = Path(__file__).parents[0] / "evaluate_fns"
sys.path.append(module_path)

# import augmenters, performance function
from evaluate_fns.augmentation import dk_aug, muslim_aug, f_aug, m_aug, muslim_f_aug, muslim_m_aug, unisex_aug
from evaluate_fns.performance import eval_model_augmentation 

# import dataset
from dacy.datasets import dane

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-e", "--eval_function", help = "'dacy' to use dacy.score (Lassen et al., 2023) or 'fairness' to use custom scoring function", default = "fairness")
    
    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main():
    # define args
    args = input_parse()

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
    model_dict = {
        "polyglot": apply_polyglot,
    }

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