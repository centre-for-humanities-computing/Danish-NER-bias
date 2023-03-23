'''
Script to evaluate DaNLP BERT in a NER task on the DANE test dataset when running several data augmentations on first & last names [PER] (e.g., majority vs minority names)

'''

# import danlp
from apply_fns.apply_fn_danlp import apply_danlp_bert

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

def main():
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
    "danlp_bert": apply_danlp_bert,
    }

    # evaluate 
    eval_model_augmentation(model_dict, augmenters, testdata)

# run script 
if __name__ == "__main__":
    main()