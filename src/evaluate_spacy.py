import pandas as pd
import spacy

#import apply_fns
#from apply_fns.apply_fn_danlp import apply_danlp_bert
#from apply_fns.apply_fn_nerda import apply_nerda

#from apply_fns.apply_fn_flair import apply_flair
#from apply_fns.apply_fn_polyglot import apply_polyglot

# Dataset
from dacy.datasets import dane
testdata = dane(splits=["test"])

### Define augmenters ###
from helper_fns.augmentation import dk_aug, muslim_aug, f_aug, m_aug, muslim_f_aug, muslim_m_aug

n = 20
# augmenter, name, n repetitions 
augmenters = [
    (dk_aug, "Danish names", n),
    (muslim_aug, "Muslim names", n),
    (f_aug, "Female names", n),
    (m_aug, "Male names", n),
    (muslim_f_aug, "Muslim female names", n),
    (muslim_m_aug, "Muslim male names", n)
]

### Define Models to Run ###
model_dict = {
    "spacy_small": "da_core_news_sm",
    "spacy_medium": "da_core_news_md",
    "spacy_large": "da_core_news_lg",
}

### Performance ###
from helper_fns.performance import eval_model_augmentation 

eval_model_augmentation(model_dict, augmenters, testdata)



