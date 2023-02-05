import pandas as pd
import spacy

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
    "dacy_small": "da_dacy_small_trf-0.1.0",
    "dacy_medium": "da_dacy_medium_trf-0.1.0",
    "dacy_large": "da_dacy_large_trf-0.1.0",
}

### Performance ###
from helper_fns.performance import eval_model_augmentation 

eval_model_augmentation(model_dict, augmenters, testdata)