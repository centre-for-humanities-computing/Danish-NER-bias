#import ssl certificate to import spacy 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import spacy

# Dataset
from dacy.datasets import dane
testdata = dane(splits=["test"], redownload=True, open_unverified_connected=True)

### Define augmenters ###
from helper_fns.augmentation import dk_aug, muslim_aug, f_aug, m_aug, muslim_f_aug, muslim_m_aug, unisex_aug

n = 20
# augmenter, name, n repetitions 
augmenters = [
    (dk_aug, "Danish names", n),
    (muslim_aug, "Muslim names", n),
    (f_aug, "Female names", n),
    (m_aug, "Male names", n),
    (muslim_f_aug, "Muslim female names", n),
    (muslim_m_aug, "Muslim male names", n),
    (unisex_aug, "Unisex names", n),
]

### Define Models to Evaluate ###
model_dict = {
    "spacy_small": "da_core_news_sm",
    "spacy_medium": "da_core_news_md",
    "spacy_large": "da_core_news_lg",
}

### Evaluate models ###
from helper_fns.performance import eval_model_augmentation 

eval_model_augmentation(model_dict, augmenters, testdata)

