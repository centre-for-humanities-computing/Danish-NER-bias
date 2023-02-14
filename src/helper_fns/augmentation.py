import augmenty 

#import cleaned data
from process_data.duplicates_clean import (
                                dk_name_dict, f_name_dict, m_name_dict, 
                                muslim_name_dict, muslim_f_dict, muslim_m_dict, 
                                unisex_name_dict
                                )

## AUGMENTATION ## 

# define pattern of augmentation
patterns = [["first_name"], ["first_name", "last_name"],
            ["first_name", "last_name", "last_name"]]

# define person tag for augmenters 
person_tag = "PER" 

# define all augmenters 
dk_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = dk_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

f_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = f_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

m_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = m_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = muslim_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_f_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = muslim_f_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_m_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = muslim_m_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

unisex_aug = augmenty.load(
    "per_replace_v1", 
    patterns = patterns, 
    names = unisex_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )