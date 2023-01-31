import augmenty 
from dacy.datasets import dane, danish_names, female_names, male_names, muslim_names, load_names

# load danish names 
dk_name_dict = danish_names()
muslim_name_dict = muslim_names()
f_name_dict = female_names()
m_name_dict = male_names()

muslim_m_dict = load_names(ethnicity="muslim", gender="male", min_prop_gender=0.5)
muslim_f_dict = load_names(ethnicity="muslim", gender="female", min_prop_gender=0.5)

# remove duplicates 


## AUGMENTATION ## 

# define pattern of augmentation
patterns = [["first_name"], ["first_name", "last_name"],
            ["first_name", "last_name", "last_name"]]

# define person tag for augmenters 
person_tag = "PER" 

# define all augmenters 
dk_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = dk_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

f_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = f_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

m_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = m_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = muslim_name_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_f_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = muslim_f_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )

muslim_m_aug = augmenty.load(
    "per_replace.v1", 
    patterns = patterns, 
    names = muslim_m_dict, 
    level = 1, 
    person_tag = person_tag, 
    replace_consistency = True
    )