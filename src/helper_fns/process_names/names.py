#pandas pd 
import pandas as pd
from dacy.datasets import danish_names, female_names, male_names, muslim_names, load_names

# define path 
from pathlib import Path
path = Path(__file__) # path to current file
path.parents[1] # two directories up
data_folder = path.parents[1] / "names_csv_files" 

### DEFINE NAMES ### 

## old dacy names ## 
old_dk_name_dict = danish_names()
old_f_name_dict = female_names()
old_m_name_dict = male_names()

## men and women first names ##
men_2023 = pd.read_csv(data_folder / "fornavne_2023_maend.csv")
women_2023 = pd.read_csv(data_folder / "fornavne_2023_kvinder.csv")

#capitalize to follow old names 
men_2023["Navn"] = men_2023["Navn"].str.title()
women_2023["Navn"] = women_2023["Navn"].str.title()

# subset names to 500 
men_2023 = list(men_2023["Navn"])[:500]
women_2023 = list(women_2023["Navn"])[:500]

# create dictionaries 
all_danish_2023_first_names = men_2023 + women_2023

dk_name_dict = {'first_name':all_danish_2023_first_names, 'last_name':old_dk_name_dict['last_name'][:1000]}
m_name_dict = {'first_name':men_2023, 'last_name':old_dk_name_dict['last_name'][:1000]}
f_name_dict = {'first_name':women_2023, 'last_name':old_dk_name_dict['last_name'][:1000]}

## unisex ##
unisex_first_names = pd.read_csv(data_folder / 'unisex-navne.csv', usecols=['Navn'])

unisex_first_names = list(unisex_first_names['Navn'])

unisex_name_dict = {'first_name':unisex_first_names, 'last_name':dk_name_dict['last_name']}

## muslim names ##
muslim_name_dict = muslim_names()
muslim_m_dict = load_names(ethnicity="muslim", gender="male", min_prop_gender=0.5)
muslim_f_dict = load_names(ethnicity="muslim", gender="female", min_prop_gender=0.5)

### REMOVE OVERLAPS ##
import pandas as pd
from helper_fns.process_names.overlap_fns import remove_duplicates 

# read in annotated
overlaps = pd.read_csv(data_folder / "overlapping_names.csv")

# create muslim/minority only list and danish only list 
muslim_only=list(overlaps.query("origin=='M'")["duplicates"])
danish_only=list(overlaps.query("origin=='DK'")["duplicates"])

# overall 
dk_name_dict["first_name"] = remove_duplicates(dk_name_dict["first_name"], muslim_only)
muslim_name_dict["first_name"] = remove_duplicates(muslim_name_dict["first_name"], danish_only)

# danish genders
f_name_dict["first_name"] = remove_duplicates(f_name_dict["first_name"], muslim_only)
m_name_dict["first_name"] = remove_duplicates(m_name_dict["first_name"], muslim_only)

# muslim genders
muslim_f_dict["first_name"] = remove_duplicates(muslim_f_dict["first_name"], danish_only)
muslim_m_dict["first_name"] = remove_duplicates(muslim_m_dict["first_name"], danish_only)


first_name_lengths = {
    "Lists": ["Majority", "Minority"],
    "All": [len(dk_name_dict["first_name"]), len(muslim_name_dict["first_name"])],
    "Women": [len(f_name_dict["first_name"]), len(muslim_f_dict["first_name"])],
    "Men": [len(m_name_dict["first_name"]), len(muslim_m_dict["first_name"])]
}

