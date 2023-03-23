'''
Script for preprocessing the name lists before data augmentation and running evaluate_model.py scripts
'''

# system tools 
from pathlib import Path

# data set wrangling
import pandas as pd
from dacy.datasets import muslim_names, load_names

### DEFINE NAMES ### 
# import names
path = Path(__file__) # path to current file
data_folder = path.parents[2] / "name_lists" 

## last names ##
last_names_2023 = pd.read_csv(data_folder / "last_names_2023.csv")
last_names_2023["Navn"] = last_names_2023["Navn"].str.title() # capitalize
last_names_2023 = list(last_names_2023["Navn"])[:500] # subset to only 500 to match 500 first names

## men and women first names ##
men_2023 = pd.read_csv(data_folder / "first_names_2023_men.csv")
women_2023 = pd.read_csv(data_folder / "first_names_2023_women.csv")

# capitalize
men_2023["Navn"] = men_2023["Navn"].str.title()
women_2023["Navn"] = women_2023["Navn"].str.title()

# subset names to 500 
men_2023 = list(men_2023["Navn"])[:500]
women_2023 = list(women_2023["Navn"])[:500]

## unisex ##
unisex_first_names = pd.read_csv(data_folder / 'unisex_names.csv')
unisex_first_names["Navn"] = unisex_first_names["Navn"].str.title() #capitalize
unisex_first_names = list(unisex_first_names['Navn'])[:500]

# create dictionaries 
all_danish_2023_first_names = men_2023 + women_2023

dk_name_dict = {'first_name':all_danish_2023_first_names, 'last_name':last_names_2023}
m_name_dict = {'first_name':men_2023, 'last_name':last_names_2023}
f_name_dict = {'first_name':women_2023, 'last_name':last_names_2023}
unisex_name_dict = {'first_name':unisex_first_names, 'last_name':last_names_2023}

## muslim names ##
muslim_name_dict = muslim_names()
muslim_m_dict = load_names(ethnicity="muslim", gender="male", min_prop_gender=0.5)
muslim_f_dict = load_names(ethnicity="muslim", gender="female", min_prop_gender=0.5)

### REMOVE OVERLAPS ##
import pandas as pd
from process_names.overlap_fns import remove_duplicates 

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