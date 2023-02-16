import pandas as pd 
from dacy.datasets import female_names, male_names

#import
f_name_dict = female_names()
m_name_dict = male_names()

# define path 
from pathlib import Path
path = Path(__file__) # path to current file
path.parents[1] # two directories up
data_folder = path.parents[1] / "names_csv_files" 

# read in csv files
men_2023 = pd.read_csv(data_folder / "first_names_2023_men.csv")
women_2023 = pd.read_csv(data_folder / "first_names_2023_women.csv")

# capitalize names to follow same structure as older list (str.title to make Åge-Hans and not Åge-hans with capitalize fn)
men_2023["Navn"] = men_2023["Navn"].str.title()
women_2023["Navn"] = women_2023["Navn"].str.title()

# first names from old and new list 
men_old_names = m_name_dict["first_name"]
all_men_2023_names = list(men_2023["Navn"])

women_old_names = f_name_dict["first_name"]
all_women_2023_names = list(women_2023["Navn"])

from overlap_fns import find_overlap

# subset men 2023 and women 2023 
subset_men_2023_names = all_men_2023_names[:500]
subset_women_2023_names = all_women_2023_names[:500]

#calculate overlap
men_all_overlap = find_overlap(men_old_names, all_men_2023_names)
women_all_overlap = find_overlap(women_old_names, all_women_2023_names)

men_subset_overlap = find_overlap(men_old_names, subset_men_2023_names)
women_subset_overlap = find_overlap(women_old_names, subset_women_2023_names)

## all overlaps 
overlaps = {
    "Overlap Old Dacy & New Lists": ["all 2023 names", "2023 subset popular names"],
    "Men overlaps" : [men_all_overlap[0], men_subset_overlap[0]],
    "Women overlaps": [women_all_overlap[0], women_subset_overlap[0]],
}

lengths = {
    "Length of Lists": ["all m/w 2023 names", "2023 subset popular", "old names"],
    "Women (Len)": [len(all_women_2023_names), len(subset_women_2023_names), len(women_old_names)],
    "Men (Len)": [len(all_men_2023_names), len(subset_men_2023_names), len(men_old_names)]
}

print("\n")

print(pd.DataFrame(overlaps))

print("\n")

print(pd.DataFrame(lengths))

print("\n")

from dacy.datasets import load_names
muslim_m_dict = load_names(ethnicity="muslim", gender="male", min_prop_gender=0.5)
muslim_w_dict = load_names(ethnicity="muslim", gender="female", min_prop_gender=0.5)

muslim_m_first = muslim_m_dict["first_name"]
muslim_w_first = muslim_w_dict["first_name"]

overlap_women_muslim = find_overlap(subset_women_2023_names, muslim_w_first)
overlap_men_muslim = find_overlap(subset_men_2023_names, muslim_m_first)

overlaps_gender_muslim = {
    "Men overlaps" : overlap_men_muslim[0],
    "Women overlaps": overlap_women_muslim[0]
}

overlaps_file = pd.DataFrame({"navn":overlap_women_muslim[1] + overlap_men_muslim[1], "origin":""})

print(overlaps_gender_muslim)
print(len(overlaps_file))