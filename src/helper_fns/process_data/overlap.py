import pandas as pd 

from duplicates_clean import dk_name_dict, f_name_dict, m_name_dict

# read in csv files
men_2023 = pd.read_csv("fornavne_2023_maend.csv")
women_2023 = pd.read_csv("fornavne_2023_kvinder.csv")

# capitalize names to follow same structure as older list (str.title to make Åge-Hans and not Åge-hans with capitalize fn)
men_2023["Navn"] = men_2023["Navn"].str.title()
women_2023["Navn"] = women_2023["Navn"].str.title()

# first names from old and new list 
men_old_names = list(m_name_dict["first_name"])
men_2023_names = list(men_2023["Navn"])

# find overlap
men_overlap = list(set(men_old_names).intersection(men_2023_names))
print(men_overlap)
print(f"Overlap between men lists: {len(men_overlap)}")
print(f"Len of OG men names list: {len(men_old_names)}")
