import pandas as pd 
from dacy.datasets import danish_names, female_names, male_names


# read in csv files
men_2023 = pd.read_csv("fornavne_2023_maend.csv")
women_2023 = pd.read_csv("fornavne_2023_kvinder.csv")

# create gender col 
men_2023["Gender"] = "Men"
women_2023["Gender"] = "Women"

# combine lists 
names_2023 = pd.concat([women_2023, men_2023])

# capitalize names to follow same structure as older list (str.title to make Åge-Hans and not Åge-hans with capitalize fn)
names_2023["Navn"] = names_2023["Navn"].str.title()

# read in older danish_names list
dk_name_dict = danish_names()
first_names_old = list(dk_name_dict["first_name"])
first_names_2023 = list(names_2023["Navn"])

# intersection
overlap = list(set(first_names_old).intersection(first_names_2023))
print(overlap)
print(len(overlap))
