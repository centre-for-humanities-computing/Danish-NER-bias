import augmenty 
from dacy.datasets import danish_names, female_names, male_names, muslim_names, load_names

### LOAD all names ##
dk_name_dict = danish_names()
muslim_name_dict = muslim_names()
f_name_dict = female_names()
m_name_dict = male_names()

muslim_m_dict = load_names(ethnicity="muslim", gender="male", min_prop_gender=0.5)
muslim_f_dict = load_names(ethnicity="muslim", gender="female", min_prop_gender=0.5)

### REMOVE duplicates ###
muslim_only = ['Samir', 'Jamal', 'Omar', 'Abbas', 'Ali', 'Ahmad', 'Ismail', 'Ibrahim', 'Abdul', 'Muhammad', 'Youssef', 'Amir', 'Bibi', 'Mariam', 'Adnan', 'Mohammad', 'Yasmin', 'Yusuf', 'Fatima', 'Said', 'Mohamed', 'Muhammed', 'Hussain', 'Hassan', 'Sami', 'Karim', 'Malik', 'Mohamad', 'Abdullah', 'Amina', 'Leila', 'Hasan', 'Mehmet', 'Mohammed', 'Bilal', 'Miriam', 'Mahmoud', 'Hussein', 'Osman', 'Mustafa', 'Fatma', 'Khaled', 'Abdullahi', 'Hamid', 'Mina', 'Abdi', 'Zahra', 'Ahmed', 'Saleh', 'Ahmet']
danish_only = ['Sofia', 'May', 'Elias', 'Lina', 'Jul', 'Nadia', 'Adam', 'Susan', 'Naja', 'Jan', 'Mona', 'Nadja', 'Hanna', 'Maria']

def remove_duplicates(all_names, names_to_filter_away):
    all_names = [name for name in all_names if name not in names_to_filter_away]
    return all_names

#overall
dk_name_dict["first_name"] = remove_duplicates(dk_name_dict["first_name"], muslim_only)
muslim_name_dict["first_name"] = remove_duplicates(muslim_name_dict["first_name"], danish_only)

# for danish genders 
f_name_dict["first_name"] = remove_duplicates(f_name_dict["first_name"], muslim_only)
m_name_dict["first_name"] = remove_duplicates(m_name_dict["first_name"], muslim_only)

# for muslim genders
muslim_f_dict["first_name"] = remove_duplicates(muslim_f_dict["first_name"], danish_only)
muslim_m_dict["first_name"] = remove_duplicates(muslim_m_dict["first_name"], danish_only)

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