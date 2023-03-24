'''
Script to give an overview of the amount of first and last names in each group (All, Women, Men in Majoirty, Minority)

To run this script and get this overview, type: 
    - python src/stats-names.py 

'''

import pandas as pd

# add custom modules 
import sys
from pathlib import Path
module_path = Path(__file__).parents[0] / "process_names"
sys.path.append(str(module_path))

from process_names.names import (
                                dk_name_dict, f_name_dict, m_name_dict, 
                                muslim_name_dict, muslim_f_dict, muslim_m_dict, 
                                unisex_name_dict
                                )


first_name_lengths = {
    "First Names": ["Majority", "Minority"],
    "All": [len(dk_name_dict["first_name"]), len(muslim_name_dict["first_name"])],
    "Women": [len(f_name_dict["first_name"]), len(muslim_f_dict["first_name"])],
    "Men": [len(m_name_dict["first_name"]), len(muslim_m_dict["first_name"])]
}

last_name_lengths = {
    "Last Names": ["Majority", "Minority"],
    "All": [len(dk_name_dict["last_name"]), len(muslim_name_dict["last_name"])],
    "Women": [len(f_name_dict["last_name"]), len(muslim_f_dict["last_name"])],
    "Men": [len(m_name_dict["last_name"]), len(muslim_m_dict["last_name"])]
}

print(pd.DataFrame(first_name_lengths))
print(pd.DataFrame(last_name_lengths))

