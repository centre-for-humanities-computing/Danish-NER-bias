The src folder contains the following: 

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```apply_fns``` | scripts used to import various language models|
| ```evaluate_fns``` | scripts used in evaluate pipeline in all evaluate_XX.py scripts |
| ```process_names``` | scripts used to preprocess names before name augmentation e.g., filtering overlaps between majority and minority name lists|
| ```evaluate_XX.py```  | Scripts for evaluating models, one for each framework (```evaluate_XX.py```)
| ```stats-names.py``` | Get count of all first and last names used in name augmentation after filtering. Divided into gender and majority/minority |