# Detecting intersectionality in NER models: A data-driven approach

This repository contains the code used to produce the results in the paper "**Detecting intersectionality in NER models: A data-driven approach**" by Lassen et al. (2023). 

The project investigates the effect of intersectional biases in several Danish models used for Named Entity Recognition (NER).


## Project Structure 
The repository has the following directory structure:
| <div style="width:120px">property</div>| Description |
|---------|:-----------|
| ```name_lists``` | Contains name lists used for data augmentation|
| ```requirements``` | Requirements file for all models, and a seperate file for polyglot |
| ```results``` | Results from all model evaluations saved as CSV files|
| ```src```  | Contains scripts for evaluating all models (```evaluate_XX.py```). Also has helper modules for preprocessing name lists (```process_names```), importing models (```apply_fns```), and augmenting names + evaluating models (```evaluate_fns```).|
| ```utils-R``` | Utils for running ```results.md```|
| ```results.md``` | Rmarkdown for producing results table from the paper (```Table 2```) |
| ```polyglot.sh``` | Installs necessary tools and packages and runs the evaluation of ```polyglot``` |
| ```setup.sh``` | Installs necessary packages for running evaluation of all models except ```polyglot```|
| ```run-models.sh``` | Runs the evaluation of all models except ```polyglot```|



The repo evaluates the following models: 
* ScandiNER
* DaCY large (da_dacy_large_trf-0.1.0)
* DaCy medium (da_dacy_medium_trf-0.1.0)
* DaCY small (da_dacy_small_trf-0.1.0)
* DaNLP BERT
* Flair
* NERDA
* SpaCy large (da_core_news_lg-3.4.0)
* SpaCy medium (da_core_news_md-3.4.0)
* SpaCy small (da_core_news_sm-3.4.0)
* Polyglot


## Pipeline 
The pipeline has been tested on Ubuntu ([UCloud](https://cloud.sdu.dk/)). It cannot be confirmed that it will work on Windows without modifications. 

For all models except ```polyglot```, first run the ```setup.sh```
```
bash setup.sh
```
This will create a virtual environment (```env```) and install the necessary packages. 

To then evaluate all models, run: 
```
bash run-models.sh
```

### Polyglot
To setup and evaluate ```polyglot```, run: 
```
sudo bash polyglot.sh
```
**NB! Notice that it is necessary to run this code with sudo as the setup requires certain devtools that will not be installed otherwise. Run at own risk!**

The ```polyglot.sh``` script will both install devtools, packages and run the evaluation of the model in a seperately created environment called ```polyenv```. 

## Acknowledgements
This project builds upon code originally developed for [DaCy](https://github.com/centre-for-humanities-computing/DaCy) and utilizes the package [augmenty](https://kennethenevoldsen.github.io/augmenty/) for data augmentation. 

