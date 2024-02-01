# REGAL: Refactoring Programs to Discover Generalizable Abstractions 


![Figure of the motivation for REGAL](assets/fig1_single.png)

## Overview
This repo contains code for our paper: REGAL: Refactoring Programs to Discover Generalizable Abstractions
REGAL is a method for learning libraries of interpretable helper functions for program synthesis in a diverse range of tasks. 
Please see the figure below (and our paper) for more details. 

![Figure of the REGAL method](assets/fig2_method.png)

## Dependencies 
Dependencies can be installed by running 

```
pip install -r requirements.txt 
```

## Scripts 
Scripts for preprocessing ReGAL can be found in `scripts/<domain>/preprocess.sh`.
These require an `OPENAI_API_KEY` environment variable to be set. 

Scripts for training ReGAL on LOGO, Date, and TextCraft can be found in 
`scripts/<domain>/train_regal.sh` 
After training, the resulting logs and saved CodeBank files will be in `logs`.


Scripts for evaluation on CodeLlama-13B are in `scripts/<domain>/test_regal.sh`

Note that for scripts to run, the main dir must be on the Python path.
