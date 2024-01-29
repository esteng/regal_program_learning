#!/bin/bash 

SEED=$1
python source/predict/predict_hf.py \
	--dataset_name date \
	--split mytest \
	--model_name codellama-7b-NL+SL \
	--seed ${SEED}
