#!/bin/bash 

seed=$1
python source/predict/predict_hf.py \
	--dataset_name date \
	--split mytest \
	--model_name codellama-34b-NL+SL\
	--seed ${seed} 
