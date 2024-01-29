#!/bin/bash 

SEED=$1
python source/predict/predict.py \
	--dataset_name date \
	--split mytest \
	--model_name gpt-3.5-turbo_NL+SL \
	--seed ${SEED}
