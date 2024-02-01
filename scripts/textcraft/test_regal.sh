#!/bin/bash

#model_name=codellama/CodeLlama-7b-Python-hf
model_name=codellama/CodeLlama-13b-Instruct-hf 
safe_model_name=$(echo ${model_name} |  sed 's/\//_/g') 

LOGDIR=$1
RUNDIR=$2
SEED=$3
echo ${safe_model_name} 
python program_refactoring/agent/agent_main.py \
	--train_json_path python_data/textcraft/gpt-4/train.jsonl \
	--train_log_path $LOGDIR \
	--test_path python_data/textcraft/gpt-4/test_d2.jsonl \
	--model_name ${model_name} \
	--out_dir ${RUNDIR}_${SEED}_seed \
	--logdir \
	--seed ${SEED} \
	--task textcraft \
	--dataset textcraft \
	--use_success \
	--use_thought \
	--max_budget ${MAX_BUDGET} \
	--budget_split ${BUDGET_SPLIT} \
	--filter \
	--test_retry True \
	--craft_retrieve True \

