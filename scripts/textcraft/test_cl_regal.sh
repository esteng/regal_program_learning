#!/bin/bash
SIZE=$1
model_name=codellama/CodeLlama-${SIZE}b-Instruct-hf 
safe_model_name=$(echo ${model_name} |  sed 's/\//_/g') 
echo ${safe_model_name}
LOGDIR=$2
RUNDIR_KEY=$3
SEED=$4
python program_refactoring/agent/agent_main.py \
	--train_json_path python_data/textcraft/gpt-4/train.jsonl \
	--train_log_path ${LOGDIR} \
	--logdir \
	--test_path python_data/textcraft/gpt-4/test.jsonl \
	--task textcraft \
	--dataset textcraft \
    --model_name ${model_name}\
	--out_dir cl_results/regal_cl${SIZE}b_${RUNDIR_KEY}/seed_${SEED} \
	--seed ${SEED} \
	--use_thought \
	--use_success \
	--test_retry True \
	--craft_retrieve True \
	--max_budget ${MAX_BUDGET} \
	--budget_split ${BUDGET_SPLIT} 
