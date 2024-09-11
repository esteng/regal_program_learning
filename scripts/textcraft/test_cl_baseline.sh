#!/bin/bash
SIZE=$1
SEED=$2
model_name=codellama/CodeLlama-${SIZE}b-Instruct-hf 
safe_model_name=$(echo ${model_name} |  sed 's/\//_/g') 
echo ${safe_model_name}
python program_refactoring/agent/agent_main.py \
	--train_json_path python_data/textcraft/gpt-4/train.jsonl \
	--train_log_path none \
	--test_path python_data/textcraft/gpt-4/dev.jsonl \
	--task textcraft \
	--dataset textcraft \
	--model_name ${model_name} \
	--out_dir cl_results/baseline_cl${SIZE}b/seed_${SEED} \
	--seed ${SEED} \
	--test_retry True \
	--use_success \
	--max_budget ${MAX_BUDGET} \
	--budget_split ${BUDGET_SPLIT} \
	--use_thought \