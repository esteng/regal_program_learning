
#!/bin/bash

SEED=$1
model_name=codellama/CodeLlama-13b-Instruct-hf 
safe_model_name=$(echo ${model_name} |  sed 's/\//_/g') 
echo ${safe_model_name} 
python program_refactoring/agent/agent_main.py \
	--train_json_path logo_data/python/train_200_better_embed.jsonl \
	--train_log_path none \
	--test_path logo_data/python/test.jsonl \
	--model_name ${model_name} \
	--out_dir test_runs/logos_llama_13b_baseline_${MAX_BUDGET}_${SEED}_seed \
	--max_budget ${MAX_BUDGET}\
	--seed ${SEED} 
