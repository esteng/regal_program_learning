#!/bin/bash
python program_refactoring/refactor_db.py \
	--collection_path logo_data/my_vectordb/ \
	--filter_every 5 \
	--refactor_every 5 \
	--task logos \
	--dataset logos \
	--tree_type big_tree \
	--do_retry \
	--add_comments \
	--helpers_second 
	# --existing_log_dir $1
