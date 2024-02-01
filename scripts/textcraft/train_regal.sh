#!/bin/bash
python program_refactoring/refactor_db.py \
	--collection_path python_data/textcraft/gpt-4/my_vectordb/ \
	--filter_every 5 \
	--refactor_every 5 \
	--task textcraft \
	--dataset textcraft \
	--tree_type big_tree \
	--max_tuple_size 3 \
	--do_retry \
	--helpers_second  \
	--craft_retrieve True 
