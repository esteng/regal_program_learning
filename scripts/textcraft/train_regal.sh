#!/bin/bash
 
python program_refactoring/refactor_db.py \
	--collection_path python_data/textcraft/gpt-4/my_vectordb_combined/ \
	--filter_every 5 \
	--refactor_every 5 \
	--task textcraft \
	--dataset textcraft \
	--tree_type big_tree \
	--max_tuple_size 4 \
	--craft_retrieve True\
	