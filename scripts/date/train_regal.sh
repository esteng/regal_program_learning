#!/bin/bash
python program_refactoring/refactor_db.py \
	--collection_path python_data/date/gpt-3.5-turbo_NL+SL/my_vectordb/ \
	--filter_every 5 \
	--refactor_every 5 \
	--task python \
	--dataset date \
	--tree_type big_tree \
	--max_tuple_size 3 \
	--do_retry \
	--helpers_second  
