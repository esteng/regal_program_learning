#!/bin/bash

for seed in 12 42 64
do
	./scripts/date/codellama_13b_baseline.sh ${seed} 
done
