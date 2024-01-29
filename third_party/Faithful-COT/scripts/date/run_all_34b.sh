#!/bin/bash

for seed in 12 42 64
do
	./scripts/date/codellama_34b_baseline.sh ${seed} 
done
