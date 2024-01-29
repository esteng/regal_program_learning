#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
	    echo "Usage: $0 <num_gpus_to_monitor> <script_to_run> <free_gpu_threshold>"
	        exit 1
fi

num_gpus="$1"
script_to_run="$2"
free_gpu_threshold="$3"

# Function to check the number of free GPUs
function check_free_gpus {
	free_gpus=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ if ($1 >= '"$free_gpu_threshold"') print $1 }' | wc -l)
	echo "$free_gpus"
}

while true; do
	free_gpus_count=$(check_free_gpus)
	echo $free_gpus_count

	if [ "$free_gpus_count" -ge "$num_gpus" ]; then
		echo "Detected $free_gpus_count free GPUs. Running script..."
		bash "$script_to_run"
		break
	fi

	sleep 60  # Adjust the sleep duration based on your monitoring needs
done

