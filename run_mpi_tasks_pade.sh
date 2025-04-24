#!/bin/bash

# Define directories
directories=( "settings_epsilonEM_pade_wrong" "settings_epsilonEMGW_pade_wrong" "settings_epsilon0_pade_wrong")

# Log file for monitoring
log_file="run_mpi_tasks_pade.log"
failed_log_file="failed_cases_padelog"

echo "Starting tasks at $(date)" > "$log_file"
echo "Failed cases log - $(date)" > "$failed_log_file"

# Iterate over each directory
for dir in "${directories[@]}"; do
    echo "Processing files in directory: $dir" >> "$log_file"
    
    # Iterate through all the files in the current directory
    for file in "$dir"/*.yaml; do
        echo "Starting task for file: $file" >> "$log_file"
        
        #nohup nice -n 20 taskset -c 32-42 mpirun -n 4 python3 runner.py "$file" > "output_breaking_${file##*/}.log" 2>&1 &
        python3 test.py "$file" > "output_${file##*/}.log" 2>&1 &
        bg_pid=$!  # Get the PID of the background process
        echo "Task started for file: $file with PID: $bg_pid" >> "$log_file"

        # Wait for the background process to complete
        wait $bg_pid
        exit_status=$?

        if [ $exit_status -ne 0 ]; then
            echo "Task failed for file: $file with error code: $exit_status" >> "$failed_log_file"
        fi

        echo "Task completed for file: $file with PID: $bg_pid" >> "$log_file"
    done
done

echo "All tasks from both directories completed at $(date)" >> "$log_file"

