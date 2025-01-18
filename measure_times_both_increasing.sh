#!/bin/bash

# Define unique sets of values
bodies=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 40000)
threads=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 40000)
simulations=(10)

# Number of repetitions per configuration
repetitions=5

# Output file for results
output="newest_ns_10_bs_32.txt"
echo "n_bodies, n_threads, n_simulations, repetition, runtime" > $output

# Iterate over unique bodies and threads values
for i in "${!bodies[@]}"; do
    n_b=${bodies[$i]}
    n_t=${threads[$i]}
    n_s=${simulations[0]}

    # Run the configuration multiple times
    for ((rep=1; rep<=repetitions; rep++)); do
        # Display current iteration details
        echo -e "\n\n=============================================="
        echo "Running simulation with:"
        echo "  n_bodies=$n_b"
        echo "  n_threads=$n_t"
        echo "  n_simulations=$n_s"
        echo "  repetition=$rep"
        echo "==============================================\n"

        # Compile with specific n_bodies, n_threads, and n_simulations
        nvcc -DN_BODIES=$n_b -DN_THREADS=$n_t -DN_SIMULATIONS=$n_s -o main main_approach_2.cu

        # Run the program and capture the runtime
        runtime=$(./main)

        # Save the result
        echo "$n_b, $n_t, $n_s, $rep, $runtime" >> $output

        # Print runtime for this configuration with extra spacing
        echo -e "\n----------------------------------------------"
        echo "Completed: n_bodies=$n_b, n_threads=$n_t, n_simulations=$n_s, repetition=$rep"
        echo "  Runtime: $runtime ms"
        echo "----------------------------------------------\n"
    done

done