#!/bin/bash

# Define ranges for parameters
# Problem sizes
bodies=(40000)
# Thread counts (unique values, not repeated)
threads=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 40000)
# Number of simulations per configuration
simulations=(10)
# Number of repetitions for each thread count
repeats=5

# Output file for results
output="results_nb_40_000_ns_10_dynamic_tree_size.txt"
echo "n_bodies, n_threads, n_simulations, runtime" > $output

for n_b in "${bodies[@]}"; do
    for n_t in "${threads[@]}"; do
        for ((i=1; i<=repeats; i++)); do
            for n_s in "${simulations[@]}"; do
                # Display current iteration details with extra spacing
                echo -e "\n\n=============================================="
                echo "Running simulation with:"
                echo "  n_bodies=$n_b"
                echo "  n_threads=$n_t (repeat $i/$repeats)"
                echo "  n_simulations=$n_s"
                echo "==============================================\n"

                # Compile with specific n_bodies, n_threads, and n_simulations
                nvcc -DN_BODIES=$n_b -DN_THREADS=$n_t -DN_SIMULATIONS=$n_s -o main main_approach_2.cu

                # Run the program and capture the runtime
                runtime=$(./main)

                # Save the result
                echo "$n_b, $n_t, $n_s, $runtime" >> $output

                # Print runtime for this configuration with extra spacing
                echo -e "\n----------------------------------------------"
                echo "Completed: n_bodies=$n_b, n_threads=$n_t, n_simulations=$n_s"
                echo "  Runtime: $runtime ms"
                echo "----------------------------------------------\n"
            done
        done
    done
done