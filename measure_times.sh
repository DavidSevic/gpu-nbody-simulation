#!/bin/bash

# Define ranges for parameters
# Problem sizes
bodies=(40000)
# Thread counts
threads=(1 1 1 2 2 2 4 4 4 8 8 8 16 16 16 32 32 32 64 64 64 128 128 128 256 256 256 512 512 512 1024 1024 1024 1024*2 1024*2 1024*2 1024*4 1024*4 1024*4 1024*8 1024*8 1024*8 1024*16 1024*16 1024*16 1024*32 1024*32 1024*32 40000 40000 40000)
# Number of simulations per configuration
simulations=(10)

# Output file for results
output="results_nb_40_000_ns_10.txt"
echo "n_bodies, n_threads, n_simulations, runtime" > $output

for n_b in "${bodies[@]}"; do
    for n_t in "${threads[@]}"; do
        for n_s in "${simulations[@]}"; do
            # Display current iteration details with extra spacing
            echo -e "\n\n=============================================="
            echo "Running simulation with:"
            echo "  n_bodies=$n_b"
            echo "  n_threads=$n_t"
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
