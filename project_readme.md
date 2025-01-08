Location:

- The Barnes-hut optimization is located in the file: main_approach_2.cu on branch: approach_2

Parameters:

- Number of bodies and number of threads can be changed in the define part at the start of the file
- Other parameters relevant to the execution are defined as global constants and can be changed as well

Initialization values:

- To initialize values on CPU, uncomment the line: 1015
- To initialize values on GPU, uncomment the line: 1016
- To load values from a file, uncomment the lines: 1019,1020

Execution:

- To compile, create the executable and run it: 
    nvcc main_approach_2.cu -o main && ./main

- To visualize bodies in the quadtree for init or final simulation step:
    python plot_quadtree.py quadtree_init_gpu.txt

- To run multiple executions with different number of threads:
    bash measure_times.sh

- To plot visualizations of Speedup and Efficiency for the multiple executions run:
    python plot_times.py results_nb_40_000_ns_10_new.txt
    *** IMPORTANT: 
        - to visualize only optimized parallel computations uncomment the line: 48 of plot_times.py
        - to visualize total computations uncomment the line: 49 of plot_times.py


*** The bash and python scripts were created using help of AI tools, to save time, as this is not the focus of this project