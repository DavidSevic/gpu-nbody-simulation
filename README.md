# Barnes-Hut N-Body Simulation on GPU

This project implements a GPU-accelerated N-body simulation using the Barnes-Hut algorithm. It includes both CPU and CUDA-based versions, with support for performance scaling experiments and visualizations. The simulation models gravitational forces between particles, efficiently approximating distant interactions using a quadtree structure.

Location:

- The Barnes-hut optimization is located in the file: project.cu

Parameters:

- Number of bodies and number of threads can be changed in the define part at the start of the file
- Other parameters relevant to the execution are defined as global constants and can be changed as well

Initialization values:

- To initialize values on CPU and store them into files, uncomment the line: 1061
- To initialize values on GPU, uncomment the line: 1062
- To load values from a file, uncomment the lines: 1065, 1066

Execution:

- To compile, create the executable and run it: 
    nvcc project.cu -o project && ./project

- To visualize bodies in the quadtree for init or final simulation step:
    python plot_quadtree.py quadtree_init_gpu.txt

- To run multiple executions for the first scaling approach:
    bash first_scaling_script.sh

- To run multiple executions for the second scaling approach:
    bash second_scaling_script.sh

- To plot visualizations of Speedup and Efficiency for the first scaling multiple executions run:
    python plot_first_scale.py first_scaling_results.txt

- To plot visualizations of Runtime for the second scaling multiple executions run:
    python plot_second_scale.py second_scaling_results.txt


*** The bash and python scripts were created using help of AI tools, to save time, as this is not the focus of this project
*** Scripts for multiple configuration on plots from the report are not here, as they require custom intput format
