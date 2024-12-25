#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <stack>
#include <curand_kernel.h>

// parameters
const double G = 6.67e-11;
const int N_BODIES = 1000;
const int N_DIM = 2;
const double DELTA_T = 1.0;
const int N_SIMULATIONS = 100;
const double LOWER_M = 1e-2;
const double HIGHER_M = 1e-1;
const double LOWER_P = -1e-1;
const double HIGHER_P = 1e-1;
const double LOWER_V = -1e-4;
const double HIGHER_v = 1e-4;

// structures
using Vector = std::array<double, N_DIM>;
using Positions = std::array<Vector, N_BODIES>;
using Velocities = std::array<Vector, N_BODIES>;
using Forces = std::array<Vector, N_BODIES>;
using Masses = std::array<double, N_BODIES>;
using Accelerations = std::array<Vector, N_BODIES>;

// Constants for indexing the Quadrant array
const int QUADRANT_SIZE = 12;
const int CHILDREN_0 = 0;
const int CHILDREN_1 = 1;
const int CHILDREN_2 = 2;
const int CHILDREN_3 = 3;
const int CENTER_OF_MASS_X = 4;
const int CENTER_OF_MASS_Y = 5;
const int TOTAL_MASS = 6;
const int X_MIN = 7;
const int X_MAX = 8;
const int Y_MIN = 9;
const int Y_MAX = 10;
const int PARTICLE_INDEX = 11;

const double THETA = 5e-1;
const int QUADTREE_MAX_DEPTH = 10;
const int QUADTREE_MAX_SIZE = static_cast<int>(pow(4, QUADTREE_MAX_DEPTH));

using Quadrant = std::array<double, QUADRANT_SIZE>;
std::vector<Quadrant> quadtree;

const int MAX_BLOCK_SIZE = 1024; // limit for threads in CUDA

double generateRandom(double lower, double upper) {
    return lower + static_cast<double>(std::rand()) / RAND_MAX * (upper - lower);
}

__device__ double generateRandomGpu(double lower, double upper, curandState* state) {
    // log scaling for positive ranges
    if (lower > 0 && upper > 0) {
        // generates [0, 1)
        double rand_val = curand_uniform(state);
        return pow(10.0, log10(lower) + rand_val * (log10(upper) - log10(lower)));
    } else {
        // linear scaling for ranges that cross zero
        // generates [0, 1)
        double rand_val = curand_uniform(state);
        rand_val = rand_val * (upper - lower) + lower;
        return rand_val;
    }
}

double generateLogRandom(double lower, double upper) {
    return std::pow(10, std::log10(lower) + static_cast<double>(std::rand()) / RAND_MAX * (std::log10(upper) - std::log10(lower)));
}

__global__ void initializeCurandStates(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    curand_init(seed, idx, 0, &states[idx]);
}

void initializeMasses(Masses& masses, double LOWER_M, double HIGHER_M) {
    for (double& mass : masses) {
        mass = generateLogRandom(LOWER_M, HIGHER_M);
    }
}

__global__ void initializeMassesGpu(double* masses, double lower, double higher, curandState* states) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;
    
    masses[idx] = generateRandomGpu(lower, higher, &states[idx]);
}

void initializeVectors(Positions& vectors, double lower, double upper) {
    for (auto& vector : vectors) {
        for (double& component : vector) {
            component = generateRandom(lower, upper);
        }
    }
}

__global__ void initializeVectorsGpu(double* vectors, double lower, double upper, curandState* states) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    for (int i = 0; i < N_DIM; ++i) {
        vectors[idx * N_DIM + i] = generateRandomGpu(lower, upper, &states[idx]);
    }
}

void initializeGpu(Masses& masses, Positions& positions, Positions& velocities) {
    // declare the cuda memory
    double* masses_d;
    double* positions_d;
    double* velocities_d;

    // allocate cuda memory
    cudaMalloc( (void**)&masses_d, N_BODIES * sizeof(double));
    cudaMalloc( (void**)&positions_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&velocities_d, N_BODIES * N_DIM * sizeof(double));

    // calculate block and grid
    int blockSize = (N_BODIES <= MAX_BLOCK_SIZE) ? N_BODIES : MAX_BLOCK_SIZE;

    dim3 dimBlock(blockSize);
	  dim3 dimGrid((N_BODIES + blockSize - 1) / blockSize);

    curandState* states_d;
    cudaMalloc((void**)&states_d, N_BODIES * sizeof(curandState));
    initializeCurandStates<<<dimGrid, dimBlock>>>(states_d, time(NULL));
    cudaDeviceSynchronize();

    // execute kernel codes
    initializeMassesGpu<<<dimGrid, dimBlock>>>(masses_d, LOWER_M, HIGHER_M, states_d);
    initializeVectorsGpu<<<dimGrid, dimBlock>>>(positions_d, LOWER_P, HIGHER_P, states_d);
    initializeVectorsGpu<<<dimGrid, dimBlock>>>(velocities_d, LOWER_V, HIGHER_v, states_d);
    cudaDeviceSynchronize();

    // copy the memory back to cpu for cpu simulation
    cudaMemcpy( masses.data(), masses_d, N_BODIES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( positions.data(), positions_d, N_BODIES * N_DIM * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( velocities.data(), velocities_d, N_BODIES * N_DIM * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(masses_d);
    cudaFree(positions_d);
    cudaFree(velocities_d);
    cudaFree(states_d);
}

void InitializeRoot(double x_min, double x_max, double y_min, double y_max) {
    Quadrant root = {-1, -1, -1, -1, 0.0, 0.0, 0.0, x_min, x_max, y_min, y_max, -1};
    quadtree.push_back(root);
}

int DetermineChild(const Vector& pos, const Quadrant& node) {
    double mid_x = (node[X_MIN] + node[X_MAX]) / 2;
    double mid_y = (node[Y_MIN] + node[Y_MAX]) / 2;

    if (pos[0] <  mid_x && pos[1] <  mid_y)  return 0; // Bottom-left
    if (pos[0] >= mid_x && pos[1] <  mid_y)  return 1; // Bottom-right
    if (pos[0] <  mid_x && pos[1] >= mid_y)  return 2; // Top-left
    return 3;                                          // Top-right
}

void QuadInsert(int particle_index, int node_index, const Positions& positions, const Masses& masses, int current_depth) {
    // Check if current depth exceeds maximum depth
    if (current_depth > QUADTREE_MAX_DEPTH) {
        //std::cout<<"reached max depth"<<std::endl;
        // At max depth, aggregate mass and center of mass
        Quadrant& node = quadtree[node_index];
        const Vector& pos = positions[particle_index];
        double mass = masses[particle_index];

        // Update center of mass
        double existing_mass = node[TOTAL_MASS];
        double existing_x = node[CENTER_OF_MASS_X];
        double existing_y = node[CENTER_OF_MASS_Y];

        node[CENTER_OF_MASS_X] = (existing_mass * existing_x + mass * pos[0]) / (existing_mass + mass);
        node[CENTER_OF_MASS_Y] = (existing_mass * existing_y + mass * pos[1]) / (existing_mass + mass);
        node[TOTAL_MASS] += mass;

        if(existing_mass == 0)
            node[PARTICLE_INDEX] = -1 * particle_index - 2; // because 0th and 1st particles
        else
            node[PARTICLE_INDEX] = -1;

        // No need to track individual particle indices at max depth
        return;
    }

    // Ensure node_index is valid
    if (node_index >= quadtree.size()) {
        std::cerr << "Invalid node index: " << node_index << std::endl;
        return;
    }

    Quadrant node = quadtree[node_index];
    const Vector& pos = positions[particle_index];
    double mass = masses[particle_index];

    bool is_empty_leaf = (node[CHILDREN_0] == -1 && node[CHILDREN_1] == -1 &&
                          node[CHILDREN_2] == -1 && node[CHILDREN_3] == -1 &&
                          node[TOTAL_MASS] == 0.0);

    if (is_empty_leaf) {
        // Assign particle to this empty leaf
        node[CENTER_OF_MASS_X] = pos[0];
        node[CENTER_OF_MASS_Y] = pos[1];
        node[TOTAL_MASS] = mass;
        node[PARTICLE_INDEX] = particle_index;
        quadtree[node_index] = node;
        return;
    }

    if (node[TOTAL_MASS] > 0.0 && node[PARTICLE_INDEX] > -1) {
        // Subdivide the node by creating children
        for (int i = 0; i < 4; ++i) {
            // i guess remove this
            if (quadtree.size() >= QUADTREE_MAX_SIZE) {
                std::cerr << "Quadtree reached maximum size during subdivision." << std::endl;
                return; // Prevent exceeding max size
            }

            Quadrant child;
            double mid_x = (node[X_MIN] + node[X_MAX]) / 2.0;
            double mid_y = (node[Y_MIN] + node[Y_MAX]) / 2.0;

            // Initialize child bounds based on quadrant
            if (i == 0) {
                child = {-1, -1, -1, -1, 0.0, 0.0, 0.0, node[X_MIN], mid_x, node[Y_MIN], mid_y, -1};
            } else if (i == 1) {
                child = {-1, -1, -1, -1, 0.0, 0.0, 0.0, mid_x, node[X_MAX], node[Y_MIN], mid_y, -1};
            } else if (i == 2) {
                child = {-1, -1, -1, -1, 0.0, 0.0, 0.0, node[X_MIN], mid_x, mid_y, node[Y_MAX], -1};
            } else {
                child = {-1, -1, -1, -1, 0.0, 0.0, 0.0, mid_x, node[X_MAX], mid_y, node[Y_MAX], -1};
            }

            int child_index = quadtree.size();
            quadtree.push_back(child);
            node[CHILDREN_0 + i] = child_index;
        }

        // Reset current node's mass and particle index
        Vector existing_pos = {node[CENTER_OF_MASS_X], node[CENTER_OF_MASS_Y]};
        double existing_mass = node[TOTAL_MASS];
        int existing_particle_index = static_cast<int>(node[PARTICLE_INDEX]);
        node[CENTER_OF_MASS_X] = 0.0;
        node[CENTER_OF_MASS_Y] = 0.0;
        node[TOTAL_MASS] = 0.0;
        node[PARTICLE_INDEX] = -1;
        quadtree[node_index] = node;

        // Insert the existing particle into the appropriate child
        int existing_child_index = DetermineChild(existing_pos, node);
        QuadInsert(existing_particle_index, node[CHILDREN_0 + existing_child_index], positions, masses, current_depth + 1);
    }

    // Determine which child quadrant the new particle belongs to
    int child_index = DetermineChild(pos, node);
    QuadInsert(particle_index, node[CHILDREN_0 + child_index], positions, masses, current_depth + 1);
}

std::pair<double, Vector> ComputeMass(int node_index) {
    Quadrant& node = quadtree[node_index];

    if (node[CHILDREN_0] == -1) {
        return {node[TOTAL_MASS], {node[CENTER_OF_MASS_X], node[CENTER_OF_MASS_Y]}};
    }

    double total_mass = 0.0;
    Vector com = {0.0, 0.0};

    for (int i = 0; i < 4; ++i) {
        if (node[CHILDREN_0 + i] != -1) {
            auto [child_mass, child_com] = ComputeMass(node[CHILDREN_0 + i]);
            total_mass += child_mass;
            com[0] += child_mass * child_com[0];
            com[1] += child_mass * child_com[1];
        }
    }

    if (total_mass > 0.0) {
        com[0] /= total_mass;
        com[1] /= total_mass;
    }

    node[TOTAL_MASS] = total_mass;
    node[CENTER_OF_MASS_X] = com[0];
    node[CENTER_OF_MASS_Y] = com[1];

    return {total_mass, com};
}

void TraverseTreeToFile(int node_index, std::ofstream& file,
                        const Positions& positions, int depth = 0) {

    const Quadrant& node = quadtree[node_index];

    file << depth << " "
         << node[X_MIN] << " " << node[X_MAX] << " "
         << node[Y_MIN] << " " << node[Y_MAX] << " "
         << node[TOTAL_MASS];

    int occupantIdx = static_cast<int>(node[PARTICLE_INDEX]);
    if (occupantIdx != -1) {
        file << " occupantIndex=" << occupantIdx
             << " occupantPos=(" << positions[occupantIdx][0]
             << "," << positions[occupantIdx][1] << ")";
    } else if (node[TOTAL_MASS] > 0) {
        // index not important, but we put quadrant center of mass
        file << " occupantIndex=" << occupantIdx
             << " occupantPos=(" << node[CENTER_OF_MASS_X]
             << "," << node[CENTER_OF_MASS_Y] << ")";
    }

    file << "\n";

    for (int i = 0; i < 4; ++i) {
        int childIdx = static_cast<int>(node[CHILDREN_0 + i]);
        if (childIdx != -1) {
            TraverseTreeToFile(childIdx, file, positions, depth + 1);
        }
    }
}

std::array<double, 4> ComputeRootBounds(const Positions& positions)
{
    double xMin =  std::numeric_limits<double>::infinity();
    double xMax = -std::numeric_limits<double>::infinity();
    double yMin =  std::numeric_limits<double>::infinity();
    double yMax = -std::numeric_limits<double>::infinity();

    // Find actual min/max among all bodies
    for (int i = 0; i < N_BODIES; ++i) {
        double x = positions[i][0];
        double y = positions[i][1];
        xMin = std::min(xMin, x);
        xMax = std::max(xMax, x);
        yMin = std::min(yMin, y);
        yMax = std::max(yMax, y);
    }

    // pad the bounding box
    double dx = xMax - xMin;
    double dy = yMax - yMin;
    double maxDim = std::max(dx, dy);

    double padFraction = 0.1;
    double pad = padFraction * maxDim;

    // If all bodies are at the same point, maxDim might be 0
    // so we add a small fallback
    if (maxDim == 0.0) {
        pad = 1e-6; // or some minimal bounding
    }

    xMin -= pad;
    xMax += pad;
    yMin -= pad;
    yMax += pad;

    return { xMin, xMax, yMin, yMax };
}

std::vector<Quadrant> buildTree(const Positions& positions, const Masses& masses) {
    quadtree.clear();

    std::array<double, 4> rootBounds = ComputeRootBounds(positions);
    double xMin = rootBounds[0];
    double xMax = rootBounds[1];
    double yMin = rootBounds[2];
    double yMax = rootBounds[3];

    InitializeRoot(xMin, xMax, yMin, yMax);

    for (int i = 0; i < N_BODIES; ++i) {
        QuadInsert(i, 0, positions, masses, 1);
    }
    ComputeMass(0);
    return quadtree;
}

void computeForces(const Positions& positions,
                   const Masses& masses,
                   Forces& forces)
{
    // For each body, traverse the quadtree
    for (int i = 0; i < N_BODIES; ++i)
    {
        Vector sum = {0.0, 0.0};

        Vector pos_i = positions[i];

        // use a stack of node indices
        std::stack<int> nodeStack;
        nodeStack.push(0); // root is index 0

        while (!nodeStack.empty())
        {
            int nodeIndex = nodeStack.top();
            nodeStack.pop();

            const Quadrant& node = quadtree[nodeIndex];

            // if this node has zero mass, skip
            double nodeMass = node[TOTAL_MASS];
            if (nodeMass <= 1e-15) {
                continue;
            }

            // if this node is a leaf with a single occupant
            int occupantIdx = static_cast<int>(node[PARTICLE_INDEX]);
            bool isLeaf = (node[CHILDREN_0] == -1 && 
                           node[CHILDREN_1] == -1 &&
                           node[CHILDREN_2] == -1 &&
                           node[CHILDREN_3] == -1);

            // compute displacement from body i to this node's center of mass
            Vector displacement;
            displacement[0] = node[CENTER_OF_MASS_X] - pos_i[0];
            displacement[1] = node[CENTER_OF_MASS_Y] - pos_i[1];

            double distance_sq = displacement[0]*displacement[0] + displacement[1]*displacement[1];
            double distance    = std::sqrt(distance_sq) + 1e-15; // small offset to avoid division by zero

            // approximate node size
            double dx = node[X_MAX] - node[X_MIN];
            double dy = node[Y_MAX] - node[Y_MIN];
            double node_size = (dx > dy) ? dx : dy;  // max dimension in 2D

            // Barnes-Hut criterion: if node is leaf OR size/distance < THETA
            // => approximate entire subtree as one body
            if (isLeaf || (node_size / distance < THETA))
            {
                // if it's a leaf for occupant i, skip self-interaction
                if (isLeaf && (occupantIdx == i || (occupantIdx + 2) == -i)) {
                    continue;
                }

                // accumulate approximate force
                double force_mag = (G * masses[i] * nodeMass) / (distance_sq);

                // normalized direction
                double nx = displacement[0] / distance;
                double ny = displacement[1] / distance;

                sum[0] += force_mag * nx;
                sum[1] += force_mag * ny;
            }
            else
            {
                for (int c = 0; c < 4; ++c)
                {
                    int childIdx = static_cast<int>(node[CHILDREN_0 + c]);
                    if (childIdx != -1) {
                        nodeStack.push(childIdx);
                    }
                }
            }
        } // end while stack

        // store final force for body i
        forces[i] = sum;
    } // end for each body
}

void updateAccelerations(const Forces& forces, const Masses& masses, Positions& accelerations) {
    for (int i = 0; i < N_BODIES; ++i) {
        for (int k = 0; k < N_DIM; ++k) {
            accelerations[i][k] = forces[i][k] / masses[i];
        }
    }
}

__global__ void updateAccelerationsGpu(double* forces, double* masses, double* accelerations) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    for (int k = 0; k < N_DIM; ++k) {
        accelerations[idx * N_DIM + k] = forces[idx * N_DIM + k] / masses[idx];
    }
}

void updateVelocities(Velocities& velocities, const Positions& accelerations, double DELTA_T) {
    for (int i = 0; i < N_BODIES; ++i) {
        for (int k = 0; k < N_DIM; ++k) {
            velocities[i][k] += accelerations[i][k] * DELTA_T;
        }
    }
}

__global__ void updateVelocitiesGpu(double* velocities, double* accelerations, double DELTA_T) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;
    
    for (int k = 0; k < N_DIM; ++k) {
        velocities[idx * N_DIM + k] += accelerations[idx * N_DIM + k] * DELTA_T;
    }
}

void updatePositions(Positions& positions, const Velocities& velocities, double DELTA_T) {
    for (int i = 0; i < N_BODIES; ++i) {
        for (int k = 0; k < N_DIM; ++k) {
            positions[i][k] += velocities[i][k] * DELTA_T;
        }
    }
}

__global__ void updatePositionsGpu(double* positions, double* velocities, double DELTA_T) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;
    
    for (int k = 0; k < N_DIM; ++k) {
        positions[idx * N_DIM + k] += velocities[idx * N_DIM + k] * DELTA_T;
    }
}

void printBodies(const Masses& masses, const Positions& positions, const Velocities& velocities) {
    for (int i = 0; i < N_BODIES; ++i) {
        std::cout << "Body " << i << ":\n";
        std::cout << "  Mass: " << masses[i] << "\n";
        std::cout << "  Position: [ ";
        for (const double& pos : positions[i]) {
            std::cout << pos << ' ';
        }
        std::cout << "]\n";
        std::cout << "  Velocity: [ ";
        for (const double& vel : velocities[i]) {
            std::cout << vel << ' ';
        }
        std::cout << "]\n";
    }
}

void savePositions(std::string& output_str, const Positions& positions, double time) {
    for (int i = 0; i < N_BODIES; ++i) {
        output_str += std::to_string(time) + " " + std::to_string(i) + " ";
        for (const double& pos : positions[i]) {
            output_str += std::to_string(pos) + " ";
        }
        output_str += "\n";
    }
}

void runSimulationCpu(Masses& masses, Positions& positions, Velocities& velocities) {
    Accelerations accelerations = {};
    Forces forces = {};

    std::ofstream positions_file("positions.txt");
    std::ofstream tree_file_init("quadtree_init.txt");
    std::ofstream tree_file_final("quadtree_final.txt");
    std::string output_str;

    double absolute_t = 0.0;

    savePositions(output_str, positions, absolute_t);
    //printBodies(masses, positions, velocities);

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < N_SIMULATIONS; ++step) {
        absolute_t += DELTA_T;

        // create the quadtree

        if(step == 0)//N_SIMULATIONS - 1)
            start = std::chrono::high_resolution_clock::now();

        quadtree = buildTree(positions, masses);

        if(step == 0){//N_SIMULATIONS - 1) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout <<std::endl << "CPU: Building tree took " << duration.count() << " milliseconds." << std::endl;
        }
        
        // Write the quadtree to a file for visualization
        if (step == 0)
            TraverseTreeToFile(0, tree_file_init, positions);
        else if (step == N_SIMULATIONS - 1)
            TraverseTreeToFile(0, tree_file_final, positions);

        if(step == 0)// N_SIMULATIONS - 1)
            start = std::chrono::high_resolution_clock::now();
        
        computeForces(positions, masses, forces);
        
        if(step == 0){//N_SIMULATIONS - 1) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout <<std::endl << "CPU: Force computations took " << duration.count() << " milliseconds." << std::endl;
        }
        
        if(step == 0)//N_SIMULATIONS - 1)
            start = std::chrono::high_resolution_clock::now();

        updateAccelerations(forces, masses, accelerations);
        
        updateVelocities(velocities, accelerations, DELTA_T);
        
        updatePositions(positions, velocities, DELTA_T);

         if(step == 0){//N_SIMULATIONS - 1) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout <<std::endl << "CPU: The rest took " << duration.count() << " milliseconds." << std::endl;
        }

        savePositions(output_str, positions, absolute_t);
        
    }
    
    positions_file << output_str;
    positions_file.close();
    tree_file_init.close();
    tree_file_final.close();
}

void runSimulationGpu(Masses masses, Positions& positions, Velocities velocities) {
    double* masses_d;
    double* positions_d;
    double* velocities_d;
    double* accelerations_d;
    double* forces_d;
    double* quadtree_d;

    cudaMalloc( (void**)&masses_d, N_BODIES * sizeof(double));
    cudaMalloc( (void**)&positions_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&velocities_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&accelerations_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&forces_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&forces_d, N_BODIES * N_DIM * sizeof(double));

    cudaMemcpy( masses_d, masses.data(), N_BODIES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( positions_d, positions.data(), N_BODIES * N_DIM * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( velocities_d, velocities.data(), N_BODIES * N_DIM * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = (N_BODIES <= MAX_BLOCK_SIZE) ? N_BODIES : MAX_BLOCK_SIZE;

    dim3 dimBlock(blockSize);
	dim3 dimGrid((N_BODIES + blockSize - 1) / blockSize);

    double absolute_t = 0.0;
    
    for (int step = 0; step < N_SIMULATIONS; ++step) {
        absolute_t += DELTA_T;

        //computeForcesGpu<<<dimGrid, dimBlock>>>(positions_d, masses_d, forces_d);
        cudaDeviceSynchronize();
        updateAccelerationsGpu<<<dimGrid, dimBlock>>>(forces_d, masses_d, accelerations_d);
        cudaDeviceSynchronize();
        updateVelocitiesGpu<<<dimGrid, dimBlock>>>(velocities_d, accelerations_d, DELTA_T);
        cudaDeviceSynchronize();
        updatePositionsGpu<<<dimGrid, dimBlock>>>(positions_d, velocities_d, DELTA_T);
    }
    
    cudaMemcpy( positions.data(), positions_d, N_BODIES * N_DIM * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(masses_d);
    cudaFree(positions_d);
    cudaFree(velocities_d);
    cudaFree(accelerations_d);
    cudaFree(forces_d);
}

void checkEqual(const auto& first, const auto& second, const std::string& name) {
    bool allEqual = true;

    for (size_t i = 0; i < first.size(); ++i) {
        for (size_t j = 0; j < first[i].size(); ++j) {
            if (std::fabs(first[i][j] - second[i][j]) > 1e-3) {
                allEqual = false;
                std::cout << "Difference at index [" << i << "][" << j << "]: "
                          << "first = " << first[i][j]
                          << ", second = " << second[i][j]
                          << " , and the diff is: " << std::fabs(first[i][j] - second[i][j]) << std::endl;
                break;
            }
        }
    }
    if (allEqual) {
        std::cout << "\nThe " << name << " are the same.";
    } else {
        std::cout << "\n\n!!!!! The " << name << " are NOT the same !!!!!\n\n";
    }
}

int main() {

    std::srand(static_cast<unsigned>(std::time(0)));

    //temp
    quadtree.reserve(QUADTREE_MAX_SIZE);

    // structures
    Masses masses;
    Positions positions;
    Velocities velocities;

    // initialization
    initializeGpu(masses, positions, velocities);
    
    /*initializeMasses(masses, LOWER_M, HIGHER_M);
    initializeVectors(positions, LOWER_P, HIGHER_P);
    initializeVectors(velocities, LOWER_V, HIGHER_v);
    */

    // cpu simulation run

    Positions positions_cpu = positions;

    auto start_cpu = std::chrono::high_resolution_clock::now();

    runSimulationCpu(masses, positions_cpu, velocities);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

    // gpu simulation run

    Positions positions_gpu = positions;

    auto start_gpu = std::chrono::high_resolution_clock::now();

    //runSimulationGpu(masses, positions_gpu, velocities);
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);

    std::cout<<std::endl<<std::endl;

    checkEqual(positions_cpu, positions_gpu, "final positions");

    std::cout<<std::endl<<std::endl;

    std::cout << "CPU computation took " << duration_cpu.count() << " milliseconds." << std::endl;
    std::cout << "GPU computation took " << duration_gpu.count() << " milliseconds." << std::endl;

    std::cout<<std::endl<<std::endl;

    return 0;
}
