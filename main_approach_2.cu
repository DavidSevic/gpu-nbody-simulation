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
const double LOWER_M = 1e-6;
const double HIGHER_M = 1e6;
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
using Quadrant = std::array<double, 12>;

// Constants for indexing the Quadrant array
constexpr int CHILDREN_0 = 0;
constexpr int CHILDREN_1 = 1;
constexpr int CHILDREN_2 = 2;
constexpr int CHILDREN_3 = 3;
constexpr int CENTER_OF_MASS_X = 4;
constexpr int CENTER_OF_MASS_Y = 5;
constexpr int TOTAL_MASS = 6;
constexpr int X_MIN = 7;
constexpr int X_MAX = 8;
constexpr int Y_MIN = 9;
constexpr int Y_MAX = 10;
constexpr int PARTICLE_INDEX = 11;

std::vector<Quadrant> quadtree;
const double THETA = 5e-1;

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

void QuadInsert(int particle_index, int node_index, const Positions& positions, const Masses& masses) {

    Quadrant node = quadtree[node_index];
    const Vector& pos = positions[particle_index];
    double mass = masses[particle_index];

    bool is_empty_leaf = (node[CHILDREN_0] == -1 && node[CHILDREN_1] == -1 &&
                          node[CHILDREN_2] == -1 && node[CHILDREN_3] == -1 &&
                          node[TOTAL_MASS] == 0.0);

    if (is_empty_leaf) {
        node[CENTER_OF_MASS_X] = pos[0];
        node[CENTER_OF_MASS_Y] = pos[1];
        node[TOTAL_MASS] = mass;
        node[PARTICLE_INDEX] = particle_index;
        // save the copy node
        quadtree[node_index] = node;
        return;
    }
    if (node[TOTAL_MASS] > 0.0 && node[PARTICLE_INDEX] != -1) {
        for (int i = 0; i < 4; ++i) {
            Quadrant child;
            double mid_x = (node[X_MIN] + node[X_MAX]) / 2;
            double mid_y = (node[Y_MIN] + node[Y_MAX]) / 2;

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

        Vector existing_pos = {node[CENTER_OF_MASS_X], node[CENTER_OF_MASS_Y]};
        double existing_mass = node[TOTAL_MASS];
        int existing_particle_index = static_cast<int>(node[PARTICLE_INDEX]);
        node[CENTER_OF_MASS_X] = 0.0;
        node[CENTER_OF_MASS_Y] = 0.0;
        node[TOTAL_MASS] = 0.0;
        node[PARTICLE_INDEX] = -1;
        int existing_child_index = DetermineChild(existing_pos, node);
        // save the copy node
        quadtree[node_index] = node;
        QuadInsert(existing_particle_index, node[CHILDREN_0 + existing_child_index], positions, masses);
    }

    int child_index = DetermineChild(pos, node);
    QuadInsert(particle_index, node[CHILDREN_0 + child_index], positions, masses);
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
        QuadInsert(i, 0, positions, masses);
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
                if (isLeaf && occupantIdx == i) {
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

void updateVelocities(Velocities& velocities, const Positions& accelerations, double DELTA_T) {
    for (int i = 0; i < N_BODIES; ++i) {
        for (int k = 0; k < N_DIM; ++k) {
            velocities[i][k] += accelerations[i][k] * DELTA_T;
        }
    }
}

void updatePositions(Positions& positions, const Velocities& velocities, double DELTA_T) {
    for (int i = 0; i < N_BODIES; ++i) {
        for (int k = 0; k < N_DIM; ++k) {
            positions[i][k] += velocities[i][k] * DELTA_T;
        }
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

void runSimulation(Masses& masses, Positions& positions, Velocities& velocities) {
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

        if(step == N_SIMULATIONS - 1)
            start = std::chrono::high_resolution_clock::now();

        quadtree = buildTree(positions, masses);

        if(step == N_SIMULATIONS - 1) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout <<std::endl << "Building tree took " << duration.count() << " milliseconds." << std::endl;
        }
        
        // Write the quadtree to a file for visualization
        if (step == 0)
            TraverseTreeToFile(0, tree_file_init, positions);
        else if (step == N_SIMULATIONS - 1)
            TraverseTreeToFile(0, tree_file_final, positions);

        if(step == N_SIMULATIONS - 1)
            start = std::chrono::high_resolution_clock::now();
        
        computeForces(positions, masses, forces);
        
        if(step == N_SIMULATIONS - 1) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout <<std::endl << "Force computations took " << duration.count() << " milliseconds." << std::endl;
        }
        
        if(step == N_SIMULATIONS - 1)
            start = std::chrono::high_resolution_clock::now();

        updateAccelerations(forces, masses, accelerations);
        
        updateVelocities(velocities, accelerations, DELTA_T);
        
        updatePositions(positions, velocities, DELTA_T);

         if(step == N_SIMULATIONS - 1) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout <<std::endl << "The rest took " << duration.count() << " milliseconds." << std::endl;
        }

        savePositions(output_str, positions, absolute_t);
        
    }
    
    positions_file << output_str;
    positions_file.close();
    tree_file_init.close();
    tree_file_final.close();
}

int main() {

    std::srand(static_cast<unsigned>(std::time(0)));


    // structures
    Masses masses;
    Positions positions;
    Velocities velocities;


    // initialization
    //initializeGpu(masses, positions, velocities);
    
    initializeMasses(masses, LOWER_M, HIGHER_M);
    initializeVectors(positions, LOWER_P, HIGHER_P);
    initializeVectors(velocities, LOWER_V, HIGHER_v);
    
    // simulation run
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(masses, positions, velocities);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout <<std::endl << "Computation took " << duration.count() << " milliseconds." << std::endl;

    return 0;
}
