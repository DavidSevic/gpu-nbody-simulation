#ifndef N_BODIES
#define N_BODIES 1000 * 40
#endif

#ifndef N_THREADS
#define N_THREADS 1024 * 1
#endif

#ifndef N_SIMULATIONS
#define N_SIMULATIONS 10
#endif

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
#include <sstream>

// parameters
const double G = 6.67e-11;
const int N_DIM = 2;
const double DELTA_T = 1.0;
const double LOWER_M = 1e-1;
const double HIGHER_M = 5e-1;
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
const int QUADTREE_MAX_SIZE = static_cast<int>((pow(4, QUADTREE_MAX_DEPTH) - 1) / 3);

using Quadrant = std::array<double, QUADRANT_SIZE>;
std::vector<Quadrant> quadtree;

const int MAX_BLOCK_SIZE = 1024;
const int MAX_SHARED_MEM_PER_BLOCK_B = 48 * 1024;
const int SHARED_MEM_BANKS_NUM = 32;

const int FIRST_KERNEL_REGISTERS_NUM = 70;
const int SECOND_KERNEL_REGISTERS_NUM = 43;
const int THIRD_KERNEL_REGISTERS_NUM = 32;


std::chrono::microseconds::rep cpu_parallel_duration = 0;
std::chrono::microseconds::rep gpu_parallel_duration = 0;


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

void loadSimulationDataFromText(const std::string& massesFile,
                                 const std::string& positionsFile,
                                 const std::string& velocitiesFile,
                                 size_t n_bodies,
                                 Masses& masses,
                                 Positions& positions,
                                 Velocities& velocities) {
    if (n_bodies > N_BODIES) {
        throw std::out_of_range("Requested number of bodies exceeds N_BODIES.");
    }

    // Helper lambda to load masses
    auto loadMasses = [&](const std::string& filename, Masses& masses) {
        std::ifstream ifs(filename);
        if (!ifs) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::string line;
        for (size_t i = 0; i < n_bodies; ++i) {
            if (!std::getline(ifs, line)) {
                throw std::runtime_error("Not enough mass entries in file: " + filename);
            }
            masses[i] = std::stod(line);
        }
        ifs.close();
    };

    // Helper lambda to load vectors
    auto loadVectors = [&](const std::string& filename, auto& vectors) {
        std::ifstream ifs(filename);
        if (!ifs) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::string line;
        for (size_t i = 0; i < n_bodies; ++i) {
            if (!std::getline(ifs, line)) {
                throw std::runtime_error("Not enough vector entries in file: " + filename);
            }
            std::istringstream iss(line);
            for (size_t dim = 0; dim < N_DIM; ++dim) {
                if (!(iss >> vectors[i][dim])) {
                    throw std::runtime_error("Failed to parse vector component in file: " + filename);
                }
            }
        }
        ifs.close();
    };

    // Load masses
    loadMasses(massesFile, masses);

    // Load positions
    loadVectors(positionsFile, positions);

    // Load velocities
    loadVectors(velocitiesFile, velocities);

    std::cout << "Loaded " << n_bodies << " bodies from text files." << std::endl;
}

int getOptimalBlockSize(int registersPerThread, int sharedMemPerBlock = 0, bool print=false) {

    // GPU limitations (NVIDIA T600)
    const int registersPerSM       = 65536;    // Total registers per SM
    const int maxThreadsPerSM      = 1024;     // Max threads per SM
    const int maxBlocksPerSM       = 16;       // Max blocks per SM
    const int threadsPerBlockLimit = 1024;     // Max threads per block
    const int threadsPerWarp       = 32;       // Threads per warp
    const int maxWarpsPerSM        = 32;       // Max warps per SM
    const int maxSharedMemPerSM    = 65536;    // Max shared memory per SM (bytes)

    // how many blocks per SM by shared memory limit
    int blocksPerSM_sharedMem = maxSharedMemPerSM / (sharedMemPerBlock > 0 ? sharedMemPerBlock : 1);

    // how many threads can fit by register usage
    int threadsPerSM_registers = registersPerSM / registersPerThread;

    // effective blocks per SM
    int effectiveBlocksPerSM = std::min(blocksPerSM_sharedMem, maxBlocksPerSM);

    // effective threads per SM
    int effectiveThreadsPerSM = std::min(threadsPerSM_registers, maxThreadsPerSM);

    // effective optimal blocksize
    int optimalBlockSize = effectiveThreadsPerSM / effectiveBlocksPerSM;

    // round up to be a multiple of warpsize
    optimalBlockSize = (optimalBlockSize / threadsPerWarp) * threadsPerWarp;

    // threads per block limit
    optimalBlockSize = std::min(optimalBlockSize, threadsPerBlockLimit);

    // actual used warps
    int warpsPerSM_actual = effectiveThreadsPerSM / threadsPerWarp;

    // calculate occupancy percentage
    float occupancyPercentage = (static_cast<float>(warpsPerSM_actual) / maxWarpsPerSM) * 100.0f;

    if (print == true) {
        std::cout << "===== calculateLaunchConfig =====" << std::endl;
        std::cout << "Registers per SM:              " << registersPerSM        << std::endl;
        std::cout << "Registers per thread (kernel): " << registersPerThread    << std::endl;
        std::cout << "Shared mem per block (kernel): " << sharedMemPerBlock     <<" bytes"<< std::endl;
        std::cout << "Blocks per SM (max):           " << maxBlocksPerSM        << std::endl;
        std::cout << "Threads per SM (max):          " << maxThreadsPerSM       << std::endl;
        std::cout << "Blocks per SM (shared mem):    " << (sharedMemPerBlock > 0 ? blocksPerSM_sharedMem : '/') << std::endl;
        std::cout << "Effective blocks per SM:       " << effectiveBlocksPerSM  << std::endl;
        std::cout << "Effective threads per SM:      " << effectiveThreadsPerSM << std::endl;
        std::cout << "Optimal block size:            " << optimalBlockSize      << std::endl;
        std::cout << "Occupancy percentage:          " << occupancyPercentage   << "%"<< std::endl;
        std::cout << "================================" << std::endl;
    }

    return optimalBlockSize;
}

__global__ void initializeCurandStatesGpu(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    for (int body_i = idx; body_i < N_BODIES; body_i += N_THREADS) {
        curand_init(seed, body_i, 0, &states[body_i]);
    }
}

void initializeMasses(Masses& masses, double LOWER_M, double HIGHER_M,
                     bool saveToFile = false, const std::string& filename = "masses_init.txt") {
    for (double& mass : masses) {
        mass = generateLogRandom(LOWER_M, HIGHER_M);
    }

    if (saveToFile) {
        std::ofstream ofs(filename);
        if (!ofs) {
            throw std::runtime_error("Failed to open file for writing masses.");
        }
        for (const double& mass : masses) {
            ofs << mass << "\n";
        }
        ofs.close();
        std::cout << "Masses saved to " << filename << std::endl;
    }
}

__global__ void initializeMassesGpu(double* masses, double lower, double higher, curandState* states) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    for (int body_i = idx; body_i < N_BODIES; body_i += N_THREADS) {
        masses[body_i] = generateRandomGpu(lower, higher, &states[body_i]);
    }
}

void initializeVectors(Positions& vectors, double lower, double upper,
                      bool saveToFile = false, const std::string& filename = "vectors_init.txt") {
    for (auto& vector : vectors) {
        for (double& component : vector) {
            component = generateRandom(lower, upper);
        }
    }

    if (saveToFile) {
        std::ofstream ofs(filename);
        if (!ofs) {
            throw std::runtime_error("Failed to open file for writing vectors.");
        }
        for (const Vector& vector : vectors) {
            for (size_t i = 0; i < N_DIM; ++i) {
                ofs << vector[i] << (i < N_DIM - 1 ? " " : "\n");
            }
        }
        ofs.close();
        std::cout << "Vectors saved to " << filename << std::endl;
    }
}

__global__ void initializeVectorsGpu(double* vectors, double lower, double upper, curandState* states) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    for (int body_i = idx; body_i < N_BODIES; body_i += N_THREADS) {
        for (int i = 0; i < N_DIM; ++i) {
            vectors[body_i * N_DIM + i] = generateRandomGpu(lower, upper, &states[body_i]);
        }
    }
}

void initializeCpu(Masses& masses, Positions& positions, Positions& velocities, bool save_to_file) {
    initializeMasses(masses, LOWER_M, HIGHER_M, save_to_file, "masses_init.txt");
    initializeVectors(positions, LOWER_P, HIGHER_P, save_to_file, "positions_init.txt");
    initializeVectors(velocities, LOWER_V, HIGHER_v, save_to_file, "velocities_init.txt");
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
    int blockSize = getOptimalBlockSize(THIRD_KERNEL_REGISTERS_NUM);

    dim3 dimBlock(blockSize);
	dim3 dimGrid((N_THREADS + blockSize - 1) / blockSize);
    //dim3 dimGrid((N_BODIES + blockSize - 1) / blockSize);

    curandState* states_d;
    cudaMalloc((void**)&states_d, N_BODIES * sizeof(curandState));
    initializeCurandStatesGpu<<<dimGrid, dimBlock>>>(states_d, time(NULL));
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
    if (current_depth >= QUADTREE_MAX_DEPTH) {
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
            if (quadtree.size() >= QUADTREE_MAX_SIZE) {
                std::cout << "Quadtree reached maximum size during subdivision." << "current depth: " << current_depth << std::endl;
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

__device__ bool push(int stack[], int *top, int value) {
    if (*top >= QUADTREE_MAX_DEPTH * 3) {
        // stack overflow
        return false;
    }
    stack[++(*top)] = value;
    return true;
}

__device__ bool pop(int stack[], int *top, int *value) {
    if (*top < 0) {
        // stack underflow
        return false;
    }
    *value = stack[(*top)--];
    return true;
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
    } else if (node[TOTAL_MASS] > 0 ) {
        // two cases: parent node (has children) or max_depth node with multiple particles (no children)
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



__global__ void computeForcesGpu(double* positions, double* masses, double* forces, double* globalMemQuadtree, int quadtreeNumElem, bool useSharedMem) {
    
    // declaration of shared memory for quadtree
    extern __shared__ double sharedMemQuadtree[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    double* quadtree = globalMemQuadtree;
    
    if (useSharedMem) {

       for (int i = threadIdx.x; i < quadtreeNumElem * QUADRANT_SIZE; i += blockDim.x) {
            sharedMemQuadtree[i / QUADRANT_SIZE + SHARED_MEM_BANKS_NUM * (i % QUADRANT_SIZE)] = globalMemQuadtree[i];
       }

        __syncthreads();

        quadtree = sharedMemQuadtree;
    }

    // repeat for multiple bodies
    for(int body_i = idx; body_i < N_BODIES; body_i += N_THREADS) {
        
        double sum[2] = {0, 0};

        // max size of the stack is the max depth of the quadtree * 3 + 1
        int nodeStack[QUADTREE_MAX_DEPTH * 3 + 1];
        int stack_top = -1;

        // push the root node (rootIndex is 0)
        if (!push(nodeStack, &stack_top, 0)) {
            printf("Stack overflow while pushing root node.\n");
            return;
        }

        while (stack_top >= 0) {
            int nodeIndex;
            if (!pop(nodeStack, &stack_top, &nodeIndex)) {
                printf("Stack underflow while popping node.\n");
                break;
            }

            double* node = useSharedMem
                ? &quadtree[nodeIndex]
                : &quadtree[nodeIndex * QUADRANT_SIZE];

            const int ACCESS_INDEX = useSharedMem ? SHARED_MEM_BANKS_NUM : 1;

            // if this node has zero mass, skip
            if (node[ACCESS_INDEX * TOTAL_MASS] <= 1e-15) {
                continue;
            }

            // if this node is a leaf with a single occupant
            int occupantIdx = static_cast<int>(node[ACCESS_INDEX * PARTICLE_INDEX]);
            bool isLeaf = (node[ACCESS_INDEX * CHILDREN_0] == -1 && 
                            node[ACCESS_INDEX * CHILDREN_1] == -1 &&
                            node[ACCESS_INDEX * CHILDREN_2] == -1 &&
                            node[ACCESS_INDEX * CHILDREN_3] == -1);

            // compute displacement from body idx to this node's center of mass
            double displacement[2] = {0, 0};
            displacement[0] = node[ACCESS_INDEX * CENTER_OF_MASS_X] - positions[body_i * 2 + 0];
            displacement[1] = node[ACCESS_INDEX * CENTER_OF_MASS_Y] - positions[body_i * 2 + 1];

            double distance_sq = displacement[0]*displacement[0] + displacement[1]*displacement[1];
            double distance    = std::sqrt(distance_sq) + 1e-15; // small offset to avoid division by zero

            // approximate node size
            double dx = node[ACCESS_INDEX * X_MAX] - node[ACCESS_INDEX * X_MIN];
            double dy = node[ACCESS_INDEX * Y_MAX] - node[ACCESS_INDEX * Y_MIN];
            double node_size = (dx > dy) ? dx : dy;  // max dimension in 2D

            // Barnes-Hut criterion: if node is leaf OR size/distance < THETA
            // => approximate entire subtree as one body
            if (isLeaf || (node_size / distance < THETA))
            {
                // if it's a leaf for occupant idx, skip self-interaction
                if (isLeaf && (occupantIdx == body_i || (occupantIdx + 2) == -body_i)) {
                    continue;
                }

                // accumulate approximate force
                double force_mag = (G * masses[body_i] * node[ACCESS_INDEX * TOTAL_MASS]) / (distance_sq);

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
                    int childIdx = static_cast<int>(node[ACCESS_INDEX * (CHILDREN_0  + c)]);
                    if (childIdx != -1) {
                        if (!push(nodeStack, &stack_top, childIdx)) {
                            printf("Stack overflow while pushing child node %d on position %d. particle: %f. thread: %d\n", childIdx, c, node[ACCESS_INDEX * PARTICLE_INDEX], body_i);
                            break;
                        }
                    }
                }
            }
        } // end while stack

        // store final force for body idx
        forces[body_i * 2 + 0] = sum[0];
        forces[body_i * 2 + 1] = sum[1];
    }
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

__global__ void updateAccVelPos(double* forces, double* masses, double* accelerations, double* velocities, double* positions, double DELTA_T) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N_BODIES)
        return;

    for (int body_i = idx; body_i < N_BODIES; body_i += N_THREADS) {
        accelerations[body_i * 2 + 0] = forces[body_i * 2 + 0] / masses[body_i];
        accelerations[body_i * 2 + 1] = forces[body_i * 2 + 1] / masses[body_i];

        velocities[body_i * 2 + 0] += accelerations[body_i * 2 + 0] * DELTA_T;
        velocities[body_i * 2 + 1] += accelerations[body_i * 2 + 1] * DELTA_T;

        positions[body_i * 2 + 0] += velocities[body_i * 2 + 0] * DELTA_T;
        positions[body_i * 2 + 1] += velocities[body_i * 2 + 1] * DELTA_T;
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

void runSimulationCpu(Masses masses, Positions& positions, Velocities velocities) {
    Accelerations accelerations = {};
    Forces forces = {};

    std::ofstream positions_file("positions_cpu.txt");
    std::ofstream tree_file_init("quadtree_init_cpu.txt");
    std::ofstream tree_file_final("quadtree_final_cpu.txt");
    std::string output_str;

    double absolute_t = 0.0;

    savePositions(output_str, positions, absolute_t);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto duration_micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    for (int step = 0; step < N_SIMULATIONS; ++step) {
        absolute_t += DELTA_T;

        // create the quadtree
        quadtree = buildTree(positions, masses);
        
        // Write the quadtree to a file for visualization
        if (step == 0)
            TraverseTreeToFile(0, tree_file_init, positions);
        else if (step == N_SIMULATIONS - 1)
            TraverseTreeToFile(0, tree_file_final, positions);

        start = std::chrono::high_resolution_clock::now();
        
        computeForces(positions, masses, forces);

        updateAccelerations(forces, masses, accelerations);
        
        updateVelocities(velocities, accelerations, DELTA_T);
        
        updatePositions(positions, velocities, DELTA_T);

        end = std::chrono::high_resolution_clock::now();
        duration_micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        cpu_parallel_duration += duration_micro.count();

        savePositions(output_str, positions, absolute_t);
    }
    
    positions_file << output_str;
    positions_file.close();
    tree_file_init.close();
    tree_file_final.close();
}

void runSimulationGpu(Masses masses, Positions& positions, Velocities velocities) {
    // declaration of GPU memory
    double* masses_d;
    double* positions_d;
    double* velocities_d;
    double* accelerations_d;
    double* forces_d;
    double* quadtree_d;

    // files for tree info
    std::ofstream tree_file_init("quadtree_init_gpu.txt");
    std::ofstream tree_file_final("quadtree_final_gpu.txt");

    // allocating GPU memory
    cudaMalloc( (void**)&masses_d, N_BODIES * sizeof(double));
    cudaMalloc( (void**)&positions_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&velocities_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&accelerations_d, N_BODIES * N_DIM * sizeof(double));
    cudaMalloc( (void**)&forces_d, N_BODIES * N_DIM * sizeof(double));

    // allocation of maximum size quadtree
    size_t realistic_size = std::min(4 * N_BODIES, QUADTREE_MAX_SIZE);
    cudaMalloc( (void**)&quadtree_d, realistic_size * sizeof(Quadrant));

    // copying initial values to GPU
    cudaMemcpy( masses_d, masses.data(), N_BODIES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( positions_d, positions.data(), N_BODIES * N_DIM * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( velocities_d, velocities.data(), N_BODIES * N_DIM * sizeof(double), cudaMemcpyHostToDevice);

    double absolute_t = 0.0;

    // timers
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto duration_micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //debug
    auto duration_micro_alloc = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration_force_computing = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    for (int step = 0; step < N_SIMULATIONS; ++step) {
        absolute_t += DELTA_T;

        // build the tree on cpu
        quadtree = buildTree(positions, masses);

        // writing tree info in first and last iteration
        if (step == 0)
            TraverseTreeToFile(0, tree_file_init, positions);
        else if (step == N_SIMULATIONS - 1)
            TraverseTreeToFile(0, tree_file_final, positions);

        // copy tree to gpu
        cudaMemcpy( quadtree_d, quadtree.data(), quadtree.size() * sizeof(Quadrant), cudaMemcpyHostToDevice);

        // calculate the quadtree size
        int quadtreeMemSize = quadtree.size() * sizeof(Quadrant);

        // use shared memory if tree can fit in it
        int sharedMemSize = quadtreeMemSize <= MAX_SHARED_MEM_PER_BLOCK_B ? quadtreeMemSize : 0;

        // calculate blocksize
        int blockSize = getOptimalBlockSize(FIRST_KERNEL_REGISTERS_NUM, sharedMemSize);
        
        // defining dimensions
        dim3 dimBlock(blockSize);
        
        // depends on N_THREADS for arbitrary number of threads approach
        dim3 dimGrid((N_THREADS + blockSize - 1) / blockSize);

        start = std::chrono::high_resolution_clock::now();

        // pass quadtree memory size for dynamic allocation of shared memory
        computeForcesGpu<<<dimGrid, dimBlock, sharedMemSize>>>(positions_d, masses_d, forces_d, quadtree_d, quadtree.size(), sharedMemSize > 0);
        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();
        duration_micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        blockSize = getOptimalBlockSize(SECOND_KERNEL_REGISTERS_NUM);

        // Set up kernel launch configuration
        dimBlock = dim3(blockSize);
        
        // depends on N_THREADS for arbitrary number of threads approach
        dimGrid = dim3((N_THREADS + blockSize - 1) / blockSize);

        updateAccVelPos<<<dimGrid, dimBlock>>>(forces_d, masses_d, accelerations_d, velocities_d, positions_d, DELTA_T);
        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();
        duration_micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        gpu_parallel_duration += duration_micro.count();

        // needed for next iterations's tree creation on cpu
        cudaMemcpy( positions.data(), positions_d, N_BODIES * N_DIM * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // deallocation of GPU memory
    cudaFree(masses_d);
    cudaFree(positions_d);
    cudaFree(velocities_d);
    cudaFree(accelerations_d);
    cudaFree(forces_d);
    cudaFree(quadtree_d);

    // closing of used files
    tree_file_init.close();
    tree_file_final.close();
}

// used for debugging
void checkEqual(const auto& first, const auto& second, const std::string& name) {
    bool allEqual = true;

    for (size_t i = 0; i < first.size(); ++i) {
        for (size_t j = 0; j < first[i].size(); ++j) {
            if (std::fabs(first[i][j] - second[i][j]) > 1e-10) {
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

    // structures
    
    Masses masses;
    Positions positions;
    Velocities velocities;

    // initialization

    //initializeCpu(masses, positions, velocities, true);
    initializeGpu(masses, positions, velocities);

    // load saved initialization values
    //loadSimulationDataFromText("masses_init.txt", "positions_init.txt", "velocities_init.txt",
    //                                N_BODIES, masses, positions, velocities);

    // cpu simulation run

    Positions positions_cpu = positions;

    auto start_cpu = std::chrono::high_resolution_clock::now();

    //runSimulationCpu(masses, positions_cpu, velocities);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

    // gpu simulation run

    Positions positions_gpu = positions;

    auto start_gpu = std::chrono::high_resolution_clock::now();

    runSimulationGpu(masses, positions_gpu, velocities);
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);

    std::cout<<std::endl<<std::endl;

    //checkEqual(positions, positions_gpu, "final positions");

    std::cout<<std::endl<<std::endl;

    //std::cout << "CPU total computation took " << duration_cpu.count() << " milliseconds." << std::endl;
    std::cout << "GPU total computation took " << duration_gpu.count() << " milliseconds." << std::endl;

    std::cout<<std::endl<<std::endl;

    //std::cout << "CPU 'parallel' computation took " << cpu_parallel_duration << " microseconds." << std::endl;
    std::cout << "GPU parallel computation took " << gpu_parallel_duration << " microseconds." << std::endl;

    return 0;
}
