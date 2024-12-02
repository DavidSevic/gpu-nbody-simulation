#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <fstream>
#include <chrono>
#include <algorithm>

// parameters
const double g = 6.67e-11;
const int n = 2;
const int n_dim = 2;
const double delta_t = 1.0;
const int n_simulations = 100;
const double lower_m = 1e-6;
const double higher_m = 1e6;
const double lower_p = -1e-1;
const double higher_p = 1e-1;
const double lower_v = -1e-4;
const double higher_v = 1e-4;

// structures
using Vector = std::array<double, n_dim>;
using Positions = std::array<Vector, n>;
using Velocities = std::array<Vector, n>;
using Forces = std::array<Vector, n>;
using Masses = std::array<double, n>;
using Accelerations = std::array<Vector, n>;

double generateRandom(double lower, double upper) {
    return lower + static_cast<double>(std::rand()) / RAND_MAX * (upper - lower);
}

double generateLogRandom(double lower, double upper) {
    return std::pow(10, std::log10(lower) + static_cast<double>(std::rand()) / RAND_MAX * (std::log10(upper) - std::log10(lower)));
}

void initializeMasses(Masses& masses, double lower_m, double higher_m) {
    for (double& mass : masses) {
        mass = generateLogRandom(lower_m, higher_m);
    }
}

void initializeVectors(Positions& vectors, double lower, double upper) {
    for (auto& vector : vectors) {
        for (double& component : vector) {
            component = generateRandom(lower, upper);
        }
    }
}

void computeForces(const Positions& positions, const Masses& masses, Forces& forces) {
    for (int i = 0; i < n; ++i) {
        Vector sum = {};
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            double distance_squared = 0.0;
            Vector displacement = {};
            for (int k = 0; k < n_dim; ++k) {
                displacement[k] = positions[j][k] - positions[i][k];
                distance_squared += displacement[k] * displacement[k];
            }

            double distance = std::sqrt(distance_squared);
            double factor = g * masses[i] * masses[j] / (distance_squared * distance);

            for (int k = 0; k < n_dim; ++k) {
                sum[k] += factor * displacement[k];
            }
        }
        forces[i] = sum;
    }
}

__global__ void computeForcesGpu(double* positions, double* masses, double* forces) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    positions[idx] = 1;
    /*
    Vector sum = {};
    for (int j = 0; j < n; ++j) {
            if (idx == j) continue;

            double distance_squared = 0.0;
            Vector displacement = {};
            for (int k = 0; k < n_dim; ++k) {
                displacement[k] = positions[j][k] - positions[i][k];
                distance_squared += displacement[k] * displacement[k];
            }

            double distance = std::sqrt(distance_squared);
            double factor = g * masses[i] * masses[j] / (distance_squared * distance);

            for (int k = 0; k < n_dim; ++k) {
                sum[k] += factor * displacement[k];
            }
        }
        forces[idx] = sum;
    */
}

void updateAccelerations(const Forces& forces, const Masses& masses, Positions& accelerations) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            accelerations[i][k] = forces[i][k] / masses[i];
        }
    }
}

void updateVelocities(Velocities& velocities, const Positions& accelerations, double delta_t) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            velocities[i][k] += accelerations[i][k] * delta_t;
        }
    }
}

void updatePositions(Positions& positions, const Velocities& velocities, double delta_t) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_dim; ++k) {
            positions[i][k] += velocities[i][k] * delta_t;
        }
    }
}

void printBodies(const Masses& masses, const Positions& positions, const Velocities& velocities) {
    for (int i = 0; i < n; ++i) {
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
    for (int i = 0; i < n; ++i) {
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

    std::ofstream positions_file("positions.txt");
    std::string output_str;

    double absolute_t = 0.0;
    savePositions(output_str, positions, absolute_t);
    printBodies(masses, positions, velocities);
    
    for (int step = 0; step < n_simulations; ++step) {
        absolute_t += delta_t;

        computeForces(positions, masses, forces);
        updateAccelerations(forces, masses, accelerations);
        updateVelocities(velocities, accelerations, delta_t);
        updatePositions(positions, velocities, delta_t);

        savePositions(output_str, positions, absolute_t);
    
        break;
    }
    
    positions_file << output_str;
    positions_file.close();
}

void runSimulationGpu(Masses masses, Positions& positions, Velocities velocities) {
    double* masses_d;
    double* positions_d;
    double* velocities_d;
    double* accelerations_d;
    double* forces_d;

    cudaMalloc( (void**)&masses_d, n * sizeof(double));
    cudaMalloc( (void**)&positions_d, n * n_dim * sizeof(double));
    cudaMalloc( (void**)&velocities_d, n * n_dim * sizeof(double));
    cudaMalloc( (void**)&accelerations_d, n * n_dim * sizeof(double));
    cudaMalloc( (void**)&forces_d, n * n_dim * sizeof(double));

    cudaMemcpy( masses_d, masses.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( positions_d, positions.data(), n * n_dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( velocities_d, velocities.data(), n * n_dim * sizeof(double), cudaMemcpyHostToDevice);

    
    for (int step = 0; step < n_simulations; ++step) {
        absolute_t += delta_t;

        computeForcesGpu<<<dimGrid, dimBlock>>>(positions, masses, forces);

        break;
    }
    

    cudaMemcpy( positions.data(), positions_d, n * n_dim * sizeof(double), cudaMemcpyDeviceToHost);
}

int main() {
    //std::srand(static_cast<unsigned>(std::time(0)));

    // structures
    Masses masses;
    Positions positions;
    Velocities velocities;

    // initialization
    initializeMasses(masses, lower_m, higher_m);
    initializeVectors(positions, lower_p, higher_p);
    initializeVectors(velocities, lower_v, higher_v);

    // cpu simulation run

    Positions positions_cpu = positions;

    auto start_cpu = std::chrono::high_resolution_clock::now();

    runSimulationCpu(masses, positions_cpu, velocities);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

    // gpu simulation run

    Positions positions_gpu = positions;

    auto start_gpu = std::chrono::high_resolution_clock::now();

    runSimulationGpu(masses, positions_gpu, velocities);
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);

    

    if(std::equal(positions_cpu.begin(), positions_cpu.end(), positions_gpu.begin())) {
        std::cout<<std::endl<<std::endl<<"The results are the same."<<std::endl<<std::endl;
    } else {
        std::cout<<std::endl<<std::endl<<"!!!!! The results are NOT the same !!!!!"<<std::endl<<std::endl;
    }
    /*
    std::cout<<"cpu: ";
    for (const auto& vec : positions_cpu)
        for (const auto& val : vec)
            std::cout << val << " ";
    std::cout << std::endl;
    std::cout<<"gpu: ";
    for (const auto& vec : positions_gpu)
        for (const auto& val : vec)
            std::cout << val << " ";
    std::cout << std::endl;
    std::cout<<"init: ";
    for (const auto& vec : positions)
        for (const auto& val : vec)
            std::cout << val << " ";
    std::cout << std::endl;
    */

    std::cout << "CPU computation took " << duration_cpu.count() << " milliseconds." << std::endl;
    std::cout << "GPU computation took " << duration_gpu.count() << " milliseconds." << std::endl;

    std::cout<<std::endl<<std::endl;

    return 0;
}
