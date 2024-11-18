#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <fstream>

const double g = 6.67 * pow(10, -11) * 1e9;

int main() {
    // parameters
    const int n = 2;
    const int n_dim = 2;

    // mass

    std::array<double, n> masses = {10.0, 10000.0};

    double lower_m = 10.0;
    double higher_m = 10.0;

    // position

    std::array<std::array<double, n_dim>, n> position_vectors = {{{0.0, 0.0}, {-100.0, 0.0}}};

    double lower_p = -10.0;
    double higher_p = 10.0;

    // velocity

    std::array<std::array<double, n_dim>, n> velocity_vectors = {{{0.0, 0.0}, {0.0, 10.0}}};

    double lower_v = -5;
    double higher_v = 5;
    /*
    std::srand(std::time(nullptr));
    for(int i = 0; i < n; ++i) {
        masses[i] = lower_m + static_cast<double>(std::rand()) / RAND_MAX * (higher_m - lower_m);
        for(int j = 0; j < n_dim; ++j) {
            position_vectors[i][j] = lower_p + static_cast<double>(std::rand()) / RAND_MAX * (higher_p - lower_p);
            velocity_vectors[i][j] = lower_v + static_cast<double>(std::rand()) / RAND_MAX * (higher_v - lower_v);
        }
    }
    */

    std::cout<<std::endl<<"#################### BODIES PROPERTIES ####################"<<std::endl<<std::endl;
    
    for(int i = 0; i < n; ++i) {
        std::cout<<"body "<<i<<":"<<std::endl;
        std::cout<<"mass: "<<masses[i]<<std::endl;
        std::cout<<"initial position: [ ";
        for(double& pos : position_vectors[i]) {
            std::cout<<pos<<' ';
        }
        std::cout<<']'<<std::endl;
        std::cout<<"initial velocity: [ ";
        for(double& vel : velocity_vectors[i]) {
            std::cout<<vel<<' ';
        }
        std::cout<<']'<<std::endl<<std::endl;
    }

    double absolute_t = 0;
    const double delta_t = 0.1;
    int n_simulations = 100;

    std::cout<<"#################### SIMULATION STARTED ####################"<<std::endl<<std::endl;

    // Open file to save positions for plotting
    std::ofstream positions_file("positions.txt");

    std::cout<<"positions at t="<<absolute_t<<":"<<std::endl<<std::endl;
    /*for(int i = 0; i < n; ++i) {
        std::cout<<"[  ";
        for(int k = 0; k < n_dim; ++k){
            std::cout<<position_vectors[i][k]<<' ';
            positions_file << "alo bre"<<"\n";
            // Save positions to file
            positions_file << absolute_t << " " << i << " " << position_vectors[i][0] << " " << position_vectors[i][1] << "\n";
        }
        std::cout<<']'<<std::endl;
    }*/

    for (int i = 0; i < n; ++i) {
        std::cout << "[  ";
        positions_file << absolute_t << " " << i << " ";
        for (int k = 0; k < n_dim; ++k) {
            positions_file << position_vectors[i][k] << " ";
        }
        positions_file << "\n";
        std::cout << "]" << std::endl;
    }

    std::cout<<std::endl;

    while(n_simulations--) {

        absolute_t += delta_t;
        
        // forces
    
        std::array<std::array<double, n_dim>, n> force_vectors;

        for(int i = 0; i < n; ++i) {
            // calculating the sum
            std::array<double, n_dim> sum = {};
            for(int j = 0; j < n; ++j) {
                if(j == i)
                    continue;
                // calculating the euclidian distance and displacement
                double euclid_distance = 0;
                std::array<double, n_dim> displacement;
                for(int k = 0; k < n_dim; ++k) {
                    euclid_distance += (position_vectors[j][k] - position_vectors[i][k]) * (position_vectors[j][k] - position_vectors[i][k]);
                    displacement[k] = position_vectors[j][k] - position_vectors[i][k];
                }
                euclid_distance = sqrt(euclid_distance);
                for(int k = 0; k < n_dim; ++k) {
                    sum[k] += (masses[i] * masses[j] * displacement[k]) / (euclid_distance * euclid_distance * euclid_distance);
                }
            }
            for(int k = 0; k < n_dim; ++k) {
                force_vectors[i][k] = g * sum[k];
            }
        }

        // accelerations

        std::array<std::array<double, n_dim>, n> acceleration_vectors;

        for(int i = 0; i < n; ++i) {
            for(int k = 0; k < n_dim; ++k) {
                acceleration_vectors[i][k] = force_vectors[i][k] / masses[i];
                //std::cout<<"a:"<<acceleration_vectors[i][k]<<std::endl;
            }
        }

        // velocities

        for(int i = 0; i < n; ++i) {
            for(int k = 0; k < n_dim; ++k)
                velocity_vectors[i][k] = velocity_vectors[i][k] + acceleration_vectors[i][k] * delta_t;
        }

        // new positions

        for(int i = 0; i < n; ++i) {
            for(int k = 0; k < n_dim; ++k)
                position_vectors[i][k] = position_vectors[i][k] + velocity_vectors[i][k] * delta_t;
        }

        // Print positions at current time step
        std::cout<<"positions at t="<<absolute_t<<":"<<std::endl<<std::endl;
        for (int i = 0; i < n; ++i) {
            std::cout << "[  ";
            positions_file << absolute_t << " " << i << " ";
            for (int k = 0; k < n_dim; ++k) {
                std::cout << position_vectors[i][k] << ' ';
                positions_file << position_vectors[i][k] << " ";
            }
            positions_file << "\n";
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    }
    positions_file.close();
}