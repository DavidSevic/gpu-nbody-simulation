#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>

const double g = 6.67 * pow(10, -11);

int main() {
    // parameters
    const int n = 2;
    const int n_dim = 2;

    // mass

    std::array<double, n> masses;

    double lower_m = 10.0;
    double higher_m = 20.0;

    // position

    std::array<std::array<double, n_dim>, n> vectors;

    double lower_p = -1.0;
    double higher_p = 1.0;

    //std::srand(std::time(nullptr));
    for(int i = 0; i < n; ++i) {
        masses[i] = lower_m + static_cast<double>(std::rand()) / RAND_MAX * (higher_m - lower_m);
        for(int j = 0; j < n_dim; ++j)
            vectors[i][j] = lower_p + static_cast<double>(std::rand()) / RAND_MAX * (higher_p - lower_p);
    }
    
    for(int i = 0; i < n; ++i) {
        std::cout<<"mass: "<<masses[i]<<std::endl;
        std::cout<<"position: [ ";
        for(double& pos : vectors[i]) {
            std::cout<<pos<<' ';
        }
        std::cout<<']'<<std::endl<<std::endl;
    }

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
                euclid_distance += (vectors[i][k] - vectors[j][k]) * (vectors[i][k] - vectors[j][k]);
                displacement[k] = vectors[i][k] - vectors[j][k];
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

    std::cout<<"forces:"<<std::endl;
    for(int i = 0; i < n; ++i) {
        std::cout<<"[  ";
        for(int k = 0; k < n_dim; ++k){
            std::cout<<force_vectors[i][k]<<' ';
        }
        std::cout<<']'<<std::endl;
    }
}