#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

const double g = 6.67 * pow(10, -11);

int main() {
    int n = 2;

    // mass

    double* masses = new double[n];

    double lower_m = 10.0;
    double higher_m = 20.0;

    // position

    std::vector<double>* vectors = new std::vector<double>[n];

    double lower_p = -1.0;
    double higher_p = 1.0;
    int n_dim = 2;

    //std::srand(std::time(nullptr));
    for(int i = 0; i < n; ++i) {
        masses[i] = lower_m + static_cast<double>(std::rand()) / RAND_MAX * (higher_m - lower_m);
        for(int j = 0; j < n_dim; ++j)
            vectors[i].push_back(lower_p + static_cast<double>(std::rand()) / RAND_MAX * (higher_p - lower_p));
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

    double* forces = new double[n];

    for(int i = 0; i < n; ++i) {
        double sum = 0;
        for(int j = 0; j < n; ++j) {
            if(j == i)
                continue;
            double distance = 0;
            for(int k = 0; k < n_dim; ++k) {
                distance += abs(vectors[i][k] - vectors[j][k]) NIJE DOBRO RACUNANJE EUCLEDIAN DISTANCE
                std::cout<<"vectors[i][k]: "<<vectors[i][k]<<" vectors[j][k]: "<<vectors[j][k]<<" => abs(vectors[i][k] - vectors[j][k]): "<<abs(vectors[i][k] - vectors[j][k])<<std::endl;
            }
            std::cout<<"dist: "<<distance<<std::endl;
            sum += (masses[i] * masses[j] * distance) / abs(distance * distance * distance);
        }
        forces[i] = g * sum;
    }

    std::cout<<"forces:"<<std::endl;
    for(int i = 0; i < n; ++i) {
        std::cout<<forces[i]<<std::endl;
    }
    
    // memory deallocation
    delete[] masses;
    delete[] vectors;
    delete[] forces;
}