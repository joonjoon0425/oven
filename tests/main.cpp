#include "oven/space/continuous.hpp"
#include <format>
#include <iostream>
#include <oven/space/discrete.hpp>

int main() {
    oven::environment::ContinuousSpace space(torch::tensor({1., 2., 3.}), torch::tensor({3., 4., 5.}));

    std::cout << space.shape() << std::endl << space.sample() << std::endl << space.contains(torch::tensor({-1,2, 3}, torch::dtype(torch::kFloat32)));
    
    return 0;
}