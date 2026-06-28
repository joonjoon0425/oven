#include "oven/tray/types.hpp"
#include <oven/tray/tray.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

int main() {
    oven::Tray tray1 = oven::ones({1, 2}, oven::kInt32);

    std::cout << torch::from_blob(tray1.data().get(), tray1.shape(), torch::TensorOptions().dtype(torch::kInt32)) << std::endl;

    oven::Tray tray2 = oven::full({1, 2}, 4.3, oven::kFloat32);

    std::cout << torch::from_blob(tray2.data().get(), tray2.shape(), torch::TensorOptions().dtype(torch::kFloat32)) << std::endl;

    oven::Tray tray3 = oven::zeros({1, 2}, oven::kFloat64);

    oven::Tray tray4 = oven::zeros({1, 2}, oven::kInt64);

    
}