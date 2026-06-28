#include "oven/tray/types.hpp"
#include <oven/tray/tray.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

int main() {
    oven::Tray tray1 = oven::zeros({1, 2}, oven::kBool);

    std::cout << torch::from_blob(tray1.data().get(), tray1.shape(), torch::TensorOptions().dtype(torch::kBool)) << std::endl;

    oven::Tray tray2 = oven::zeros({1, 2}, oven::kFloat32);

    std::cout << torch::from_blob(static_cast<float*>(tray2.data().get()), tray2.shape()) << std::endl;

    oven::Tray tray3 = oven::zeros({1, 2}, oven::kFloat64);

    oven::Tray tray4 = oven::zeros({1, 2}, oven::kInt64);
}