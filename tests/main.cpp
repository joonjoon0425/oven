#include "oven/tray/types.hpp"
#include <nlohmann/json_fwd.hpp>
#include <oven/tray/tray.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <oven/utils/configuration.hpp>

int main() {
    oven::Configuration config;
    config.open("../tests/config.json");
    
    oven::SmallVector shape = {config.get<oven::SmallVector>("/shape")};
    float val = config.get<float>("/val");

    oven::Tray tray1 = oven::ones(shape, oven::kInt32);

    std::cout << torch::from_blob(tray1.data().get(), tray1.shape(), torch::TensorOptions().dtype(torch::kInt32)) << std::endl;

    oven::Tray tray2 = oven::full(shape, val, oven::kFloat32);

    std::cout << torch::from_blob(tray2.data().get(), tray2.shape(), torch::TensorOptions().dtype(torch::kFloat32)) << std::endl;
    
}