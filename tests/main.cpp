#include "oven/tray/types.hpp"
#include <nlohmann/json_fwd.hpp>
#include <oven/tray/tray.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <oven/utils/configuration.hpp>

int main() {
    oven::Configuration config;
    config.open("../tests/config.json");
    
    oven::SmallVector shape1 = {config.get<oven::SmallVector>("/shape1")};
    oven::SmallVector shape2 = {config.get<oven::SmallVector>("/shape2")};
    float val = config.get<float>("/val");

    oven::Tray tray1 = oven::ones(shape1, oven::kInt32);
    oven::Tray tray2 = oven::full(shape2, val, oven::kInt32);

    oven::Tray added = tray1 + tray2;

    std::cout << torch::from_blob(added.data().get(), added.shape(), torch::TensorOptions().dtype(torch::kInt32)) << std::endl;
    
}