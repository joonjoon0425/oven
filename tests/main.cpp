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
    float val1 = config.get<float>("/val1");
    float val2 = config.get<float>("/val2");

    std::string op = config.get<std::string>("/op");

    oven::Tray tray1 = oven::full(shape1, val1);
    oven::Tray tray2 = oven::full(shape2, val2);
    oven::Tray tray3;
    oven::Tray tray4;

    if (op == "add") tray3 = tray1 + tray2;
    else if (op == "sub") tray3 = tray1 - tray2;
    else if (op == "mul") tray3 = tray1 * tray2;
    else if (op == "div") tray3 = tray1 / tray2;
    else {
        if (op == "le") tray3 = tray1 < tray2;
        else if (op == "leq") tray3 = tray1 <= tray2;
        else if (op == "ge") tray3 = tray1 > tray2;
        else if (op == "geq") tray3 = tray1 >= tray2;
        else if (op == "eq") tray3 = tray1 == tray2;
        else if (op == "neq") tray3 = tray1 != tray2;
        tray4 = oven::where(tray3, tray1, tray2);
    }

    std::cout << torch::from_blob(tray1.data().get(), tray1.shape(), torch::TensorOptions().dtype(torch::kFloat32)) << std::endl;
    std::cout << torch::from_blob(tray2.data().get(), tray2.shape(), torch::TensorOptions().dtype(torch::kFloat32)) << std::endl;
    std::cout << torch::from_blob(tray3.data().get(), tray3.shape(), torch::TensorOptions().dtype(torch::kBool)) << std::endl;
    std::cout << torch::from_blob(tray4.data().get(), tray3.shape(), torch::TensorOptions().dtype(torch::kFloat32)) << std::endl;
}