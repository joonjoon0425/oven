#include "oven/tray/types.hpp"
#include <ATen/ops/random.h>
#include <c10/core/ScalarType.h>
#include <nlohmann/json_fwd.hpp>
#include <oven/tray/tray.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <oven/utils/configuration.hpp>
#include <torch/types.h>

torch::Dtype tt(oven::DType type) {
    if (type == oven::kBool) return torch::kBool;
    else if (type == oven::kInt32) return torch::kInt32;
    else if (type == oven::kInt64) return torch::kInt64;
    else if (type == oven::kFloat32) return torch::kFloat32;
    else if (type == oven::kFloat64) return torch::kFloat64;
}

torch::Tensor toTorch(const oven::Tray& tray) {
    return torch::from_blob(tray.data().get(), tray.shape(), torch::TensorOptions().dtype(tt(tray.dtype())));
}

int main() {
    oven::Configuration config;
    config.open("../tests/config.json");
    
    oven::SmallVector shape1 = {config.get<oven::SmallVector>("/shape1")};
    oven::SmallVector shape2 = {config.get<oven::SmallVector>("/shape2")};
    float val1 = config.get<float>("/val1");
    float val2 = config.get<float>("/val2");

    std::string op = config.get<std::string>("/op");

    oven::Tray tray1 = oven::rand(shape1);
    oven::Tray tray2 = oven::rand(shape2);
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

    std::cout << toTorch(tray1) << std::endl;
    std::cout << toTorch(tray2) << std::endl;
    std::cout << toTorch(tray3) << std::endl;
    std::cout << toTorch(tray4) << std::endl;
}