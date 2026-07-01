#include "oven/tray/operations.hpp"
#include "oven/tray/types.hpp"
#include <ATen/ops/random.h>
#include <c10/core/ScalarType.h>
#include <nlohmann/json_fwd.hpp>
#include <oven/tray/tray.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <oven/utils/configuration.hpp>
#include <torch/types.h>

#include <fstream>

#include "oven/tray/dispatcher.hpp"

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
    
    std::ofstream fout("../tests/output.txt", std::ios::app);
    std::ifstream file("../tests/config.json");

    oven::SmallVector shape1 = {config.get<oven::SmallVector>("/shape1")};
    oven::SmallVector shape2 = {config.get<oven::SmallVector>("/shape2")};
    float val1 = config.get<float>("/val1");
    float val2 = config.get<float>("/val2");
    auto blob = config.get<std::vector<float>>("/blob/data");
    auto blob_shape = config.get<std::vector<int64_t>>("/blob/shape");

    std::string op = config.get<std::string>("/op");
    std::string rd = config.get<std::string>("/rd/name");

    oven::Tray tray1;
    oven::Tray tray2;
    oven::Tray tray3;
    oven::Tray tray4 = oven::ones({1, 2, 3});

    if (rd == "u_real") {
        tray1 = oven::rand(shape1, config.get<float>("/rd/l"), config.get<float>("/rd/h"));
        tray2 = oven::rand(shape2, config.get<float>("/rd/l"), config.get<float>("/rd/h"));
    } else if (rd == "u_int") {
        tray1 = oven::randint(shape1, config.get<float>("/rd/l"), config.get<float>("/rd/h"));
        tray2 = oven::randint(shape2, config.get<float>("/rd/l"), config.get<float>("/rd/h"));
    } else if (rd == "normal") {
        tray1 = oven::randn(shape1, config.get<float>("/rd/mean"), config.get<float>("/rd/std"));
        tray2 = oven::randn(shape2, config.get<float>("/rd/mean"), config.get<float>("/rd/std"));
    }
    tray3 = oven::randint(shape2, val1, val2);

    if (op == "add") tray3 = tray1 + tray2;
    else if (op == "sub") tray3 = tray1 - tray2;
    else if (op == "mul") tray3 = tray1 * tray2;
    else if (op == "div") tray3 = tray1 / tray2;
    else if (op == "exp") tray3 = oven::exp(tray1);
    else if (op == "gather") tray3 = oven::gather(tray1, 1, tray2);
    else if (op == "scatter") {
        fout << "Before scatter: \n" << toTorch(tray1) << std::endl;
        fout << "Index tensor: \n" << toTorch(tray2) << std::endl;
        fout << "Source tensor: \n" << toTorch(tray3) << std::endl;
        oven::scatter_(tray1, 1, tray2, tray3);
        fout << "After scatter: \n" << toTorch(tray1) << std::endl;
    } else {
        if (op == "le") tray3 = tray1 < tray2;
        else if (op == "leq") tray3 = tray1 <= tray2;
        else if (op == "ge") tray3 = tray1 > tray2;
        else if (op == "geq") tray3 = tray1 >= tray2;
        else if (op == "eq") tray3 = tray1 == tray2;
        else if (op == "neq") tray3 = tray1 != tray2;
        tray4 = oven::where(tray3, tray1, tray2);
    }
    auto blob_tray = oven::from_blob(blob.data(), blob_shape, oven::detail::CppTypeToDType_v<float>);
    fout << toTorch(tray1) << std::endl;
    fout << toTorch(tray2) << std::endl;
    fout << toTorch(tray3) << std::endl;
    fout << toTorch(tray4) << std::endl;
    fout << "From blob: \n" << toTorch(blob_tray) << std::endl;
    fout << "OPERATIONS SCALAR_TENSOR\n"
         << toTorch(tray1 < val1)
         << std::endl << toTorch(val2 > tray2) << std::endl;
    fout << file.rdbuf();

    fout.close();
    file.close();
}