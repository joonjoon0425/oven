#include <c10/core/TensorOptions.h>
#include <iostream>
#include <torch/cuda.h>
#include <torch/torch.h>

int main() {
    std::cout << "===1. Tensor creation===" << std::endl;
    torch::Tensor t1 = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    std::cout << "t1:\n" << t1 << std::endl;
    std::cout << "Shape: " << t1.sizes() << ", Dtype: " << t1.dtype() << "\n\n";

    std::cout << "===2. CUDA Device check===" << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "===CUDA AVAILABLE! Sending tensor to GPU===" << std::endl;
        torch::Device device(torch::kCUDA);

        torch::Tensor t1_gpu = t1.to(device);

        torch::Tensor t_result = torch::matmul(t1_gpu, t1_gpu.t());
        std::cout << "GPU calculation result of\n" << t1 << "\nand\n" << t1.t() << "\nis\n" << t_result.to(torch::kCPU) << std::endl;
    } else {
        std::cout <<"===CUDA UNAVAILABLE!===" << std::endl;
    }

    std::cout << "===3. Autograd engine===" << std::endl;

    torch::Tensor x = torch::tensor({2.0, 3.0}, torch::requires_grad(true));

    torch::Tensor y = x.pow(2) + 5;

    torch::Tensor loss = y.sum();

    loss.backward();

    std::cout << "Gradient of x is: \n" << x.grad() << std::endl;

    return 0;
}