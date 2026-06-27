#pragma once
#include <torch/torch.h>

namespace oven::environment {

struct StepResult {
    // TODO
};

class Environment {
};

template <typename Derived>
class EnvironmentCRTP : public Environment{

};

}