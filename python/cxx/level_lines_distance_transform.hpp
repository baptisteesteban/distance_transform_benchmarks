#pragma once

#include <torch/torch.h>

namespace dt::python
{
  torch::Tensor propagation(const torch::Tensor& img);
} // namespace dt::python