#pragma once

#include <dt/image2d.hpp>

namespace dt
{
  template <typename T>
  image2d<T> imread(const char* filename);
} // namespace dt