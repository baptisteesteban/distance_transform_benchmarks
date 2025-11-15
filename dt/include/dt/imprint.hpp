#pragma once

#include <dt/image2d_view.hpp>

#include <iostream>

namespace dt
{
  template <typename T>
  void imprint(const image2d_view<T>& img, std::ostream& out = std::cout);
} // namespace dt