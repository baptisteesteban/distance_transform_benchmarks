#pragma once

#include <dt/image2d_view.hpp>

namespace dt
{
  template <typename T>
  void fill(image2d_view<T>& img, const T& val) noexcept
  {
    for (int y = 0; y < img.height(); y++)
    {
      for (int x = 0; x < img.width(); x++)
        img(x, y) = val;
    }
  }
} // namespace dt