#pragma once

#include <dt/image2d.hpp>

#include <limits>
#include <random>

namespace dt
{
  template <typename T>
  image2d<T> random_image2d(int width, int height)
  {
    std::random_device               r;
    std::default_random_engine       e(r());
    std::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    image2d<T> img(width, height);
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
        img(x, y) = dist(e);
    }
    return img;
  }
} // namespace dt