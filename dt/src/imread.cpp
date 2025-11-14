#include <dt/imread.hpp>

#include <format>
#include <stdexcept>

#include "stb_image.h"

namespace dt
{
  template <>
  image2d<std::uint8_t> imread(const char* filename)
  {
    int           w, h, n;
    std::uint8_t* data = stbi_load(filename, &w, &h, &n, 0);
    if (!data)
      throw std::runtime_error(std::format("Error while loading {}", filename));
    if (n != 1)
    {
      stbi_image_free(data);
      throw std::invalid_argument("Input image must have 1 component (grayscale) of 8 bits");
    }

    image2d<std::uint8_t> out(w, h);
    for (int y = 0; y < out.height(); y++)
    {
      for (int x = 0; x < out.width(); x++)
        out(x, y) = data[y * w + x];
    }
    stbi_image_free(data);
    return out;
  }
} // namespace dt