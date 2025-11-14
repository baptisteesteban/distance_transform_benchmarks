#include <dt/imprint.hpp>

#include <format>
#include <iostream>

namespace dt
{
  template <>
  void imprint<std::uint8_t>(const image2d_view<std::uint8_t>& img, std::ostream& out)
  {
    for (int y = 0; y < img.height(); y++)
    {
      out << '[';
      for (int x = 0; x < img.width() - 1; x++)
        out << std::format("{:3} ", img(x, y));
      out << std::format("{:3}]\n", img(img.width() - 1, y));
    }
  }
} // namespace dt