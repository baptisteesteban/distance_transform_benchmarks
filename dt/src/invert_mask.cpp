#include <dt/image2d.hpp>

namespace dt
{
  image2d<std::uint8_t> invert_mask(const image2d_view<std::uint8_t>& mask)
  {
    image2d<std::uint8_t> res(mask.width(), mask.height());
    for (int y = 0; y < res.height(); ++y)
    {
      for (int x = 0; x < res.width(); x++)
        res(x, y) = (mask(x, y) == 0);
    }
    return res;
  }
} // namespace dt