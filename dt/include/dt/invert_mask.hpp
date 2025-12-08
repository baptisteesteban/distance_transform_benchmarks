#pragma once

#include <dt/image2d.hpp>

namespace dt
{
  image2d<std::uint8_t> invert_mask(const image2d_view<std::uint8_t>& mask);
}