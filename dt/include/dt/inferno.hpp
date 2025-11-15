#pragma once

#include <dt/image2d.hpp>
#include <dt/rgb.hpp>

namespace dt
{
  image2d<rgb8> inferno(const image2d_view<std::uint8_t>& normalized) noexcept;
}