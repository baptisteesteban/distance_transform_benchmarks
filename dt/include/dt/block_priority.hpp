#pragma once

#include <dt/image2d.hpp>

namespace dt
{
  std::uint32_t compute_block_priorities(const dt::image2d_view<std::uint8_t>& mask, std::uint8_t* priorities,
                                         std::uint32_t* cdf);
} // namespace dt
