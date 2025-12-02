#pragma once

#include <dt/image2d.hpp>

namespace dt
{
  void geodesic_distance_transform(const image2d_view<std::uint8_t>& img, const image2d_view<std::uint8_t>& mask,
                                   image2d_view<float>& dist, float v, float lambda);

  image2d<float> geodesic_distance_transform(const image2d_view<std::uint8_t>& img,
                                             const image2d_view<std::uint8_t>& mask, float v, float lambda);
} // namespace dt