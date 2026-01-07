#pragma once

#include <dt/image2d.hpp>

namespace dt
{
  void generalised_distance_transform(const image2d_view<std::uint8_t>& img, const image2d_view<std::uint8_t>& mask,
                                      image2d_view<float>& dist, float lambda, float v = 1e10);

  image2d<float> generalised_distance_transform(const image2d_view<std::uint8_t>& img,
                                                const image2d_view<std::uint8_t>& mask, float lambda, float v = 1e10);

  void generalised_distance_transform_blocks(const image2d_view<std::uint8_t>& img,
                                             const image2d_view<std::uint8_t>& mask, image2d_view<float>& dist,
                                             float lambda, float v = 1e10);

  image2d<float> generalised_distance_transform_blocks(const image2d_view<std::uint8_t>& img,
                                                       const image2d_view<std::uint8_t>& mask, float lambda,
                                                       float v = 1e10);

  void generalised_distance_transform_task(const image2d_view<std::uint8_t>& img,
                                           const image2d_view<std::uint8_t>& mask, image2d_view<float>& D, float lambda,
                                           float v = 1e10);

  image2d<float> generalised_distance_transform_task(const image2d_view<std::uint8_t>& img,
                                                     const image2d_view<std::uint8_t>& mask, float lambda,
                                                     float v = 1e10);
} // namespace dt