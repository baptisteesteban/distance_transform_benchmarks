#pragma once

#include <dt/image2d.hpp>

namespace dt
{
  void geodesic_distance_transform(const image2d_view<std::uint8_t>& img, const image2d_view<std::uint8_t>& mask,
                                   image2d_view<float>& dist, float lambda, float v = 1e10);

  image2d<float> geodesic_distance_transform(const image2d_view<std::uint8_t>& img,
                                             const image2d_view<std::uint8_t>& mask, float lambda, float v = 1e10);

  void geodesic_distance_transform_chessboard(const image2d_view<std::uint8_t>& img,
                                              const image2d_view<std::uint8_t>& mask, image2d_view<float>& dist,
                                              float lambda, float v = 1e10);

  image2d<float> geodesic_distance_transform_chessboard(const image2d_view<std::uint8_t>& img,
                                                        const image2d_view<std::uint8_t>& mask, float lambda,
                                                        float v = 1e10);

  void geodesic_distance_transform_task(const image2d_view<std::uint8_t>& img, const image2d_view<std::uint8_t>& mask,
                                        image2d_view<float>& D, float lambda, float v = 1e10);

  image2d<float> geodesic_distance_transform_task(const image2d_view<std::uint8_t>& img,
                                                  const image2d_view<std::uint8_t>& mask, float lambda, float v = 1e10);
} // namespace dt