#pragma once

#include <dt/image2d.hpp>

#include <cstdint>

namespace dt
{
  void level_lines_distance_transform_fg_gpu(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                                             image2d_view<std::uint32_t>& D, int* nrounds = nullptr);
  image2d<std::uint32_t> level_lines_distance_transform_fg_gpu(const image2d_view<std::uint8_t>& m,
                                                               const image2d_view<std::uint8_t>& M,
                                                               int*                              nrounds = nullptr);

  // Version with cooperative groups
  void                   level_lines_distance_transform_fg_cg_gpu(const image2d_view<std::uint8_t>& m,
                                                                  const image2d_view<std::uint8_t>& M, image2d_view<std::uint32_t>& D,
                                                                  int* nrounds = nullptr);
  image2d<std::uint32_t> level_lines_distance_transform_fg_cg_gpu(const image2d_view<std::uint8_t>& m,
                                                                  const image2d_view<std::uint8_t>& M,
                                                                  int*                              nrounds = nullptr);

  void level_lines_distance_transform_chessboard_gpu(const image2d_view<std::uint8_t>& m,
                                                     const image2d_view<std::uint8_t>& M,
                                                     image2d_view<std::uint32_t>&      D);

  image2d<std::uint32_t> level_lines_distance_transform_chessboard_gpu(const image2d_view<std::uint8_t>& m,
                                                                       const image2d_view<std::uint8_t>& M);
} // namespace dt