#include "dt/propagation.hpp"
#include <dt/border.hpp>
#include <dt/immersion.hpp>
#include <dt/level_lines_distance_transform.hpp>
#include <dt/random_image2d.hpp>
#include <dt/transfert.hpp>

#include <gtest/gtest.h>

#include <cstdint>

#include "helpers.hpp"

TEST(LevelLinesDistanceTransformGPU, FG)
{
  const auto img = dt::add_median_border(dt::random_image2d<std::uint8_t>(200, 200));

  // CPU version
  const auto [m, M] = dt::immersion(img);
  const auto D_cpu  = dt::propagation<std::uint32_t>(m, M);

  // GPU version
  const auto d_img      = dt::host_to_device(img);
  const auto [d_m, d_M] = dt::immersion_gpu(d_img);
  const auto d_D        = dt::level_lines_distance_transform_fg_gpu(d_m, d_M);
  const auto D_gpu      = dt::device_to_host(d_D);

  ASSERT_IMAGES_EQ(D_cpu, D_gpu);
}

TEST(LevelLinesDistanceTransformGPU_Chessboard, FG)
{
  const auto img = dt::add_median_border(dt::random_image2d<std::uint8_t>(200, 200));

  // CPU version
  const auto [m, M] = dt::immersion(img);
  const auto D_cpu  = dt::propagation<std::uint32_t>(m, M);

  // GPU version
  const auto d_img      = dt::host_to_device(img);
  const auto [d_m, d_M] = dt::immersion_gpu(d_img);
  const auto d_D        = dt::level_lines_distance_transform_chessboard_gpu(d_m, d_M);
  const auto D_gpu      = dt::device_to_host(d_D);

  ASSERT_IMAGES_EQ(D_cpu, D_gpu);
}