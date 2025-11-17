#include "helpers.hpp"
#include <dt/image2d.hpp>
#include <dt/immersion.hpp>
#include <dt/iterative_distance_transform.hpp>
#include <dt/propagation.hpp>

#include <gtest/gtest.h>

static constexpr std::uint8_t data[] = {
    3, 3, 3, 3, 3, //
    3, 3, 2, 5, 3, //
    3, 7, 1, 6, 3, //
    3, 2, 2, 8, 3, //
    3, 3, 3, 3, 3  //
};

static constexpr std::uint16_t ref_D_data[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 1, 0, 2, 0, 0, //
    0, 0, 0, 0, 1, 0, 2, 0, 0, //
    0, 0, 4, 0, 2, 0, 3, 0, 0, //
    0, 0, 0, 0, 1, 0, 3, 0, 0, //
    0, 0, 1, 1, 1, 0, 5, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0  //
};

TEST(LevelLinesDistanceTransform, Propagation)
{
  const dt::image2d_view ref_D(ref_D_data, 9, 9, 18);

  dt::image2d_view img(data, 5, 5, 5);
  const auto [m, M] = dt::immersion(img);
  const auto D      = dt::propagation<std::uint16_t>(m, M);

  ASSERT_IMAGES_EQ(D, ref_D);
}

TEST(LevelLinesDistanceTransform, Iterative)
{
  const dt::image2d_view ref_D(ref_D_data, 9, 9, 18);

  dt::image2d_view img(data, 5, 5, 5);
  const auto [m, M] = dt::immersion(img);
  const auto D      = dt::iterative_distance_transform<std::uint16_t>(m, M);

  ASSERT_IMAGES_EQ(D, ref_D);
}