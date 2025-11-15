#include <dt/image2d.hpp>
#include <dt/immersion.hpp>
#include <dt/propagation.hpp>

#include <gtest/gtest.h>

static constexpr std::uint8_t data[] = {
    3, 3, 3, 3, 3, //
    3, 3, 2, 5, 3, //
    3, 7, 1, 6, 3, //
    3, 2, 2, 8, 3, //
    3, 3, 3, 3, 3  //
};

static constexpr std::uint16_t ref_D[] = {
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

TEST(LevelLineDistanceTransform, Propagation)
{
  dt::image2d_view img(data, 5, 5, 5);
  const auto [m, M] = dt::immersion(img);
  const auto D      = dt::propagation<std::uint16_t>(m, M);

  ASSERT_EQ(D.width(), 9);
  ASSERT_EQ(D.height(), 9);
  for (int y = 0; y < D.height(); y++)
  {
    for (int x = 0; x < D.width(); x++)
      ASSERT_EQ(D(x, y), ref_D[y * 9 + x]);
  }
}