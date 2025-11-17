#include <dt/imread.hpp>
#include <dt/imsave.hpp>

#include <gtest/gtest.h>

#include "helpers.hpp"

TEST(Io, uint8)
{
  const std::uint8_t data[] = {10, 3, 5, 8, 1, 3, 0, 4, 10, 4};
  dt::image2d_view   in(data, 5, 2, 5);
  dt::imsave("test_tmp_io.png", in);
  auto out = dt::imread<std::uint8_t>("test_tmp_io.png");

  ASSERT_IMAGES_EQ(in, out);
}