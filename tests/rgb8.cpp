#include <dt/rgb.hpp>

#include <gtest/gtest.h>

TEST(Color, Rgb8)
{
  dt::rgb8 c{1, 4, 3};
  ASSERT_EQ(c.r, 1);
  ASSERT_EQ(c.g, 4);
  ASSERT_EQ(c.b, 3);
  ASSERT_EQ(sizeof(dt::rgb8), 3);
}