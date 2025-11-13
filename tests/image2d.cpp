#include <dt/image2d.hpp>

#include <gtest/gtest.h>

TEST(Image2D, uint8_cpu)
{
  dt::image2d<std::uint8_t> img(3, 2);

  // Test meta data
  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.pitch(), 3);
  ASSERT_EQ(img.elem_size(), 1);
  ASSERT_NE(img.buffer(), nullptr);
}