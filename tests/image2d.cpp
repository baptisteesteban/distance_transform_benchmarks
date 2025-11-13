#include <dt/image2d.hpp>

#include <gtest/gtest.h>

TEST(Image2D, uint8_cpu)
{
  std::uint8_t              data[] = {3, 8, 5, 2, 1, 9};
  dt::image2d<std::uint8_t> img(3, 2);

  // Test meta data
  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.pitch(), 3);
  ASSERT_EQ(img.elem_size(), 1);
  ASSERT_NE(img.buffer(), nullptr);
  ASSERT_TRUE(img.valid());

  // Test read / write
  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      img(x, y) = data[y * 3 + x];
  }

  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      ASSERT_EQ(img(x, y), data[y * 3 + x]);
  }
}

TEST(Image2D, uint16_cpu)
{
  std::uint16_t              data[] = {3, 8, 5, 2, 1, 9};
  dt::image2d<std::uint16_t> img(3, 2);

  // Test meta data
  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.pitch(), 6);
  ASSERT_EQ(img.elem_size(), 2);
  ASSERT_NE(img.buffer(), nullptr);
  ASSERT_TRUE(img.valid());

  // Test read / write
  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      img(x, y) = data[y * 3 + x];
  }

  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
      ASSERT_EQ(img(x, y), data[y * 3 + x]);
  }
}