#include <dt/border.hpp>
#include <dt/image2d.hpp>

#include <cstring>

#include <gtest/gtest.h>

static constexpr std::uint8_t data[] = {
    3, 2, 5, //
    7, 1, 6, //
    2, 2, 8  //
};

static constexpr std::uint8_t ref_data[] = {
    3, 3, 3, 3, 3, //
    3, 3, 2, 5, 3, //
    3, 7, 1, 6, 3, //
    3, 2, 2, 8, 3, //
    3, 3, 3, 3, 3  //
};

TEST(MedianBorder, Image2DView)
{
  std::uint8_t           out_data[25];
  const dt::image2d_view img(data, 3, 3, 3);
  dt::image2d_view       out(out_data, 5, 5, 5);
  dt::add_median_border<const std::uint8_t>(img, out);
  for (int y = 0; y < out.height(); y++)
  {
    for (int x = 0; x < out.width(); x++)
      ASSERT_EQ(out(x, y), ref_data[y * 5 + x]);
  }
}

TEST(MedianBorder, Image2D)
{
  dt::image2d<std::uint8_t> img(3, 3);
  std::memcpy(img.buffer(), data, img.width() * img.height());
  auto out = dt::add_median_border(img);
  for (int y = 0; y < out.height(); y++)
  {
    for (int x = 0; x < out.width(); x++)
      ASSERT_EQ(out(x, y), ref_data[y * 5 + x]);
  }
}

TEST(CopyBorder, Image2DView)
{
  static constexpr std::uint8_t ref_copy_data[] = {
      3, 2,  5, //
      7, 12, 6, //
      2, 2,  8  //
  };

  std::uint8_t out_data[25];
  std::memset(out_data, 12, 25);
  const dt::image2d_view img(data, 3, 3, 3);
  dt::image2d_view       out(out_data, 3, 3, 3);
  dt::copy_border(img, out);

  for (int y = 0; y < out.height(); y++)
  {
    for (int x = 0; x < out.width(); x++)
      ASSERT_EQ(out(x, y), ref_copy_data[y * 3 + x]);
  }
}