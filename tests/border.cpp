#include <dt/border.hpp>
#include <dt/image2d.hpp>

#include <gtest/gtest.h>

#include <cstring>

#include "helpers.hpp"

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
  const dt::image2d_view ref_img(ref_data, 5, 5, 5);

  std::uint8_t           out_data[25];
  const dt::image2d_view img(data, 3, 3, 3);
  dt::image2d_view       out(out_data, 5, 5, 5);
  dt::add_median_border<const std::uint8_t>(img, out);
  ASSERT_IMAGES_EQ(out, ref_img);
}

TEST(MedianBorder, Image2D)
{
  const dt::image2d_view ref_img(ref_data, 5, 5, 5);

  dt::image2d<std::uint8_t> img(3, 3);
  std::memcpy(img.buffer(), data, img.width() * img.height());
  auto out = dt::add_median_border(img);
  ASSERT_IMAGES_EQ(out, ref_img);
}

TEST(CopyBorder, Image2DView)
{
  static constexpr std::uint8_t ref_copy_data[] = {
      3, 2,  5, //
      7, 12, 6, //
      2, 2,  8  //
  };
  const dt::image2d_view ref_img(ref_copy_data, 3, 3, 3);

  std::uint8_t out_data[25];
  std::memset(out_data, 12, 25);
  const dt::image2d_view img(data, 3, 3, 3);
  dt::image2d_view       out(out_data, 3, 3, 3);
  dt::copy_border(img, out);
  ASSERT_IMAGES_EQ(out, ref_img);
}