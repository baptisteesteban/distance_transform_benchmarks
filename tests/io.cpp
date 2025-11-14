#include <dt/imread.hpp>
#include <dt/imsave.hpp>

#include <gtest/gtest.h>

TEST(Io, uint8)
{
  const std::uint8_t data[] = {10, 3, 5, 8, 1, 3, 0, 4, 10, 4};
  dt::image2d_view   in(data, 5, 2, 5);
  dt::imsave("test_tmp_io.png", in);
  auto out = dt::imread<std::uint8_t>("test_tmp_io.png");

  ASSERT_EQ(in.width(), out.width());
  ASSERT_EQ(in.height(), out.height());
  ASSERT_EQ(in.pitch(), out.pitch());
  ASSERT_EQ(in.elem_size(), out.elem_size());

  for (int y = 0; y < in.height(); y++)
  {
    for (int x = 0; x < in.width(); x++)
      ASSERT_EQ(in(x, y), out(x, y));
  }
}