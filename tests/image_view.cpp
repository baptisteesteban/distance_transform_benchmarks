#include <dt/image_view.hpp>

#include <gtest/gtest.h>

TEST(Image2DView, default_constructor)
{
  dt::image2d_view<void> img;
  ASSERT_EQ(img.width(), 0);
  ASSERT_EQ(img.height(), 0);
  ASSERT_EQ(img.pitch(), 0);
  ASSERT_EQ(img.buffer(), nullptr);
}

TEST(Image2DView, argument_constructor_and_assignment_operators)
{
  std::uint8_t data[6];

  // Constructor with argument
  dt::image2d_view<void> img(data, 3, 2, 3);
  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.pitch(), 3);
  ASSERT_NE(img.buffer(), nullptr);

  // Copy constructor
  auto img2 = dt::image2d_view<void>(img);
  ASSERT_EQ(img2.width(), 3);
  ASSERT_EQ(img2.height(), 2);
  ASSERT_EQ(img2.pitch(), 3);
  ASSERT_NE(img2.buffer(), nullptr);

  // Move constructor
  auto img3 = dt::image2d_view<void>(std::move(img2));
  ASSERT_EQ(img2.buffer(), nullptr);
  ASSERT_EQ(img3.width(), 3);
  ASSERT_EQ(img3.height(), 2);
  ASSERT_EQ(img3.pitch(), 3);
  ASSERT_NE(img3.buffer(), nullptr);

  // Copy assignment
  auto img4 = img;
  ASSERT_EQ(img4.width(), 3);
  ASSERT_EQ(img4.height(), 2);
  ASSERT_EQ(img4.pitch(), 3);
  ASSERT_NE(img4.buffer(), nullptr);

  // Move assignment
  auto img5 = std::move(img4);
  ASSERT_EQ(img4.buffer(), nullptr);
  ASSERT_EQ(img5.width(), 3);
  ASSERT_EQ(img5.height(), 2);
  ASSERT_EQ(img5.pitch(), 3);
  ASSERT_NE(img5.buffer(), nullptr);
}