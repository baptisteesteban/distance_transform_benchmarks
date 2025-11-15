#include <dt/image2d_view.hpp>

#include <gtest/gtest.h>

TEST(Image2DView, default_constructor)
{
  dt::image2d_view<void> img;
  ASSERT_EQ(img.width(), 0);
  ASSERT_EQ(img.height(), 0);
  ASSERT_EQ(img.pitch(), 0);
  ASSERT_EQ(img.buffer(), nullptr);
  ASSERT_FALSE(img.valid());
}

TEST(Image2DView, argument_constructor_and_assignment_operators)
{
  std::uint8_t data[6];

  // Constructor with argument
  dt::image2d_view<void> img(data, 3, 2, 3, 1);
  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.pitch(), 3);
  ASSERT_EQ(img.elem_size(), 1);
  ASSERT_NE(img.buffer(), nullptr);
  ASSERT_TRUE(img.valid());

  // Copy constructor
  auto img2 = dt::image2d_view<void>(img);
  ASSERT_EQ(img2.width(), 3);
  ASSERT_EQ(img2.height(), 2);
  ASSERT_EQ(img2.pitch(), 3);
  ASSERT_EQ(img.elem_size(), 1);
  ASSERT_NE(img2.buffer(), nullptr);
  ASSERT_TRUE(img2.valid());

  // Move constructor
  auto img3 = dt::image2d_view<void>(std::move(img2));
  ASSERT_EQ(img2.buffer(), nullptr);
  ASSERT_FALSE(img2.valid());
  ASSERT_EQ(img3.width(), 3);
  ASSERT_EQ(img3.height(), 2);
  ASSERT_EQ(img3.pitch(), 3);
  ASSERT_EQ(img.elem_size(), 1);
  ASSERT_NE(img3.buffer(), nullptr);
  ASSERT_TRUE(img3.valid());

  // Copy assignment
  auto img4 = img;
  ASSERT_EQ(img4.width(), 3);
  ASSERT_EQ(img4.height(), 2);
  ASSERT_EQ(img4.pitch(), 3);
  ASSERT_EQ(img.elem_size(), 1);
  ASSERT_NE(img4.buffer(), nullptr);
  ASSERT_TRUE(img4.valid());

  // Move assignment
  auto img5 = std::move(img4);
  ASSERT_EQ(img4.buffer(), nullptr);
  ASSERT_FALSE(img4.valid());
  ASSERT_EQ(img5.width(), 3);
  ASSERT_EQ(img5.height(), 2);
  ASSERT_EQ(img5.pitch(), 3);
  ASSERT_EQ(img.elem_size(), 1);
  ASSERT_NE(img5.buffer(), nullptr);
  ASSERT_TRUE(img5.valid());

  // In domain
  ASSERT_FALSE(img.in_domain(-1, 0));
  ASSERT_FALSE(img.in_domain(0, -1));
  ASSERT_FALSE(img.in_domain(-1, -1));
  ASSERT_FALSE(img.in_domain(3, 2));
  ASSERT_TRUE(img.in_domain(0, 0));
}

TEST(Image2DView, TypedView)
{
  std::uint16_t    data[] = {5, 9, 7, 3, 4, 6};
  dt::image2d_view img(data, 3, 2, 6);

  ASSERT_EQ(img.width(), 3);
  ASSERT_EQ(img.height(), 2);
  ASSERT_EQ(img.pitch(), 6);
  ASSERT_EQ(img.elem_size(), 2);
  ASSERT_TRUE(img.valid());

  ASSERT_EQ(img(0, 0), 5);
  ASSERT_EQ(img(1, 0), 9);
  ASSERT_EQ(img(2, 0), 7);
  ASSERT_EQ(img(0, 1), 3);
  ASSERT_EQ(img(1, 1), 4);
  ASSERT_EQ(img(2, 1), 6);
}