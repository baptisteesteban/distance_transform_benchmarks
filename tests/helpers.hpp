#pragma once

#include <dt/image2d_view.hpp>

#include <gtest/gtest.h>

#include <format>
#include <type_traits>

template <typename T, typename V>
testing::AssertionResult images_equality_comparison(const dt::image2d_view<T>& img1, const dt::image2d_view<V>& img2)
{
  if (img1.width() != img2.width())
    return testing::AssertionFailure() << std::format("The images have different width (img1: {}, img2: {})",
                                                      img1.width(), img2.width());
  if (img1.height() != img2.height())
    return testing::AssertionFailure() << std::format("The images have different height (img1: {}, img2: {})",
                                                      img1.height(), img2.height());

  if (img1.memory_kind() != dt::e_memory_kind::CPU)
    return testing::AssertionFailure() << "Img1 data must be on CPU to be tested";
  if (img2.memory_kind() != dt::e_memory_kind::CPU)
    return testing::AssertionFailure() << "Img2 data must be on CPU to be tested";

  for (int x = 0; x < img1.width(); x++)
  {
    for (int y = 0; y < img1.height(); y++)
    {
      if (img1(x, y) != img2(x, y))
        return testing::AssertionFailure()
               << std::format("img1({}, {}) ({}) != img2({}, {}) ({})", x, y, img1(x, y), x, y, img2(x, y));
    }
  }
  return testing::AssertionSuccess();
}

#define ASSERT_IMAGES_EQ(img1, img2) ASSERT_TRUE(images_equality_comparison(img1, img2));