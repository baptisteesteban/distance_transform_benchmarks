#include <dt/image2d.hpp>
#include <dt/random_image2d.hpp>
#include <dt/transfert.hpp>

#include <gtest/gtest.h>

#include "helpers.hpp"

TEST(Transfert, id_uint8)
{
  const auto img    = dt::random_image2d<std::uint8_t>(20, 20);
  const auto d_img  = dt::host_to_device(img);
  const auto result = dt::device_to_host(d_img);
  ASSERT_IMAGES_EQ(img, result);
}

TEST(Transfert, id_uint32)
{
  const auto img    = dt::random_image2d<std::uint32_t>(20, 20);
  const auto d_img  = dt::host_to_device(img);
  const auto result = dt::device_to_host(d_img);
  ASSERT_IMAGES_EQ(img, result);
}