#include <dt/image2d.hpp>
#include <dt/immersion.hpp>
#include <dt/imprint.hpp>
#include <dt/random_image2d.hpp>
#include <dt/transfert.hpp>

#include <cstring>

#include <gtest/gtest.h>

#include "helpers.hpp"

static constexpr std::uint8_t data[] = {
    3, 2, 5, //
    7, 1, 6, //
    2, 2, 8  //
};
static constexpr std::uint8_t m_data_ref[] = {
    3, 2, 2, 2, 5, //
    3, 1, 1, 1, 5, //
    7, 1, 1, 1, 6, //
    2, 1, 1, 1, 6, //
    2, 2, 2, 2, 8  //
};
static constexpr std::uint8_t M_data_ref[] = {
    3, 3, 2, 5, 5, //
    7, 7, 2, 6, 6, //
    7, 7, 1, 6, 6, //
    7, 7, 2, 8, 8, //
    2, 2, 2, 8, 8  //
};

TEST(Immersion, Image2DView)
{
  const dt::image2d_view m_ref(m_data_ref, 5, 5, 5);
  const dt::image2d_view M_ref(M_data_ref, 5, 5, 5);

  std::uint8_t     m_data[25];
  std::uint8_t     M_data[25];
  dt::image2d_view img(data, 3, 3, 3);
  dt::image2d_view m(m_data, 5, 5, 5);
  dt::image2d_view M(M_data, 5, 5, 5);
  dt::immersion(img, m, M);

  ASSERT_IMAGES_EQ(m, m_ref);
  ASSERT_IMAGES_EQ(M, M_ref);
}

TEST(Immersion, Image2D)
{
  const dt::image2d_view m_ref(m_data_ref, 5, 5, 5);
  const dt::image2d_view M_ref(M_data_ref, 5, 5, 5);

  dt::image2d<std::uint8_t> img(3, 3);
  std::memcpy(img.buffer(), data, img.width() * img.height());
  auto [m, M] = dt::immersion(img);

  ASSERT_IMAGES_EQ(m, m_ref);
  ASSERT_IMAGES_EQ(M, M_ref);
}

TEST(Immersion, Image2DGPU)
{
  constexpr int WIDTH = 200, HEIGHT = 200;
  auto          img         = dt::random_image2d<std::uint8_t>(WIDTH, HEIGHT);
  const auto [m_ref, M_ref] = dt::immersion(img);
  auto d_img                = dt::host_to_device(img);
  const auto [d_m, d_M]     = dt::immersion_gpu(d_img, dt::e_immersion_impl::GLOBAL);
  const auto m              = dt::device_to_host(d_m);
  const auto M              = dt::device_to_host(d_M);

  ASSERT_IMAGES_EQ(m, m_ref);
  ASSERT_IMAGES_EQ(M, M_ref);
}

TEST(Immersion, Image2DGPUSharedMemory)
{
  constexpr int WIDTH = 16, HEIGHT = 16;
  auto          img         = dt::random_image2d<std::uint8_t>(WIDTH, HEIGHT);
  const auto [m_ref, M_ref] = dt::immersion(img);
  auto d_img                = dt::host_to_device(img);
  const auto [d_m, d_M]     = dt::immersion_gpu(d_img);
  const auto m              = dt::device_to_host(d_m);
  const auto M              = dt::device_to_host(d_M);

  ASSERT_IMAGES_EQ(m, m_ref);
  ASSERT_IMAGES_EQ(M, M_ref);
}