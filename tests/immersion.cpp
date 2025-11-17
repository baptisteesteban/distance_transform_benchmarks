#include <dt/image2d.hpp>
#include <dt/immersion.hpp>
#include <dt/random_image2d.hpp>
#include <dt/transfert.hpp>

#include <cstring>

#include <gtest/gtest.h>

static constexpr std::uint8_t data[] = {
    3, 2, 5, //
    7, 1, 6, //
    2, 2, 8  //
};
static constexpr std::uint8_t m_ref[] = {
    3, 2, 2, 2, 5, //
    3, 1, 1, 1, 5, //
    7, 1, 1, 1, 6, //
    2, 1, 1, 1, 6, //
    2, 2, 2, 2, 8  //
};
static constexpr std::uint8_t M_ref[] = {
    3, 3, 2, 5, 5, //
    7, 7, 2, 6, 6, //
    7, 7, 1, 6, 6, //
    7, 7, 2, 8, 8, //
    2, 2, 2, 8, 8  //
};

TEST(Immersion, Image2DView)
{
  std::uint8_t     m_data[25];
  std::uint8_t     M_data[25];
  dt::image2d_view img(data, 3, 3, 3);
  dt::image2d_view m(m_data, 5, 5, 5);
  dt::image2d_view M(M_data, 5, 5, 5);
  dt::immersion(img, m, M);

  for (int y = 0; y < m.height(); y++)
  {
    for (int x = 0; x < m.width(); x++)
    {
      ASSERT_EQ(m(x, y), m_ref[y * 5 + x]);
      ASSERT_EQ(M(x, y), M_ref[y * 5 + x]);
    }
  }
}

TEST(Immersion, Image2D)
{
  dt::image2d<std::uint8_t> img(3, 3);
  std::memcpy(img.buffer(), data, img.width() * img.height());
  auto [m, M] = dt::immersion(img);

  ASSERT_EQ(m.width(), 5);
  ASSERT_EQ(m.height(), 5);
  ASSERT_EQ(m.width(), M.width());
  ASSERT_EQ(m.height(), M.height());
  for (int y = 0; y < m.height(); y++)
  {
    for (int x = 0; x < m.width(); x++)
    {
      ASSERT_EQ(m(x, y), m_ref[y * 5 + x]);
      ASSERT_EQ(M(x, y), M_ref[y * 5 + x]);
    }
  }
}

TEST(Immersion, Image2DGPU)
{
  auto img                  = dt::random_image2d<std::uint8_t>(200, 200);
  const auto [m_ref, M_ref] = dt::immersion(img);
  auto d_img                = dt::host_to_device(img);
  const auto [d_m, d_M]     = dt::immersion_gpu(d_img);
  const auto m              = dt::device_to_host(d_m);
  const auto M              = dt::device_to_host(d_M);

  ASSERT_EQ(m.width(), m_ref.width());
  ASSERT_EQ(m.height(), m_ref.height());
  ASSERT_EQ(m.width(), M.width());
  ASSERT_EQ(m.height(), M.height());

  for (int y = 0; y < m.height(); y++)
  {
    for (int x = 0; x < m.width(); x++)
    {
      ASSERT_EQ(m(x, y), m_ref(x, y));
      ASSERT_EQ(M(x, y), M_ref(x, y));
    }
  }
}