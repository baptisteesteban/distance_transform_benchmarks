#pragma once

#include <dt/image2d.hpp>

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace dt
{

  template <typename T, typename O = std::remove_cvref_t<T>>
  void immersion(const image2d_view<T>& img, image2d_view<O>& m, image2d_view<O>& M);

  template <typename T, typename O = std::remove_cvref_t<T>>
  std::pair<image2d<O>, image2d<O>> immersion(const image2d_view<T>& img);

  void immersion_gpu(const image2d_view<std::uint8_t>& img, image2d_view<std::uint8_t>& m,
                     image2d_view<std::uint8_t>& M);

  std::pair<image2d<std::uint8_t>, image2d<std::uint8_t>> immersion_gpu(const image2d_view<std::uint8_t>& img);
  /*
   * Implementations
   */

  namespace impl
  {
    template <typename T, typename O = std::remove_cvref_t<T>>
    void immersion_cpu(const image2d_view<T>& img, image2d_view<O>& m, image2d_view<O>& M) noexcept
    {
      // 2-Faces
      for (int y = 0; y < img.height(); y++)
      {
        for (int x = 0; x < img.width(); x++)
        {
          m(2 * x, 2 * y) = img(x, y);
          M(2 * x, 2 * y) = img(x, y);
        }
      }

      // 1-Faces horizontal
      for (int y = 0; y < m.height(); y += 2)
      {
        for (int x = 1; x < m.width(); x += 2)
        {
          const auto [min, max] = std::minmax(m(x - 1, y), m(x + 1, y));
          m(x, y)               = min;
          M(x, y)               = max;
        }
      }

      // 1-Faces vertical
      for (int y = 1; y < m.height(); y += 2)
      {
        for (int x = 0; x < m.width(); x += 2)
        {
          const auto [min, max] = std::minmax(m(x, y - 1), m(x, y + 1));
          m(x, y)               = min;
          M(x, y)               = max;
        }
      }

      // 0-Faces
      for (int y = 1; y < m.height(); y += 2)
      {
        for (int x = 1; x < m.width(); x += 2)
        {
          const auto min_r = std::min(std::min(m(x - 1, y), m(x + 1, y)), std::min(m(x, y - 1), m(x, y + 1)));
          const auto max_r = std::max(std::max(M(x - 1, y), M(x + 1, y)), std::max(M(x, y - 1), M(x, y + 1)));
          m(x, y)          = min_r;
          M(x, y)          = max_r;
        }
      }
    }
  } // namespace impl

  template <typename T, typename O>
  void immersion(const image2d_view<T>& img, image2d_view<O>& m, image2d_view<O>& M)
  {
    assert(img.memory_kind() == m.memory_kind() && img.memory_kind() == M.memory_kind());
    assert(m.width() == 2 * img.width() - 1 && m.height() == 2 * img.height() - 1);
    assert(m.width() == M.width() && m.height() == M.height());
    switch (img.memory_kind())
    {
    case e_memory_kind::CPU:
      impl::immersion_cpu(img, m, M);
      break;
    default:
      throw std::invalid_argument("Unimplemented immersion");
    }
  }

  template <typename T, typename O>
  std::pair<image2d<O>, image2d<O>> immersion(const image2d_view<T>& img)
  {
    int        W = 2 * img.width() - 1;
    int        H = 2 * img.height() - 1;
    image2d<O> m(W, H, img.memory_kind());
    image2d<O> M(W, H, img.memory_kind());
    immersion(img, m, M);
    return {std::move(m), std::move(M)};
  }
} // namespace dt