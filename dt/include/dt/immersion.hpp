#pragma once

#include <dt/image2d.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>

namespace dt
{
  template <typename T, typename O = std::remove_cvref_t<T>>
  void immersion(const image2d_view<T>& img, image2d_view<O>& m, image2d_view<O>& M) noexcept
  {
    assert(m.width() == 2 * img.width() - 1 && m.height() == 2 * img.height() - 1);
    assert(m.width() == M.width() && m.height() == M.height());

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

  template <typename T, typename O = std::remove_cvref_t<T>>
  std::pair<image2d<O>, image2d<O>> immersion(const image2d<T>& img)
  {
    int        W = 2 * img.width() - 1;
    int        H = 2 * img.height() - 1;
    image2d<O> m(W, H);
    image2d<O> M(W, H);
    immersion(img, m, M);
    return {std::move(m), std::move(M)};
  }
} // namespace dt