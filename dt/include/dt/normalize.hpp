#pragma once

#include "dt/image2d.hpp"
#include "dt/image_view.hpp"
#include <concepts>
#include <limits>
#include <type_traits>
namespace dt
{
  template <typename T, typename O>
  void normalize(const image2d_view<T>& in, image2d<O>& out) noexcept;

  template <typename O, typename T>
  image2d<O> normalize(const image2d_view<T>& in) noexcept;

  /*
   * Implementation
   */

  template <typename T, typename O>
  void normalize(const image2d_view<T>& in, image2d<O>& out) noexcept
  {
    using V = std::remove_cvref_t<T>;

    // First pass: compute min and max
    V min = std::numeric_limits<V>::max();
    V max = std::numeric_limits<V>::min();
    for (int y = 0; y < in.height(); y++)
    {
      for (int x = 0; x < in.width(); x++)
      {
        min = std::min(min, in(x, y));
        max = std::max(max, in(x, y));
      }
    }

    // Second pass: normalize and store in output image
    const double denom = static_cast<double>(max - min);
    for (int y = 0; y < in.height(); y++)
    {
      for (int x = 0; x < in.width(); x++)
      {
        double v = static_cast<double>(in(x, y) - min) / denom;
        if constexpr (!std::floating_point<O>)
          v *= std::numeric_limits<O>::max();
        out(x, y) = v;
      }
    }
  }

  template <typename O, typename T>
  image2d<O> normalize(const image2d_view<T>& in) noexcept
  {
    image2d<O> out(in.width(), in.height());
    normalize(in, out);
    return out;
  }
} // namespace dt