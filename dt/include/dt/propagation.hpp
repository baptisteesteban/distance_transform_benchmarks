#pragma once

#include "dt/c4.hpp"
#include "dt/point2d.hpp"
#include <algorithm>
#include <dt/image2d.hpp>
#include <dt/image_view.hpp>
#include <dt/structures/circular_bucket_queue.hpp>

#include <limits>
#include <type_traits>

namespace dt
{
  template <typename T, typename O>
  void propagation(const image2d_view<T>& m, const image2d_view<T>& M, image2d_view<O>& out)
  {
    static constexpr O UNVISITED = std::numeric_limits<O>::max();

    // Fill the distance with unvisited
    for (int y = 0; y < out.height(); y++)
    {
      for (int x = 0; x < out.width(); x++)
        out(x, y) = UNVISITED;
    }
    image2d<std::remove_cvref_t<T>> F(out.width(), out.height());
    structures::CircularBucketQueue q;

    // First value (chosen as being a point of the image border)
    const auto root = point2d{0, 0};
    q.push(0, root);
    out(root) = 0;
    F(root)   = m(root);

    // Propagation
    while (!q.empty())
    {
      const auto [d, p] = q.pop();
      for (const auto& n : c4(p))
      {
        if (!out.in_domain(n) || out(n) != UNVISITED)
          continue;

        auto         f     = std::clamp(F(p), m(n), M(n));
        std::uint8_t delta = std::abs(static_cast<std::int16_t>(f) - F(p));
        q.push(delta, n);
        out(n) = d + delta;
        F(n)   = f;
      }
    }
  }

  template <typename O, typename T>
  image2d<O> propagation(const image2d_view<T>& m, const image2d_view<T>& M)
  {
    image2d<O> out(m.width(), m.height());
    propagation<T, O>(m, M, out);
    return out;
  }
} // namespace dt