#pragma once

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
  }

  template <typename T, typename O>
  image2d<O> propagation(const image2d<T>& m, const image2d<T>& M)
  {
    image2d<O> out(m.width(), m.height());
    propagation<T, O>(m, M, out);
    return out;
  }
} // namespace dt