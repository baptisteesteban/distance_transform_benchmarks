#pragma once

#include "dt/point2d.hpp"
#include <algorithm>
#include <dt/border.hpp>
#include <dt/fill.hpp>
#include <dt/image2d.hpp>

#include <limits>
#include <ranges>
#include <type_traits>

namespace dt
{
  template <typename T, typename O>
  void iterative_distance_transform(const image2d_view<T>& m, const image2d_view<T>& M, image2d_view<O>& D);

  template <typename O, typename T>
  image2d<O> iterative_distance_transform(const image2d_view<T>& m, const image2d_view<T>& M);

  /*
   * Implementations
   */

  namespace details
  {
    class forward_nbh_t
    {
    public:
      constexpr forward_nbh_t() noexcept = default;

      constexpr auto operator()(const point2d& p) const noexcept
      {
        return std::views::transform(offsets, [&p](const auto& o) { return p + o; });
      }

    private:
      static constexpr point2d offsets[] = {
          {1, 0},  //
          {-1, 1}, //
          {0, 1},  //
          {1, 1}   //
      };
    };

    class backward_nbh_t
    {
    public:
      backward_nbh_t() noexcept = default;

      constexpr auto operator()(const point2d& p) const noexcept
      {
        return std::views::transform(offsets, [&p](const auto& o) { return p + o; });
      }

    private:
      static constexpr point2d offsets[] = {
          {-1, 0},  //
          {-1, -1}, //
          {0, -1},  //
          {1, -1}   //
      };
    };

    template <bool Forward, typename T, typename V, typename O>
    bool distance_iter(const image2d_view<T>& m, const image2d_view<T>& M, image2d_view<V>& F,
                       image2d_view<O>& D) noexcept
    {
      using nbh_t = std::conditional_t<Forward, forward_nbh_t, backward_nbh_t>;
      static constexpr nbh_t nbh;

      const int     start_x = Forward ? 1 : m.width() - 2;
      const int     start_y = Forward ? 1 : m.height() - 2;
      const int     end_x   = Forward ? m.width() - 1 : 0;
      const int     end_y   = Forward ? m.height() - 1 : 0;
      constexpr int inc     = Forward ? 1 : -1;

      bool changed = false;

      for (int y = start_y; y != end_y; y += inc)
      {
        for (int x = start_x; x != end_x; x += inc)
        {
          const auto p = point2d(x, y);
          auto       f = F(p);
          auto       d = D(p);
          for (const auto& n : nbh(p))
          {
            const auto q  = std::clamp(F(n), m(p), M(p));
            const auto dn = std::abs(F(n) - q) + D(n);
            if (dn < d)
            {
              d       = dn;
              f       = q;
              changed = true;
            }
          }
          D(p) = d;
          F(p) = f;
        }
      }

      return changed;
    }
  } // namespace details

  template <typename T, typename O>
  void iterative_distance_transform(const image2d_view<T>& m, const image2d_view<T>& M, image2d_view<O>& D)
  {
    using V                      = std::remove_cvref_t<T>;
    static constexpr O UNVISITED = std::numeric_limits<O>::max();

    fill(D, UNVISITED);
    set_border<O>(D, 0);

    image2d<V> F(m.width(), m.height());
    copy_border(m, F);

    bool changed = true;
    while (changed)
    {
      bool c1 = details::distance_iter<true>(m, M, F, D);
      bool c2 = details::distance_iter<false>(m, M, F, D);
      changed = c1 || c2;
    }
  }

  template <typename O, typename T>
  image2d<O> iterative_distance_transform(const image2d_view<T>& m, const image2d_view<T>& M)
  {
    image2d<O> out(m.width(), m.height());
    iterative_distance_transform(m, M, out);
    return out;
  }
} // namespace dt