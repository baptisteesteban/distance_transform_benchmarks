#pragma once

#include <dt/border.hpp>
#include <dt/image2d.hpp>

#include <ranges>
#include <type_traits>

namespace dt
{
  template <typename T, typename O>
  void iterative_distance_transform(const image2d_view<T>& m, const image2d_view<T>& M, image2d_view<O>& D) noexcept;

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

    template <bool forward, typename T, typename V, typename O>
    bool distance_iter(const image2d_view<T>& m, const image2d_view<T>& M, image2d_view<V>& F,
                       image2d_view<O>& D) noexcept
    {
      using nbh_t = std::conditional_t<forward, forward_nbh_t, backward_nbh_t>;
      static constexpr nbh_t nbh;

      bool changed = false;

      // TODO

      return changed;
    }
  } // namespace details

  template <typename T, typename O>
  void iterative_distance_transform(const image2d_view<T>& m, const image2d_view<T>& M, image2d_view<O>& D) noexcept
  {
    using V = std::remove_cvref_t<T>;

    image2d<V> F(m.width(), m.height());
    set_border(D, 0);
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