#pragma once

#include <dt/point2d.hpp>

#include <ranges>

namespace dt
{
  class c8_t
  {
  public:
    constexpr c8_t() noexcept = default;
    auto operator()(const point2d& p) const noexcept
    {
      return std::views::transform(offsets, [p](const auto& o) { return p + o; });
    }

  private:
    static constexpr point2d offsets[] = {
        {-1, 0},  //
        {-1, -1}, //
        {0, -1},  //
        {1, -1},  //
        {1, 0},   //
        {1, 1},   //
        {0, 1},   //
        {-1, 1}   //
    };
  };

  static constexpr c8_t c8 = {};
} // namespace dt