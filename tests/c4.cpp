#include <dt/c4.hpp>

#include <gtest/gtest.h>

TEST(C4, c4)
{
  static constexpr dt::point2d ref_n[] = {
      {2, 7}, //
      {3, 6}, //
      {4, 7}, //
      {3, 8}  //
  };
  const dt::point2d p(3, 7);
  for (const auto& [i, n] : std::views::enumerate(dt::c4(p)))
    ASSERT_EQ(n, ref_n[i]);
}