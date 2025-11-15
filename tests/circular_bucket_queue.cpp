#include <dt/point2d.hpp>
#include <dt/structures/circular_bucket_queue.hpp>

#include <gtest/gtest.h>

TEST(Structures, CircularBucketQueue)
{
  using namespace dt;

  auto q = structures::CircularBucketQueue();
  q.push(3, point2d{0, 0});
  q.push(0, point2d{1, 0});
  ASSERT_EQ(q.size(), 2);

  const auto [d1, p1] = q.pop();
  ASSERT_EQ(d1, 0);
  ASSERT_EQ(p1, point2d(1, 0));
  ASSERT_EQ(q.size(), 1);
  ASSERT_FALSE(q.empty());

  const auto [d2, p2] = q.pop();
  ASSERT_TRUE(q.empty());
  ASSERT_EQ(q.size(), 0);
  ASSERT_EQ(d2, 3);
  ASSERT_EQ(p2, point2d(0, 0));

  q.push(8, point2d{2, 0});
  ASSERT_FALSE(q.empty());
  ASSERT_EQ(q.size(), 1);

  const auto [d3, p3] = q.pop();
  ASSERT_EQ(d3, 11);
  ASSERT_EQ(p3, point2d(2, 0));
  ASSERT_EQ(q.size(), 0);
  ASSERT_TRUE(q.empty());
}