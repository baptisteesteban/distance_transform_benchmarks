#include "dt/point2d.hpp"
#include <dt/structures/circular_bucket_queue.hpp>

#include <cassert>

namespace dt::structures
{
  CircularBucketQueue::CircularBucketQueue()
    : m_distance(0)
    , m_cur(0)
    , m_size(0)
  {
  }

  void CircularBucketQueue::push(std::uint8_t d, const point2d& p)
  {
    const int ind = (m_cur + d) % 256;
    m_queues[ind].emplace_back(p);
    m_size += 1;
  }

  std::pair<std::uint16_t, point2d> CircularBucketQueue::pop()
  {
    assert(!empty());
    while (m_queues[m_cur].empty())
    {
      m_cur = (m_cur + 1) % 256;
      m_distance += 1;
    }
    auto p = m_queues[m_cur].back();
    m_queues[m_cur].pop_back();
    m_size -= 1;
    return {m_distance, std::move(p)};
  }
} // namespace dt::structures