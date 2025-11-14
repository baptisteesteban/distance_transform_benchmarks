#pragma once

#include "dt/point2d.hpp"
#include <cstdint>
#include <vector>

namespace dt::structures
{
  class CircularBucketQueue
  {
  public:
    CircularBucketQueue();
    CircularBucketQueue(const CircularBucketQueue&)            = delete;
    CircularBucketQueue(CircularBucketQueue&&)                 = delete;
    CircularBucketQueue& operator=(const CircularBucketQueue&) = delete;
    CircularBucketQueue& operator=(CircularBucketQueue&&)      = delete;

    void                              push(std::uint8_t d, int x, int y);
    std::pair<std::uint16_t, point2d> pop();
    int                               size() const noexcept;
    bool                              empty() const noexcept;

  private:
    std::uint16_t        m_distance;
    int                  m_cur;
    int                  m_size;
    std::vector<point2d> m_queues[256];
  };

  /*
   * Implementations
   */

  inline int CircularBucketQueue::size() const noexcept
  {
    return m_size;
  }

  inline bool CircularBucketQueue::empty() const noexcept
  {
    return m_size == 0;
  }
} // namespace dt::structures