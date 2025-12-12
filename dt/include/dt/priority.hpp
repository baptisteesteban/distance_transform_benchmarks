#pragma once

#include <algorithm>

namespace dt
{
  class manhattan_distance_object
  {
  public:
    constexpr manhattan_distance_object(int cx, int cy)
      : m_cx(cx)
      , m_cy(cy)
    {
    }

    constexpr int distanceof(int x, int y, int /*width*/, int /*height*/) const
    {
      int dx = x > m_cx ? x - m_cx : m_cx - x;
      int dy = y > m_cy ? y - m_cy : m_cy - y;
      return dx + dy;
    }

    constexpr int distanceCDF(int d, int width, int height) const
    {
      if (d < 0)
        return 0;

      int count = 0;
      int x0    = std::max(m_cx - d, 0);
      int x1    = std::min(m_cx + d, width - 1);

      for (int x = x0; x <= x1; ++x)
      {
        int dx  = x > m_cx ? x - m_cx : m_cx - x;
        int rem = d - dx;
        int y0  = m_cy - rem;
        if (y0 < 0)
          y0 = 0;
        int y1 = m_cy + rem;
        if (y1 >= height)
          y1 = height - 1;
        if (y1 >= y0)
          count += (y1 - y0 + 1);
      }

      return count;
    }

    constexpr int priorityof(int x, int y, int width, int height, int K) const
    {
      int d     = distanceof(x, y, width, height);
      int count = distanceCDF(d, width, height);
      int total = width * height + 1;
      return count * K / total;
    }

  private:
    int m_cx;
    int m_cy;
  };
} // namespace dt