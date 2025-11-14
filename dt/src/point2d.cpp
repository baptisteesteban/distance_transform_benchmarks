#include <dt/point2d.hpp>

namespace dt
{
  point2d::point2d(int x, int y) noexcept
    : m_x(x)
    , m_y(y)
  {
  }


  bool point2d::operator==(const point2d& other) const noexcept
  {
    return m_x == other.m_x && m_y == other.m_y;
  }
  bool point2d::operator!=(const point2d& other) const noexcept
  {
    return !(*this == other);
  }
} // namespace dt