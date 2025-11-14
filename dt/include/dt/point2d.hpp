#pragma once

namespace dt
{
  class point2d
  {
  public:
    point2d()                          = default;
    point2d(const point2d&)            = default;
    point2d(point2d&&)                 = default;
    point2d& operator=(const point2d&) = default;
    point2d& operator=(point2d&&)      = default;

    point2d(int x, int y) noexcept;

    int&      x() noexcept;
    const int x() const noexcept;
    int&      y() noexcept;
    const int y() const noexcept;

    bool operator==(const point2d&) const noexcept;
    bool operator!=(const point2d&) const noexcept;

  private:
    int m_x = 0;
    int m_y = 0;
  };

  inline int& point2d::x() noexcept
  {
    return m_x;
  }

  inline const int point2d::x() const noexcept
  {
    return m_x;
  }

  inline int& point2d::y() noexcept
  {
    return m_y;
  }

  inline const int point2d::y() const noexcept
  {
    return m_y;
  }
} // namespace dt