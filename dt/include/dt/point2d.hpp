#pragma once

namespace dt
{
  class point2d
  {
  public:
    // Default constructors
    constexpr point2d() noexcept                          = default;
    constexpr point2d(const point2d&) noexcept            = default;
    constexpr point2d(point2d&&) noexcept                 = default;
    constexpr point2d& operator=(const point2d&) noexcept = default;
    constexpr point2d& operator=(point2d&&) noexcept      = default;

    // Constructor
    constexpr point2d(int x, int y) noexcept;

    // Accessors
    constexpr int&      x() noexcept;
    constexpr const int x() const noexcept;
    constexpr int&      y() noexcept;
    constexpr const int y() const noexcept;

    // Comparison
    constexpr bool operator==(const point2d&) const noexcept;
    constexpr bool operator!=(const point2d&) const noexcept;

  private:
    int m_x = 0;
    int m_y = 0;
  };

  // Operators
  constexpr point2d operator+(const point2d& p1, const point2d& p2) noexcept;
  constexpr point2d operator-(const point2d& p1, const point2d& p2) noexcept;
  constexpr point2d operator*(const point2d& p, int v) noexcept;
  constexpr point2d operator*(int v, const point2d& p) noexcept;

  /*
   * Implementations
   */

  constexpr point2d::point2d(int x, int y) noexcept
    : m_x(x)
    , m_y(y)
  {
  }

  inline constexpr int& point2d::x() noexcept
  {
    return m_x;
  }

  inline constexpr const int point2d::x() const noexcept
  {
    return m_x;
  }

  inline constexpr int& point2d::y() noexcept
  {
    return m_y;
  }

  inline constexpr const int point2d::y() const noexcept
  {
    return m_y;
  }


  inline constexpr bool point2d::operator==(const point2d& other) const noexcept
  {
    return m_x == other.m_x && m_y == other.m_y;
  }

  inline constexpr bool point2d::operator!=(const point2d& other) const noexcept
  {
    return !(*this == other);
  }

  inline constexpr point2d operator+(const point2d& p1, const point2d& p2) noexcept
  {
    return {p1.x() + p2.x(), p1.y() + p2.y()};
  }

  inline constexpr point2d operator-(const point2d& p1, const point2d& p2) noexcept
  {
    return {p1.x() - p2.x(), p1.y() - p2.y()};
  }

  inline constexpr point2d operator*(const point2d& p, int v) noexcept
  {
    return {p.x() * v, p.y() * v};
  }

  inline constexpr point2d operator*(int v, const point2d& p) noexcept
  {
    return p * v;
  }
} // namespace dt