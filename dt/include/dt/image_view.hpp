#pragma once

#include <cstdint>

namespace dt
{
  template <typename T>
  class image2d_view;

  template <>
  class image2d_view<void>
  {
  public:
    // Constructors
    image2d_view() noexcept;
    image2d_view(std::uint8_t* buffer, int width, int height, int pitch) noexcept;
    image2d_view(const image2d_view<void>& other) noexcept;
    image2d_view(image2d_view<void>&& other) noexcept;

    // Assignment operators
    image2d_view<void>& operator=(const image2d_view<void>& other) noexcept;
    image2d_view<void>& operator=(image2d_view<void>&& other) noexcept;

    // Metadata
    int                 width() const noexcept;
    int                 height() const noexcept;
    int                 pitch() const noexcept;
    std::uint8_t*       buffer() noexcept;
    const std::uint8_t* buffer() const noexcept;

  protected:
    std::uint8_t* m_buffer;
    int           m_width;
    int           m_height;
    int           m_pitch;
  };

  /*
   * Implementations
   */

  inline int image2d_view<void>::width() const noexcept
  {
    return m_width;
  }

  inline int image2d_view<void>::height() const noexcept
  {
    return m_height;
  }

  inline int image2d_view<void>::pitch() const noexcept
  {
    return m_pitch;
  }

  inline std::uint8_t* image2d_view<void>::buffer() noexcept
  {
    return m_buffer;
  }

  inline const std::uint8_t* image2d_view<void>::buffer() const noexcept
  {
    return m_buffer;
  }
} // namespace dt