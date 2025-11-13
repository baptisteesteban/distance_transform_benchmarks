#pragma once

#include <cstdint>

namespace dt::details
{
  class image2d_data
  {
  public:
    image2d_data(const image2d_data&)            = delete;
    image2d_data(image2d_data&&)                 = delete;
    image2d_data& operator=(const image2d_data&) = delete;
    image2d_data& operator=(image2d_data&&)      = delete;

    std::uint8_t*       buffer() noexcept;
    const std::uint8_t* buffer() const noexcept;
    int                 pitch() const noexcept;

  protected:
    image2d_data();

  protected:
    std::uint8_t* m_buffer;
    int           m_pitch;
  };

  /*
   * Implementations
   */

  inline std::uint8_t* image2d_data::buffer() noexcept
  {
    return m_buffer;
  }

  inline const std::uint8_t* image2d_data::buffer() const noexcept
  {
    return m_buffer;
  }

  inline int image2d_data::pitch() const noexcept
  {
    return m_pitch;
  }
} // namespace dt::details