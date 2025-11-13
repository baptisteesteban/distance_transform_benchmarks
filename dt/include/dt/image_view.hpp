#pragma once

#include <cstdint>

namespace dt
{
  template <typename T>
  class Image2DView;

  template <>
  class Image2DView<void>
  {
  public:
    // Constructors
    Image2DView() noexcept;
    Image2DView(std::uint8_t* buffer, int width, int height, int pitch) noexcept;
    Image2DView(const Image2DView<void>& other) noexcept;
    Image2DView(Image2DView<void>&& other) noexcept;

    // Assignment operators
    Image2DView<void>& operator=(const Image2DView<void>& other) noexcept;
    Image2DView<void>& operator=(Image2DView<void>&& other) noexcept;


  protected:
    std::uint8_t* m_buffer;
    int           m_width;
    int           m_height;
    int           m_pitch;
  };
} // namespace dt