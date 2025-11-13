#include <dt/image_view.hpp>

#include <utility>

namespace dt
{
  Image2DView<void>::Image2DView() noexcept
    : m_buffer(nullptr)
    , m_width(0)
    , m_height(0)
    , m_pitch(0)
  {
  }

  Image2DView<void>::Image2DView(std::uint8_t* buffer, int width, int height, int pitch) noexcept
    : m_buffer(buffer)
    , m_width(width)
    , m_height(height)
    , m_pitch(pitch)
  {
  }


  Image2DView<void>::Image2DView(const Image2DView<void>& other) noexcept
    : m_buffer(other.m_buffer)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_pitch(other.m_pitch)
  {
  }

  Image2DView<void>::Image2DView(Image2DView<void>&& other) noexcept
    : m_buffer(nullptr)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_pitch(other.m_pitch)
  {
    std::swap(m_buffer, other.m_buffer);
  }

  Image2DView<void>& Image2DView<void>::operator=(const Image2DView<void>& other) noexcept
  {
    m_buffer = other.m_buffer;
    m_width  = other.m_width;
    m_height = other.m_height;
    m_pitch  = other.m_pitch;
    return *this;
  }

  Image2DView<void>& Image2DView<void>::operator=(Image2DView<void>&& other) noexcept
  {
    std::swap(m_buffer, other.m_buffer);
    m_width  = other.m_width;
    m_height = other.m_height;
    m_pitch  = other.m_pitch;
    return *this;
  }
} // namespace dt