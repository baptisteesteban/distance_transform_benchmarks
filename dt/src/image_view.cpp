#include <dt/image_view.hpp>

#include <utility>

namespace dt
{
  image2d_view<void>::image2d_view() noexcept
    : m_buffer(nullptr)
    , m_width(0)
    , m_height(0)
    , m_pitch(0)
    , m_elem_size(0)
  {
  }

  image2d_view<void>::image2d_view(std::uint8_t* buffer, int width, int height, int pitch, int elem_size) noexcept
    : m_buffer(buffer)
    , m_width(width)
    , m_height(height)
    , m_pitch(pitch)
    , m_elem_size(elem_size)
  {
  }


  image2d_view<void>::image2d_view(const image2d_view<void>& other) noexcept
    : m_buffer(other.m_buffer)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_pitch(other.m_pitch)
    , m_elem_size(other.m_elem_size)
  {
  }

  image2d_view<void>::image2d_view(image2d_view<void>&& other) noexcept
    : m_buffer(nullptr)
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_pitch(other.m_pitch)
    , m_elem_size(other.m_elem_size)
  {
    std::swap(m_buffer, other.m_buffer);
  }

  image2d_view<void>& image2d_view<void>::operator=(const image2d_view<void>& other) noexcept
  {
    m_buffer    = other.m_buffer;
    m_width     = other.m_width;
    m_height    = other.m_height;
    m_pitch     = other.m_pitch;
    m_elem_size = other.m_elem_size;
    return *this;
  }

  image2d_view<void>& image2d_view<void>::operator=(image2d_view<void>&& other) noexcept
  {
    std::swap(m_buffer, other.m_buffer);
    m_width     = other.m_width;
    m_height    = other.m_height;
    m_pitch     = other.m_pitch;
    m_elem_size = other.m_elem_size;
    return *this;
  }
} // namespace dt