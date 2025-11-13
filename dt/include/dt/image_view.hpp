#pragma once

#include <cassert>
#include <cstdint>
#include <utility>

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
    image2d_view(std::uint8_t* buffer, int width, int height, int pitch, int elem_size) noexcept;
    image2d_view(const image2d_view<void>& other) noexcept;
    image2d_view(image2d_view<void>&& other) noexcept;

    // Assignment operators
    image2d_view<void>& operator=(const image2d_view<void>& other) noexcept;
    image2d_view<void>& operator=(image2d_view<void>&& other) noexcept;

    // Metadata
    int                 width() const noexcept;
    int                 height() const noexcept;
    int                 pitch() const noexcept;
    int                 elem_size() const noexcept;
    std::uint8_t*       buffer() noexcept;
    const std::uint8_t* buffer() const noexcept;

    // Check
    bool valid() const noexcept;
    bool in_domain(int x, int y) const noexcept;

    // Accessor
    std::uint8_t*       operator()(int x, int y) noexcept;
    const std::uint8_t* operator()(int x, int y) const noexcept;

  protected:
    std::uint8_t* m_buffer;
    int           m_width;
    int           m_height;
    int           m_pitch;
    int           m_elem_size;
  };

  template <typename T>
  class image2d_view : public image2d_view<void>
  {
  public:
    // Constructors
    image2d_view() noexcept;
    image2d_view(T* buffer, int width, int height, int pitch) noexcept;
    image2d_view(const image2d_view<T>& other) noexcept;
    image2d_view(image2d_view<T>&& other) noexcept;

    // Assignment operators
    image2d_view<T>& operator=(const image2d_view<T>& other) noexcept;
    image2d_view<T>& operator=(image2d_view<T>&& other) noexcept;

    // Accessors
    const T& operator()(int x, int y) const noexcept;
    T&       operator()(int x, int y) noexcept;
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

  inline int image2d_view<void>::elem_size() const noexcept
  {
    return m_elem_size;
  }

  inline std::uint8_t* image2d_view<void>::buffer() noexcept
  {
    return m_buffer;
  }

  inline const std::uint8_t* image2d_view<void>::buffer() const noexcept
  {
    return m_buffer;
  }

  inline bool image2d_view<void>::valid() const noexcept
  {
    return m_buffer != nullptr;
  }

  inline bool image2d_view<void>::in_domain(int x, int y) const noexcept
  {
    return x >= 0 && y >= 0 && x < m_width && y < m_height;
  }

  inline std::uint8_t* image2d_view<void>::operator()(int x, int y) noexcept
  {
    assert(in_domain(x, y) && m_buffer);
    return m_buffer + (y * m_pitch + x * m_elem_size);
  }

  inline const std::uint8_t* image2d_view<void>::operator()(int x, int y) const noexcept
  {
    assert(in_domain(x, y) && m_buffer);
    return m_buffer + (y * m_pitch + x * m_elem_size);
  }


  template <typename T>
  image2d_view<T>::image2d_view() noexcept
    : image2d_view<void>()
  {
  }

  template <typename T>
  image2d_view<T>::image2d_view(T* buffer, int width, int height, int pitch) noexcept
    : image2d_view<void>(reinterpret_cast<std::uint8_t*>(buffer), width, height, pitch, sizeof(T))
  {
  }

  template <typename T>
  image2d_view<T>::image2d_view(const image2d_view<T>& other) noexcept
    : image2d_view<void>(other)
  {
  }

  template <typename T>
  image2d_view<T>::image2d_view(image2d_view<T>&& other) noexcept
    : image2d_view<void>(other)
  {
  }

  template <typename T>
  image2d_view<T>& image2d_view<T>::operator=(const image2d_view<T>& other) noexcept
  {
    *this = other;
    return *this;
  }

  template <typename T>
  image2d_view<T>& image2d_view<T>::operator=(image2d_view<T>&& other) noexcept
  {
    *this = std::move(other);
    return *this;
  }


  template <typename T>
  const T& image2d_view<T>::operator()(int x, int y) const noexcept
  {
    assert(in_domain(x, y) && m_buffer);
    auto p = static_cast<image2d_view<void>>(*this)(x, y);
    return *reinterpret_cast<T*>(p);
  }

  template <typename T>
  T& image2d_view<T>::operator()(int x, int y) noexcept
  {
    assert(in_domain(x, y) && m_buffer);
    auto p = static_cast<image2d_view<void>>(*this)(x, y);
    return *reinterpret_cast<T*>(p);
  }
} // namespace dt