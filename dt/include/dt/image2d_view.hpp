#pragma once

#include <dt/point2d.hpp>

#include <cassert>
#include <cstdint>
#include <utility>

namespace dt
{
  enum class e_memory_kind
  {
    CPU,
    GPU
  };

  template <typename T>
  class image2d_view;

  template <>
  class image2d_view<void>
  {
  public:
    // Constructors
    image2d_view() noexcept;
    image2d_view(std::uint8_t* buffer, int width, int height, int pitch, int elem_size,
                 e_memory_kind memory_kind = e_memory_kind::CPU) noexcept;
    image2d_view(const image2d_view<void>& other) noexcept;
    image2d_view(image2d_view<void>&& other) noexcept;

    // Assignment operators
    image2d_view<void>& operator=(const image2d_view<void>& other) noexcept;
    image2d_view<void>& operator=(image2d_view<void>&& other) noexcept;

// Metadata
#ifdef __CUDACC__
    __host__ __device__
#endif
        int
        width() const noexcept;
#ifdef __CUDACC__
    __host__ __device__
#endif
        int
        height() const noexcept;
#ifdef __CUDACC__
    __host__ __device__
#endif
        int
                        pitch() const noexcept;
    int                 elem_size() const noexcept;
    std::uint8_t*       buffer() noexcept;
    const std::uint8_t* buffer() const noexcept;
    e_memory_kind       memory_kind() const noexcept;

    // Check
    bool valid() const noexcept;
#ifdef __CUDACC__
    __host__ __device__
#endif
        bool
         in_domain(int x, int y) const noexcept;
    bool in_domain(const point2d& p) const noexcept;

// Accessor
#ifdef __CUDACC__
    __host__ __device__
#endif
        std::uint8_t*
        operator()(int x, int y) noexcept;
#ifdef __CUDACC__
    __host__ __device__
#endif
        const           std::uint8_t*
                        operator()(int x, int y) const noexcept;
    std::uint8_t*       operator()(const point2d& p) noexcept;
    const std::uint8_t* operator()(const point2d& p) const noexcept;

  protected:
    std::uint8_t* m_buffer;
    int           m_width;
    int           m_height;
    int           m_pitch;
    int           m_elem_size;
    e_memory_kind m_memory_kind;
  };

  template <typename T>
  class image2d_view : public image2d_view<void>
  {
  public:
    // Constructors
    image2d_view() noexcept;
    image2d_view(T* buffer, int width, int height, int pitch, e_memory_kind memory_kind = e_memory_kind::CPU) noexcept;
    image2d_view(const image2d_view<T>& other) noexcept;
    image2d_view(image2d_view<T>&& other) noexcept;

    // Assignment operators
    image2d_view<T>& operator=(const image2d_view<T>& other) noexcept;
    image2d_view<T>& operator=(image2d_view<T>&& other) noexcept;

// Accessors
#ifdef __CUDACC__
    __host__ __device__
#endif
        const T&
        operator()(int x, int y) const noexcept;
#ifdef __CUDACC__
    __host__ __device__
#endif
        T&
             operator()(int x, int y) noexcept;
    const T& operator()(const point2d& p) const noexcept;
    T&       operator()(const point2d& p) noexcept;
  };

  /*
   * Implementations
   */
#ifdef __CUDACC__
  __host__ __device__
#endif
      inline int
      image2d_view<void>::width() const noexcept
  {
    return m_width;
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
      inline int
      image2d_view<void>::height() const noexcept
  {
    return m_height;
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
      inline int
      image2d_view<void>::pitch() const noexcept
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
#ifdef __CUDACC__
  __host__ __device__
#endif
      inline bool
      image2d_view<void>::in_domain(int x, int y) const noexcept
  {
    return x >= 0 && y >= 0 && x < m_width && y < m_height;
  }

  inline bool image2d_view<void>::in_domain(const point2d& p) const noexcept
  {
    return in_domain(p.x(), p.y());
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
      inline std::uint8_t*
      image2d_view<void>::operator()(int x, int y) noexcept
  {
    assert(in_domain(x, y) && m_buffer);
    return m_buffer + (y * m_pitch + x * m_elem_size);
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
      inline const std::uint8_t*
      image2d_view<void>::operator()(int x, int y) const noexcept
  {
    assert(in_domain(x, y) && m_buffer);
    return m_buffer + (y * m_pitch + x * m_elem_size);
  }

  inline std::uint8_t* image2d_view<void>::operator()(const point2d& p) noexcept
  {
    assert(in_domain(p));
    return (*this)(p.x(), p.y());
  }

  inline const std::uint8_t* image2d_view<void>::operator()(const point2d& p) const noexcept
  {
    assert(in_domain(p));
    return (*this)(p.x(), p.y());
  }

  inline e_memory_kind image2d_view<void>::memory_kind() const noexcept
  {
    return m_memory_kind;
  }

  template <typename T>
  image2d_view<T>::image2d_view() noexcept
    : image2d_view<void>()
  {
  }

  template <typename T>
  image2d_view<T>::image2d_view(T* buffer, int width, int height, int pitch, e_memory_kind memory_kind) noexcept
    : image2d_view<void>((std::uint8_t*)buffer, width, height, pitch, sizeof(T), memory_kind)
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
#ifdef __CUDACC__
  __host__ __device__
#endif
      const T&
      image2d_view<T>::operator()(int x, int y) const noexcept
  {
    assert(in_domain(x, y) && m_buffer && m_memory_kind == e_memory_kind::CPU);
    auto p = static_cast<image2d_view<void>>(*this)(x, y);
    return *reinterpret_cast<const T*>(p);
  }

  template <typename T>
#ifdef __CUDACC__
  __host__ __device__
#endif
      T&
      image2d_view<T>::operator()(int x, int y) noexcept
  {
    assert(in_domain(x, y) && m_buffer && m_memory_kind == e_memory_kind::CPU);
    auto p = static_cast<image2d_view<void>>(*this)(x, y);
    return *reinterpret_cast<T*>(p);
  }

  template <typename T>
  const T& image2d_view<T>::operator()(const point2d& p) const noexcept
  {
    return (*this)(p.x(), p.y());
  }

  template <typename T>
  T& image2d_view<T>::operator()(const point2d& p) noexcept
  {
    return (*this)(p.x(), p.y());
  }
} // namespace dt