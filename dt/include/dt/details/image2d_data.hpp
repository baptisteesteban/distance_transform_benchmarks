#pragma once

namespace dt::details
{
  template <typename T>
  class image2d_data
  {
  public:
    image2d_data(const image2d_data&)            = delete;
    image2d_data(image2d_data&&)                 = delete;
    image2d_data& operator=(const image2d_data&) = delete;
    image2d_data& operator=(image2d_data&&)      = delete;

    T*       buffer() noexcept;
    const T* buffer() const noexcept;
    int      pitch() const noexcept;

  protected:
    image2d_data();

  protected:
    T*  m_buffer;
    int m_pitch;
  };

  /*
   * Implementations
   */

  template <typename T>
  image2d_data<T>::image2d_data()
    : m_buffer(nullptr)
    , m_pitch(0)
  {
  }

  template <typename T>
  inline T* image2d_data<T>::buffer() noexcept
  {
    return m_buffer;
  }

  template <typename T>
  inline const T* image2d_data<T>::buffer() const noexcept
  {
    return m_buffer;
  }

  template <typename T>
  inline int image2d_data<T>::pitch() const noexcept
  {
    return m_pitch;
  }
} // namespace dt::details