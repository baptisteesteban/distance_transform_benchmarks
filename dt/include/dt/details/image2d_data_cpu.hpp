#pragma once

#include <dt/details/image2d_data.hpp>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace dt::details
{
  template <typename T>
  struct image2d_data_cpu final : public image2d_data<T>
  {
    image2d_data_cpu(int width, int height);
    virtual ~image2d_data_cpu();
  };

  /*
   * Implementations
   */

  template <typename T>
  image2d_data_cpu<T>::image2d_data_cpu(int width, int height)
  {
    std::uint8_t* buffer = reinterpret_cast<std::uint8_t*>(std::malloc(width * height * sizeof(T)));
    if (!buffer)
      throw std::runtime_error("Unable to allocate buffer");
    this->m_buffer = buffer;
    this->m_pitch  = width;
  }

  template <typename T>
  image2d_data_cpu<T>::~image2d_data_cpu()
  {
    if (this->m_buffer)
      std::free(this->m_buffer);
  }
} // namespace dt::details