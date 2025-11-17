#pragma once

#include <dt/details/image2d_data.hpp>

#include <cuda_runtime.h>

#include <format>
#include <stdexcept>

namespace dt::details
{
  template <typename T>
  struct image2d_data_gpu final : public image2d_data
  {
    image2d_data_gpu(int width, int height);
    virtual ~image2d_data_gpu();
  };

  /*
   * Implementations
   */

  template <typename T>
  image2d_data_gpu<T>::image2d_data_gpu(int width, int height)
  {
    std::uint8_t* buffer;
    std::size_t   pitch;
    auto          err = cudaMallocPitch(&buffer, &pitch, width * sizeof(T), height);
    if (err != cudaSuccess)
      throw std::runtime_error(std::format("Unable to allocate GPU memory: {}", cudaGetErrorString(err)));
    this->m_buffer = buffer;
    this->m_pitch  = pitch;
  }

  template <typename T>
  image2d_data_gpu<T>::~image2d_data_gpu()
  {
    if (this->m_buffer)
      cudaFree(this->m_buffer);
  }
} // namespace dt::details