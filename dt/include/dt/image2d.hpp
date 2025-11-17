#pragma once

#include "dt/details/image2d_data_gpu.hpp"
#include <dt/details/image2d_data.hpp>
#include <dt/details/image2d_data_cpu.hpp>
#include <dt/image2d_view.hpp>

#include <memory>
#include <stdexcept>

namespace dt
{
  template <typename T>
  class image2d final : public image2d_view<T>
  {
  public:
    image2d(int width, int height, e_memory_kind memory_kind = e_memory_kind::CPU);

  private:
    std::shared_ptr<details::image2d_data> m_data;
  };

  /*
   * Implementations
   */

  template <typename T>
  image2d<T>::image2d(int width, int height, e_memory_kind memory_kind)
    : dt::image2d_view<T>(nullptr, width, height, 0, memory_kind)
  {
    switch (memory_kind)
    {
    case e_memory_kind::CPU:
      m_data = std::make_shared<details::image2d_data_cpu<T>>(width, height);
      break;
    case e_memory_kind::GPU:
      m_data = std::make_shared<details::image2d_data_gpu<T>>(width, height);
      break;
    default:
      throw std::invalid_argument("Invalid memory kind for image2d allocation");
    }
    this->m_buffer = m_data->buffer();
    this->m_pitch  = m_data->pitch();
  }
} // namespace dt