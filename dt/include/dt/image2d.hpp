#pragma once

#include "dt/details/image2d_data.hpp"
#include <dt/details/image2d_data_cpu.hpp>
#include <dt/image_view.hpp>

#include <memory>

namespace dt
{
  template <typename T>
  class image2d final : public image2d_view<T>
  {
  public:
    image2d(int width, int height);

  private:
    std::shared_ptr<details::image2d_data> m_data;
  };

  /*
   * Implementations
   */

  template <typename T>
  image2d<T>::image2d(int width, int height)
    : dt::image2d_view<T>(nullptr, width, height, 0)
  {
    m_data         = std::make_shared<details::image2d_data_cpu<T>>(width, height);
    this->m_buffer = m_data->buffer();
    this->m_pitch  = m_data->pitch();
  }
} // namespace dt