#pragma once

#include <dt/image2d.hpp>

#include <cassert>

namespace dt
{
  void host_to_device(const image2d_view<void>& src, image2d_view<void>& dst);

  void device_to_host(const image2d_view<void>& src, image2d_view<void>& dst);

  template <typename T>
  image2d<T> host_to_device(const image2d_view<T>& src);

  template <typename T>
  image2d<T> device_to_host(const image2d_view<T>& src);

  /*
   * Implementations
   */

  template <typename T>
  image2d<T> host_to_device(const image2d_view<T>& src)
  {
    assert(src.memory_kind() == e_memory_kind::CPU);
    image2d<T> dst(src.width(), src.height(), e_memory_kind::GPU);
    host_to_device(src, dst);
    return dst;
  }

  template <typename T>
  image2d<T> device_to_host(const image2d_view<T>& src)
  {
    assert(src.memory_kind() == e_memory_kind::GPU);
    image2d<T> dst(src.width(), src.height(), e_memory_kind::CPU);
    device_to_host(src, dst);
    return dst;
  }
} // namespace dt