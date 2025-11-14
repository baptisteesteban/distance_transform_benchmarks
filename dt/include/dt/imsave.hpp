#pragma once

#include <dt/image_view.hpp>

#include <concepts>
#include <type_traits>

namespace dt
{
  namespace impl
  {
    template <typename T>
    void imsave(const char* filename, const T* buffer, int width, int height, int pitch);
  }

  template <typename T>
  void imsave(const char* filename, const image2d_view<T>& img);

  /*
   * Specialization
   */

  template <typename T>
    requires(std::same_as<std::remove_cvref_t<T>, std::uint8_t>)
  void imsave(const char* filename, const image2d_view<T>& img)
  {
    impl::imsave(filename, img.buffer(), img.width(), img.height(), img.pitch());
  }
} // namespace dt