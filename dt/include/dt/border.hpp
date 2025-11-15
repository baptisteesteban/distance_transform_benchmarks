#pragma once

#include <dt/image2d.hpp>

#include <algorithm>
#include <cassert>
#include <memory>
#include <type_traits>

namespace dt
{
  template <typename T>
  void set_border(image2d_view<T>& img, T val) noexcept;

  template <typename T, typename O = std::remove_cvref_t<T>>
  void copy_border(const image2d_view<T>& img, image2d_view<O>& out) noexcept;

  template <typename T, typename O = std::remove_cvref_t<T>>
  void add_border(const image2d_view<T>& img, image2d_view<O>& out, const T val) noexcept;

  template <typename T, typename O = std::remove_cvref_t<T>>
  image2d<O> add_border(const image2d<T>& img, const T val);

  template <typename T, typename O = std::remove_cvref_t<T>>
  void add_median_border(const image2d_view<T>& img, image2d_view<O>& out) noexcept;

  template <typename T, typename O = std::remove_cvref_t<T>>
  image2d<O> add_median_border(const image2d<T>& img);

  /*
   * Implementations
   */

  namespace details
  {
    template <typename T>
    T get_median_border_value(const image2d_view<T>& img) noexcept
    {
      const int N             = 2 * (img.width() + img.height() - 2);
      auto      border_values = std::make_unique_for_overwrite<std::remove_cvref_t<T>[]>(N);
      int       i             = 0;
      for (int x = 0; x < img.width(); x++)
      {
        border_values[i++] = img(x, 0);
        border_values[i++] = img(x, img.height() - 1);
      }
      for (int y = 1; y < img.height() - 1; y++)
      {
        border_values[i++] = img(0, y);
        border_values[i++] = img(img.width() - 1, y);
      }
      std::sort(border_values.get(), border_values.get() + N);
      return border_values[(N - 1) / 2];
    }
  } // namespace details

  template <typename T>
  void set_border(image2d_view<T>& img, T val) noexcept
  {
    const int hm = img.height() - 1;
    const int wm = img.width() - 1;

    for (int x = 0; x < img.width(); x++)
    {
      img(x, 0)  = val;
      img(x, hm) = val;
    }
    for (int y = 1; y < img.height() - 1; y++)
    {
      img(0, y)  = val;
      img(wm, y) = val;
    }
  }

  template <typename T, typename O>
  void copy_border(const image2d_view<T>& img, image2d_view<O>& out) noexcept
  {
    assert(img.width() == out.width() && img.height() == out.height());

    const int hm = img.height() - 1;
    const int wm = img.width() - 1;

    for (int x = 0; x < img.width(); x++)
    {
      out(x, 0)  = img(x, 0);
      out(x, hm) = img(x, hm);
    }
    for (int y = 1; y < img.height() - 1; y++)
    {
      out(0, y)  = img(0, y);
      out(wm, y) = img(wm, y);
    }
  }

  template <typename T, typename O>
  void add_border(const image2d_view<T>& img, image2d_view<O>& out, const T val) noexcept
  {
    assert(out.width() == img.width() + 2 && out.height() == img.height() + 2);
    // Original image data
    for (int y = 0; y < img.height(); y++)
    {
      for (int x = 0; x < img.width(); x++)
        out(x + 1, y + 1) = img(x, y);
    }

    // Border
    set_border(out, val);
  }

  template <typename T, typename O>
  image2d<O> add_border(const image2d<T>& img, const T val)
  {
    image2d<O> out(img.width() + 2, img.height() + 2);
    add_border<T, O>(img, out, val);
    return out;
  }

  template <typename T, typename O>
  void add_median_border(const image2d_view<T>& img, image2d_view<O>& out) noexcept
  {
    const auto v = details::get_median_border_value(img);
    add_border<T, O>(img, out, v);
  }

  template <typename T, typename O>
  image2d<O> add_median_border(const image2d<T>& img)
  {
    const auto v = details::get_median_border_value(img);
    return add_border<T, O>(img, v);
  }
} // namespace dt