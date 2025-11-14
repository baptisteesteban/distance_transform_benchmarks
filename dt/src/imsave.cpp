#include <dt/imsave.hpp>

#include <stdexcept>
#include <string_view>

#include "stb_image_write.h"

namespace dt::impl
{
  template <>
  void imsave<std::uint8_t>(const char* filename, const std::uint8_t* buffer, int width, int height, int pitch)
  {
    if (!std::string_view(filename).ends_with(".png"))
      throw std::invalid_argument("Only PNG output is supported");

    stbi_write_png(filename, width, height, 1, buffer, pitch);
  }

  template <>
  void imsave<rgb8>(const char* filename, const rgb8* buffer, int width, int height, int pitch)
  {
    if (!std::string_view(filename).ends_with(".png"))
      throw std::invalid_argument("Only PNG output is supported");

    stbi_write_png(filename, width, height, 3, buffer, pitch);
  }
} // namespace dt::impl