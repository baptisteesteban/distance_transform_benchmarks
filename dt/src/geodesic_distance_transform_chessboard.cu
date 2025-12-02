#include <dt/image2d.hpp>

#include "dt_block_passes.cuh"

namespace dt
{
  void geodesic_distance_transform_chessboard(const image2d_view<std::uint8_t>& img,
                                              const image2d_view<std::uint8_t>& mask, image2d_view<float>& dist,
                                              float v, float lambda)
  {
    /// TODO
  }

  image2d<float> geodesic_distance_transform_chessboard(const image2d_view<std::uint8_t>& img,
                                                        const image2d_view<std::uint8_t>& mask, float v, float lambda)
  {
    assert(img.width() == mask.width() && img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && mask.memory_kind() == e_memory_kind::GPU);
    image2d<float> D(img.width(), img.height(), e_memory_kind::GPU);
    geodesic_distance_transform_chessboard(img, mask, D, v, lambda);
    return D;
  }
} // namespace dt