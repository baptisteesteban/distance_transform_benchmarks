#include <dt/level_lines_distance_transform.hpp>

#include "lldt_block_passes.cuh"
#include "utils.cuh"

namespace dt
{
  void level_lines_distance_transform_task_priority_gpu(const image2d_view<std::uint8_t>& m,
                                                        const image2d_view<std::uint8_t>& M,
                                                        image2d_view<std::uint32_t>&      D)
  {
    // TODO
  }

  image2d<std::uint32_t> level_lines_distance_transform_task_priority_gpu(const image2d_view<std::uint8_t>& m,
                                                                          const image2d_view<std::uint8_t>& M)
  {
    image2d<std::uint32_t> D(m.width(), m.height(), e_memory_kind::GPU);
    level_lines_distance_transform_task_priority_gpu(m, M, D);
    return D;
  }
} // namespace dt