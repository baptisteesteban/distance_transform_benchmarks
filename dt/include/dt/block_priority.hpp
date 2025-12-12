#pragma once

#include <dt/image2d.hpp>
#include <thrust/device_vector.h>

namespace dt
{
  // Compute per-block Manhattan distance from blocks containing masked pixels
  // Returns a device_vector of distances for each block
  thrust::device_vector<int> compute_block_distances(const dt::image2d_view<std::uint8_t>& mask, int block_size);

  // Compute per-block priorities using empirical CDF of distances
  // Blocks with masked pixels have priority 0
  // K is the maximum priority value (typically 64)
  thrust::device_vector<int> compute_block_priorities(const dt::image2d_view<std::uint8_t>& mask, int block_size,
                                                      int K);

  // Overload that works with grid dimensions directly
  thrust::device_vector<int> compute_block_distances(const dt::image2d_view<std::uint8_t>& mask, int gridDimX,
                                                     int gridDimY, int block_size);

  // Compute priorities from pre-computed distances
  thrust::device_vector<int> compute_priorities_from_distances(const thrust::device_vector<int>& distances, int K);
} // namespace dt
