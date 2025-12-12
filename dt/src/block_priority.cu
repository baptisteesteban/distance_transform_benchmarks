#include <dt/block_priority.hpp>
#include <dt/transfert.hpp>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <climits>
#include <vector>

namespace
{
  __global__ void mask_has_pixel(const dt::image2d_view<std::uint8_t>& mask, std::uint8_t* out)
  {
    const int      bx = blockIdx.x;
    const int      by = blockIdx.y;
    const int      x  = bx * blockDim.x + threadIdx.x;
    const int      y  = by * blockDim.y + threadIdx.y;
    __shared__ int any;
    if (threadIdx.x == 0 && threadIdx.y == 0)
      any = 0;
    __syncthreads();

    if (x < mask.width() && y < mask.height())
    {
      if (mask(x, y) > 0)
        atomicExch(&any, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
      out[by * gridDim.x + bx] = static_cast<std::uint8_t>(any);
  }

  __global__ void block_min_distance(const int* sx, const int* sy, int nsrc, int* out)
  {
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int bid = by * gridDim.x + bx;

    int best = INT_MAX;
    for (int i = 0; i < nsrc; ++i)
    {
      const int d = abs(bx - sx[i]) + abs(by - sy[i]);
      best        = std::min(best, d);
    }
    out[bid] = best;
  }
} // namespace

namespace dt
{
  thrust::device_vector<int> compute_block_distances(const dt::image2d_view<std::uint8_t>& mask, int gridDimX,
                                                     int gridDimY, int block_size)
  {
    const auto d_mask = mask;
    dim3       gridDim(gridDimX, gridDimY);
    dim3       blockDim(block_size, block_size);
    const int  size = gridDimX * gridDimY;
    const int  gx   = gridDimX;

    // Detect which blocks have masked pixels
    thrust::device_vector<std::uint8_t> d_has(size);
    mask_has_pixel<<<gridDim, blockDim>>>(d_mask, thrust::raw_pointer_cast(d_has.data()));
    cudaDeviceSynchronize();

    // Extract source coordinates on GPU using thrust::copy_if
    thrust::device_vector<int> d_indices(size);
    thrust::device_vector<int> d_sx, d_sy;

    // Generate indices [0, size)
    thrust::sequence(d_indices.begin(), d_indices.end());

    // Extract indices where has_pixel is true
    thrust::device_vector<int> d_src_indices(size);
    auto end_it = thrust::copy_if(d_indices.begin(), d_indices.end(), d_has.begin(), d_src_indices.begin(),
                                  thrust::identity<std::uint8_t>());
    d_src_indices.resize(thrust::distance(d_src_indices.begin(), end_it));

    const int nsrc = d_src_indices.size();

    // Convert indices to x,y coordinates (copy to host for simplicity)
    std::vector<int> h_src_indices(nsrc);
    thrust::copy(d_src_indices.begin(), d_src_indices.end(), h_src_indices.begin());

    std::vector<int> h_sx(nsrc), h_sy(nsrc);
    for (int i = 0; i < nsrc; ++i)
    {
      h_sx[i] = h_src_indices[i] % gx;
      h_sy[i] = h_src_indices[i] / gx;
    }

    d_sx = thrust::device_vector<int>(h_sx.begin(), h_sx.end());
    d_sy = thrust::device_vector<int>(h_sy.begin(), h_sy.end());

    // Compute distances on GPU
    thrust::device_vector<int> d_dist(size);
    block_min_distance<<<gridDim, dim3(1, 1)>>>(thrust::raw_pointer_cast(d_sx.data()),
                                                thrust::raw_pointer_cast(d_sy.data()), nsrc,
                                                thrust::raw_pointer_cast(d_dist.data()));
    cudaDeviceSynchronize();

    return d_dist;
  }

  thrust::device_vector<int> compute_block_distances(const dt::image2d_view<std::uint8_t>& mask, int block_size)
  {
    int gridDimX = (mask.width() + block_size - 1) / block_size;
    int gridDimY = (mask.height() + block_size - 1) / block_size;
    return compute_block_distances(mask, gridDimX, gridDimY, block_size);
  }

  thrust::device_vector<int> compute_priorities_from_distances(const thrust::device_vector<int>& distances, int K)
  {
    const int size = distances.size();
    if (size == 0)
      return {};

    // Do CDF computation on CPU to avoid complex GPU template issues
    std::vector<int> h_dists(size);
    thrust::copy(distances.begin(), distances.end(), h_dists.begin());

    int maxd = *thrust::max_element(distances.begin(), distances.end());

    std::vector<int> freq(maxd + 1, 0);
    for (int v : h_dists)
    {
      if (v >= 0)
        ++freq[v];
    }

    std::vector<int> cum(maxd + 1);
    thrust::inclusive_scan(freq.begin(), freq.end(), cum.begin());

    const int total = size + 1;

    std::vector<int> h_priority(size);
    for (int i = 0; i < size; ++i)
    {
      int d = h_dists[i];
      if (d <= 0)
        h_priority[i] = 0;
      else
      {
        int count     = (d <= maxd) ? cum[d] : cum.back();
        h_priority[i] = count * K / total;
      }
    }

    // Return as device_vector
    return thrust::device_vector<int>(h_priority.begin(), h_priority.end());
  }

  thrust::device_vector<int> compute_block_priorities(const dt::image2d_view<std::uint8_t>& mask, int block_size, int K)
  {
    auto d_dists = compute_block_distances(mask, block_size);
    return compute_priorities_from_distances(d_dists, K);
  }
} // namespace dt
