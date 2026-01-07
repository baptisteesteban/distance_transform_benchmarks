#include <dt/image2d.hpp>

#include "dt_block_passes.cuh"
#include "utils.cuh"

namespace dt
{
  __global__ static void block_propagation(const image2d_view<std::uint8_t>& img, image2d_view<float>& D, float v,
                                           float l_eucl, float l_grad, bool even, bool* active, bool* active_candidate,
                                           bool* changed)
  {
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int bid = by * gridDim.x + bx;

    if (!active[bid])
      return;

    const int x = bx * BLOCK_SIZE;
    const int y = by * BLOCK_SIZE + threadIdx.x;

    // Loading tile into shared memory
    __shared__ std::uint8_t s_img[TILE_SIZE][TILE_SIZE];
    __shared__ float        s_D[TILE_SIZE][TILE_SIZE];

    {
      // tx and ty are the tile indices
      for (int tx = 0; tx < TILE_SIZE; ++tx)
      {
        for (int dty = -BLOCK_SIZE; dty <= BLOCK_SIZE; dty += BLOCK_SIZE)
        {
          const int ty = threadIdx.x + dty + 1;
          if (ty < 0 || ty >= TILE_SIZE)
            continue;

          const int gx = x + tx - 1;
          const int gy = y + dty;

          bool valid    = gx >= 0 && gx < img.width() && gy >= 0 && gy < img.height();
          s_img[ty][tx] = valid ? img(gx, gy) : 0;
          s_D[ty][tx]   = valid ? D(gx, gy) : v;
        }
      }
    }
    __syncthreads();

    // Block level distance transform
    __shared__ int block_changed;
    if (threadIdx.x == 0)
    {
      block_changed = 1;
      active[bid]   = false;
    }
    __syncthreads();

    while (block_changed)
    {
      __syncthreads();
      if (threadIdx.x == 0)
        block_changed = 0;
      __syncthreads();

      int t_changed = 0;
      t_changed |= pass<true>(s_img, s_D, D.width(), D.height(), l_eucl, l_grad);
      __syncthreads();
      t_changed |= pass<false>(s_img, s_D, D.width(), D.height(), l_eucl, l_grad);
      __syncthreads();
      t_changed |= pass_T<true>(s_img, s_D, D.width(), D.height(), l_eucl, l_grad);
      __syncthreads();
      t_changed |= pass_T<false>(s_img, s_D, D.width(), D.height(), l_eucl, l_grad);
      __syncthreads();

      if (t_changed)
        atomicOr_block(&block_changed, t_changed);
      __syncthreads();

      if (threadIdx.x == 0 && block_changed)
      {
        *changed = true;
        if (by > 0 && block_changed & BLOCK_CHANGED_TOP)
          active_candidate[(by - 1) * gridDim.x + bx] = true;
        if (by < gridDim.y - 1 && block_changed & BLOCK_CHANGED_BOTTOM)
          active_candidate[(by + 1) * gridDim.x + bx] = true;
        if (bx > 0 && block_changed & BLOCK_CHANGED_LEFT)
          active_candidate[by * gridDim.x + bx - 1] = true;
        if (bx < gridDim.x - 1 && block_changed & BLOCK_CHANGED_RIGHT)
          active_candidate[by * gridDim.x + bx + 1] = true;
        if (bx > 0 && by > 0 && block_changed & BLOCK_CHANGED_TOP_LEFT)
          active_candidate[(by - 1) * gridDim.x + bx - 1] = true;
        if (bx < gridDim.x - 1 && by > 0 && block_changed & BLOCK_CHANGED_TOP_RIGHT)
          active_candidate[(by - 1) * gridDim.x + bx + 1] = true;
        if (bx > 0 && by < gridDim.y - 1 && block_changed & BLOCK_CHANGED_BOTTOM_LEFT)
          active_candidate[(by + 1) * gridDim.x + bx - 1] = true;
        if (bx < gridDim.x - 1 && by < gridDim.y - 1 && block_changed & BLOCK_CHANGED_BOTTOM_RIGHT)
          active_candidate[(by + 1) * gridDim.x + bx + 1] = true;
      }
    }
    __syncthreads();

    // Loading tile into global memory
    {
      const int ty = threadIdx.x + 1;
      for (int tx = 0; tx < BLOCK_SIZE; ++tx)
      {
        if (x + tx < img.width() && y < img.height())
          D(x + tx, y) = s_D[ty][tx + 1];
      }
    }
    __syncthreads();
  }

  void generalised_distance_transform_blocks(const image2d_view<std::uint8_t>& img,
                                             const image2d_view<std::uint8_t>& mask, image2d_view<float>& D,
                                             float lambda, float v)
  {
    assert(img.width() == D.width() && img.height() == D.height() && img.width() == mask.width() &&
           img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && D.memory_kind() == e_memory_kind::GPU);
    assert(lambda >= 0 && lambda <= 1);
    const float l_grad = lambda;
    const float l_eucl = 1 - lambda;

    const float local_dist[] = {std::sqrt(2.f), 1.f, std::sqrt(2.f)};
    cudaMemcpyToSymbol(local_dist2d, local_dist, 3 * sizeof(float));

    // Initialization
    {
      dim3 gridDim((D.width() + 31) / 32, (D.height() + 31) / 32);
      dim3 blockDim(32, 32);
      initialize_generalised_distance_map<<<gridDim, blockDim>>>(mask, D, v);
      cudaDeviceSynchronize();
    }

    // Convergence
    bool* changed;
    cudaMallocManaged(&changed, sizeof(bool));
    *changed = true;

    // Kernel parameters
    const int grid_width  = (img.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_height = (img.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3      grid_dim(grid_width, grid_height);
    dim3      block_dim(BLOCK_SIZE);

    // Active block for current round and next round
    const auto size = grid_width * grid_height;
    bool*      active[2];
    cudaMalloc(&active[0], size * sizeof(bool));
    cudaMemset(active[0], true, size * sizeof(bool));
    cudaMalloc(&active[1], size * sizeof(bool));
    cudaMemset(active[1], false, size * sizeof(bool));

    bool even = false;
    while (*changed)
    {
      *changed = false;
      block_propagation<<<grid_dim, block_dim>>>(img, D, v, l_eucl, l_grad, even, active[even], active[!even], changed);
      even = !even;
      cudaDeviceSynchronize();
    }

    // Free memory
    cudaFree(active[0]);
    cudaFree(active[1]);
    cudaFree(changed);

    // Handle error
    if (const auto err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(std::format("Error while running distance transform: {}", cudaGetErrorString(err)));
  }

  image2d<float> generalised_distance_transform_blocks(const image2d_view<std::uint8_t>& img,
                                                       const image2d_view<std::uint8_t>& mask, float lambda, float v)
  {
    assert(img.width() == mask.width() && img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && mask.memory_kind() == e_memory_kind::GPU);
    image2d<float> D(img.width(), img.height(), e_memory_kind::GPU);
    generalised_distance_transform_blocks(img, mask, D, lambda, v);
    return D;
  }
} // namespace dt