#include <dt/image2d.hpp>

#include <cooperative_groups.h>

#include "dt_block_passes.cuh"
#include "task_queue.cuh"
#include "utils.cuh"

namespace dt
{
  namespace cg = cooperative_groups;

  __device__ static bool block_propagation_job(const image2d_view<std::uint8_t>& img, image2d_view<float>& D,
                                               float l_grad, float l_eucl, float v, DeviceTaskQueue tq)
  {
    __shared__ std::uint8_t s_img[TILE_SIZE][TILE_SIZE];
    __shared__ float        s_D[TILE_SIZE][TILE_SIZE];


    __shared__ int block_id;
    if (threadIdx.x == 0)
      block_id = tq.popTask();
    __syncthreads();

    if (block_id < 0)
      return false;

    const int grid_width  = tq.gridDimX;
    const int grid_height = tq.gridDimY;
    const int bx          = block_id % grid_width;
    const int by          = block_id / grid_width;
    const int x           = bx * BLOCK_SIZE;
    const int y           = by * BLOCK_SIZE + threadIdx.x;

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

    int            block_changed = 0;
    __shared__ int iter_block_changed;
    if (threadIdx.x == 0)
      iter_block_changed = 1;
    __syncthreads();
    while (iter_block_changed)
    {
      if (threadIdx.x == 0)
        iter_block_changed = 0;
      __syncthreads();

      int t_changed = 0;
      t_changed |= pass<true>(s_img, s_D, img.width(), img.height(), l_eucl, l_grad, bx, by, grid_width);
      __syncthreads();
      t_changed |= pass<false>(s_img, s_D, img.width(), img.height(), l_eucl, l_grad, bx, by, grid_width);
      __syncthreads();
      t_changed |= pass_T<true>(s_img, s_D, img.width(), img.height(), l_eucl, l_grad, bx, by, grid_height);
      __syncthreads();
      t_changed |= pass_T<false>(s_img, s_D, img.width(), img.height(), l_eucl, l_grad, bx, by, grid_height);
      __syncthreads();

      if (t_changed)
        atomicOr_block(&iter_block_changed, t_changed);
      __syncthreads();

      block_changed |= iter_block_changed;
    }
    __syncthreads();

    // Enqueue new active blocks
    if (threadIdx.x == 0)
    {
      if (block_changed != 0)
      {
        if ((bx > 0) && (block_changed & BLOCK_CHANGED_LEFT))
          tq.enqueueTask(bx - 1, by);
        if ((bx < grid_width - 1) && (block_changed & BLOCK_CHANGED_RIGHT))
          tq.enqueueTask(bx + 1, by);
        if ((by > 0) && (block_changed & BLOCK_CHANGED_TOP))
          tq.enqueueTask(bx, by - 1);
        if ((by < grid_height - 1) && (block_changed & BLOCK_CHANGED_BOTTOM))
          tq.enqueueTask(bx, by + 1);

        if (bx > 0 && by > 0 && block_changed & BLOCK_CHANGED_TOP_LEFT)
          tq.enqueueFar(bx - 1, by - 1);
        if (bx < grid_width - 1 && by > 0 && block_changed & BLOCK_CHANGED_TOP_RIGHT)
          tq.enqueueFar(bx + 1, by - 1);
        if (bx > 0 && by < grid_height - 1 && block_changed & BLOCK_CHANGED_BOTTOM_LEFT)
          tq.enqueueFar(bx - 1, by + 1);
        if (bx < grid_width - 1 && by < grid_height - 1 && block_changed & BLOCK_CHANGED_BOTTOM_RIGHT)
          tq.enqueueFar(bx + 1, by + 1);
      }
    }
    __syncthreads();

    {
      const int ty = threadIdx.x + 1;
      for (int tx = 0; tx < BLOCK_SIZE; ++tx)
      {
        if (x + tx < img.width() && y < img.height())
          D(x + tx, y) = s_D[ty][tx + 1];
      }
    }
    __syncthreads();

    // Clear block status after writing results, allowing re-enqueue by neighbors
    if (threadIdx.x == 0)
      tq.clearBlockStatus(block_id);
    __syncthreads();

    return true;
  }

  __global__ static void block_propagation(image2d_view<std::uint8_t> img, image2d_view<float> D, float l_grad,
                                           float l_eucl, float v, DeviceTaskQueue tq)
  {
    auto          grid        = cg::this_grid();
    std::uint64_t queue_flags = 1;

    const int NUM_WORKERS      = gridDim.x;
    const int LEVEL_0_WORKSIZE = tq.level0_worksize;
    const int WORKER_JOB_SIZE  = std::max<int>(10, LEVEL_0_WORKSIZE / NUM_WORKERS);

    while (queue_flags > 0)
    {
      for (int k = 0; k < WORKER_JOB_SIZE; k++)
      {
        if (!block_propagation_job(img, D, l_grad, l_eucl, v, tq))
          break;
        __syncthreads();
      }
      grid.sync();
      queue_flags = tq.finishRound();
    }
  }

  void generalised_distance_transform_task(const image2d_view<std::uint8_t>& img,
                                           const image2d_view<std::uint8_t>& mask, image2d_view<float>& D, float lambda,
                                           float v)
  {
    assert(img.width() == D.width() && img.height() == D.height() && img.width() == mask.width() &&
           img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && D.memory_kind() == e_memory_kind::GPU);
    assert(lambda >= 0 && lambda <= 1);
    float l_grad = lambda;
    float l_eucl = 1 - lambda;

    const float local_dist[] = {std::sqrt(2.f), 1.f, std::sqrt(2.f)};
    cudaMemcpyToSymbol(local_dist2d, local_dist, 3 * sizeof(float));

    const int grid_width  = (img.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_height = (img.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3      grid_dim(grid_width, grid_height);
    {
      dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
      initialize_generalised_distance_map<<<grid_dim, blockDim>>>(mask, D, v);
      cudaDeviceSynchronize();
    }

    TaskQueue tq(mask, grid_width, grid_height);
    {
      auto block_queue = tq.get_device_queue();
      initialize_task_queue<false><<<grid_dim, 1>>>(block_queue);
      std::swap(block_queue.next_queue, block_queue.current_queue);
      initialize_task_queue<true><<<grid_dim, 1>>>(block_queue);
      cudaDeviceSynchronize();
    }

    const int      num_blocks_per_sm = 4;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);

    auto                       block_queue = tq.get_device_queue();
    image2d_view<std::uint8_t> nc_img(img);
    void*                      kernelArgs[] = {&nc_img, &D, &l_grad, &l_eucl, &v, &block_queue};
    dim3                       dim_grid(device_prop.multiProcessorCount * num_blocks_per_sm);
    cudaLaunchCooperativeKernel((void*)block_propagation, dim_grid, BLOCK_SIZE, kernelArgs);

    cudaDeviceSynchronize();
    if (const auto err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(std::format("Error while running distance transform: {}", cudaGetErrorString(err)));
  }

  image2d<float> generalised_distance_transform_task(const image2d_view<std::uint8_t>& img,
                                                     const image2d_view<std::uint8_t>& mask, float lambda, float v)
  {
    assert(img.width() == mask.width() && img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && mask.memory_kind() == e_memory_kind::GPU);
    image2d<float> D(img.width(), img.height(), e_memory_kind::GPU);
    generalised_distance_transform_task(img, mask, D, lambda, v);
    return D;
  }
} // namespace dt