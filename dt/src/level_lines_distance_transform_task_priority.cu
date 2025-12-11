#include <dt/level_lines_distance_transform.hpp>
#include <dt/priority.hpp>

#include <cooperative_groups.h>

#include <cassert>

#include "lldt_block_passes.cuh"
#include "task_queue.cuh"
#include "utils.cuh"

namespace dt
{
  namespace cg = cooperative_groups;

  __device__ static bool block_propagation_job(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                                               image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D,
                                               DeviceTaskQueue tq)
  {
    // Loading tile into shared memory
    __shared__ std::uint8_t s_m[TILE_SIZE][TILE_SIZE];
    __shared__ std::uint8_t s_M[TILE_SIZE][TILE_SIZE];
    __shared__ std::uint8_t s_F[TILE_SIZE][TILE_SIZE];
    __shared__ std::uint32_t s_D[TILE_SIZE][TILE_SIZE];

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
      // Here, x and y denote the start of the block line in memory
      const std::uint8_t m0 = m(0, 0);

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

          bool valid  = gx >= 0 && gx < m.width() && gy >= 0 && gy < m.height();
          s_m[ty][tx] = valid ? m(gx, gy) : m0;
          s_M[ty][tx] = valid ? M(gx, gy) : m0;
          s_F[ty][tx] = valid ? F(gx, gy) : m0;
          if ((tx == 0 && ty == 0) || (tx == 0 && ty == TILE_SIZE - 1) || (tx == TILE_SIZE - 1 && ty == 0) ||
              (tx == TILE_SIZE - 1 && ty == TILE_SIZE - 1)) // Handling corners
            s_D[ty][tx] = std::numeric_limits<std::int32_t>::max();
          else
            s_D[ty][tx] = valid ? D(gx, gy) : 0;
        }
      }
    }
    __syncthreads();

    // Block level distance transform
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
      t_changed |= pass<true>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed |= pass<false>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed |= pass_T<true>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed |= pass_T<false>(s_m, s_M, s_F, s_D);
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
      tq.blockStatus.clear(block_id);
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
      }
    }
    __syncthreads();

    // Loading tile into global memory
    {
      const int ty = threadIdx.x + 1;
      for (int tx = 0; tx < BLOCK_SIZE; ++tx)
      {
        if (x + tx < m.width() && y < m.height())
        {
          F(x + tx, y) = s_F[ty][tx + 1];
          D(x + tx, y) = s_D[ty][tx + 1];
        }
      }
    }
    __syncthreads();

    return true;
  }

  __global__ static void block_propagation(image2d_view<std::uint8_t> m, image2d_view<std::uint8_t> M,
                                           image2d_view<std::uint8_t> F, image2d_view<std::uint32_t> D,
                                           DeviceTaskQueue tq)
  {
    auto          grid        = cg::this_grid();
    std::uint64_t queue_flags = 1;

    const int NUM_WORKERS      = gridDim.x;
    const int LEVEL_0_WORKSIZE = lldt_priority().distanceCDF(0, tq.gridDimX, tq.gridDimY);
    const int WORKER_JOB_SIZE  = std::max(10, LEVEL_0_WORKSIZE / NUM_WORKERS);

    while (queue_flags > 0)
    {
      for (int k = 0; k < WORKER_JOB_SIZE; k++)
      {
        if (!block_propagation_job(m, M, F, D, tq))
          break;
        __syncthreads();
      }
      grid.sync();
      queue_flags = tq.finishRound();
    }
  }

  void level_lines_distance_transform_task_priority_gpu(const image2d_view<std::uint8_t>& m,
                                                        const image2d_view<std::uint8_t>& M,
                                                        image2d_view<std::uint32_t>&      D)
  {
    assert(m.width() == M.width() && m.height() == M.height() && m.width() == D.width() && m.height() == D.height());
    assert(m.memory_kind() == e_memory_kind::GPU && M.memory_kind() == e_memory_kind::GPU &&
           D.memory_kind() == e_memory_kind::GPU);

    image2d<std::uint8_t> F(D.width(), D.height(), e_memory_kind::GPU);
    {
      dim3 block_dim(32, 32);
      dim3 grid_dim(D.width() + 31 / 32, D.height() + 31 / 32);
      init<<<grid_dim, block_dim>>>(m, D, F);
      cudaDeviceSynchronize();
    }

    const int nblocks_width  = (D.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int nblocks_height = (D.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    TaskQueue tq(nblocks_width, nblocks_height);
    {
      auto block_queue = tq.getDeviceQueue();
      std::swap(block_queue.currentQueue, block_queue.nextQueue);
      initialize_task_queue<<<1, 64>>>(block_queue);
    }

    const int      num_blocks_per_sm = 4;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);

    auto                       block_queue = tq.getDeviceQueue();
    image2d_view<std::uint8_t> nc_m(m);
    image2d_view<std::uint8_t> nc_M(M);
    void*                      kernelArgs[] = {&nc_m, &nc_M, &F, &D, &block_queue};
    dim3                       dim_grid(device_prop.multiProcessorCount * num_blocks_per_sm);
    cudaLaunchCooperativeKernel((void*)block_propagation, dim_grid, BLOCK_SIZE, kernelArgs);
    if (const auto err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(
          std::format("Error while running level lines distance transform: {}", cudaGetErrorString(err)));
  }

  image2d<std::uint32_t> level_lines_distance_transform_task_priority_gpu(const image2d_view<std::uint8_t>& m,
                                                                          const image2d_view<std::uint8_t>& M)
  {
    assert(m.width() == M.width() && m.height() == M.height());
    assert(m.memory_kind() == e_memory_kind::GPU && M.memory_kind() == e_memory_kind::GPU);

    image2d<std::uint32_t> D(m.width(), m.height(), e_memory_kind::GPU);
    level_lines_distance_transform_task_priority_gpu(m, M, D);
    return D;
  }
} // namespace dt