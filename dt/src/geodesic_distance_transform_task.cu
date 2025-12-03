#include <dt/image2d.hpp>

#include <cooperative_groups.h>

#include "dt_block_passes.cuh"
#include "task_queue.cuh"
#include "utils.cuh"

namespace dt
{
  namespace cg = cooperative_groups;

  __global__ static void block_propagation(image2d_view<std::uint8_t> img, image2d_view<float> D, float l_grad,
                                           float l_eucl, DeviceTaskQueue tq)
  {
    auto          grid        = cg::this_grid();
    std::uint64_t queue_flags = 1;

    const int NUM_WORKERS      = gridDim.x;
    const int LEVEL_0_WORKSIZE = PriorityTools::distanceCDF(0, tq.gridDimX, tq.gridDimY);
    const int WORKER_JOB_SIZE  = std::max(10, LEVEL_0_WORKSIZE / NUM_WORKERS);

    while (queue_flags > 0)
    {
      for (int k = 0; k < WORKER_JOB_SIZE; k++)
      {
        if (/* TODO: Block propagation job */ true)
          break;
        __syncthreads();
      }
      grid.sync();
      queue_flags = tq.finishRound();
    }
  }

  void geodesic_distance_transform_task(const image2d_view<std::uint8_t>& img, const image2d_view<std::uint8_t>& mask,
                                        image2d_view<float>& D, float v, float lambda)
  {
    assert(img.width() == D.width() && img.height() == D.height() && img.width() == mask.width() &&
           img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && D.memory_kind() == e_memory_kind::GPU);
    assert(lambda >= 0 && lambda <= 1);
    float l_grad = lambda;
    float l_eucl = 1 - lambda;

    const float local_dist[] = {std::sqrt(2.f), 1.f, std::sqrt(2.f)};
    cudaMemcpyToSymbol(local_dist2d, local_dist, 3 * sizeof(float));

    {
      constexpr int INIT_BLOCK_SIZE = 32;
      dim3          gridDim((D.width() + INIT_BLOCK_SIZE - 1) / INIT_BLOCK_SIZE,
                            (D.height() + INIT_BLOCK_SIZE - 1) / INIT_BLOCK_SIZE);
      dim3          blockDim(INIT_BLOCK_SIZE, INIT_BLOCK_SIZE);
      initialize_geodesic_distance_map<<<gridDim, blockDim>>>(mask, D, v);
      cudaDeviceSynchronize();
    }

    const int grid_width  = (img.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_height = (img.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    TaskQueue tq(grid_width, grid_height);
    {
      auto block_queue = tq.getDeviceQueue();
      std::swap(block_queue.currentQueue, block_queue.nextQueue);
      initialize_task_queue<<<1, 64>>>(block_queue);
    }

    const int      num_blocks_per_sm = 4;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);

    auto                       block_queue = tq.getDeviceQueue();
    image2d_view<std::uint8_t> nc_img(img);
    void*                      kernelArgs[] = {&nc_img, &D, &l_grad, &l_eucl, &block_queue};
    dim3                       dim_grid(device_prop.multiProcessorCount * num_blocks_per_sm);
    cudaLaunchCooperativeKernel((void*)block_propagation, dim_grid, BLOCK_SIZE, kernelArgs);

    if (const auto err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(std::format("Error while running distance transform: {}", cudaGetErrorString(err)));
  }

  image2d<float> geodesic_distance_transform_task(const image2d_view<std::uint8_t>& img,
                                                  const image2d_view<std::uint8_t>& mask, float v, float lambda)
  {
    assert(img.width() == mask.width() && img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && mask.memory_kind() == e_memory_kind::GPU);
    image2d<float> D(img.width(), img.height(), e_memory_kind::GPU);
    geodesic_distance_transform_task(img, mask, D, v, lambda);
    return D;
  }
} // namespace dt