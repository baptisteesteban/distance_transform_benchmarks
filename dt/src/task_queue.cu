#include <dt/block_priority.hpp>
#include <dt/image2d.hpp>

#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "task_queue.cuh"

namespace dt
{
  __constant__ int gQueueOffsets[HPQueue::MAX_PRIORITY];

  HPQueue::HPQueue(int* storage, int* offsets)
  {
    memset(this, 0, sizeof(HPQueue));
    m_base    = storage;
    m_offsets = offsets;
  }

  namespace
  {
    void setupEqualOffsets(int* output, int K, int queue_size)
    {
      // Allocate equal space for each priority level
      // This ensures any priority can hold up to queue_size/K blocks
      int space_per_priority = queue_size / K;
      for (int i = 0; i < K; i++)
      {
        output[i] = i * space_per_priority;
      }
      // Set the sentinel value for the last offset
      output[K] = queue_size;
    }
  } // namespace

  TaskQueue::TaskQueue(int gridDimX, int gridDimY, const dt::image2d_view<std::uint8_t>& mask, int block_size)
    : gActiveBlocks(gridDimX * gridDimY)
  {
    m_gridDimX = gridDimX;
    m_gridDimY = gridDimY;
    m_round    = 0;

    int size          = gridDimX * gridDimY;
    int total_storage = 2 * size * HPQueue::MAX_PRIORITY;
    cudaMalloc(&gQueueStorage, total_storage * sizeof(int));
    cudaMalloc(&gHPQueue[0], sizeof(HPQueue));
    cudaMalloc(&gHPQueue[1], sizeof(HPQueue));
    cudaMalloc(&gBlockPriority, size);

    auto d_priorities = dt::compute_block_priorities(mask, block_size, HPQueue::MAX_PRIORITY);

    auto             priorities = std::make_unique_for_overwrite<std::uint8_t[]>(size);
    std::vector<int> h_priorities(size);
    thrust::copy(d_priorities.begin(), d_priorities.end(), h_priorities.begin());
    for (int i = 0; i < size; ++i)
      priorities[i] = static_cast<std::uint8_t>(h_priorities[i]);
    cudaMemcpy(gBlockPriority, priorities.get(), size, cudaMemcpyHostToDevice);

    // Compute number of blocks with priority 0 and store for device-side kernels
    int level0_count = 0;
    for (int i = 0; i < size; ++i)
      if (h_priorities[i] == 0)
        ++level0_count;
    m_level0_worksize = level0_count;

    std::vector<int> offsets(HPQueue::MAX_PRIORITY + 1);
    int              queue_size = size * HPQueue::MAX_PRIORITY; // Each queue gets half the total
    setupEqualOffsets(offsets.data(), HPQueue::MAX_PRIORITY, queue_size);

    int* devgQueueOffsetPtr;
    cudaMemcpyToSymbol(gQueueOffsets, offsets.data(), HPQueue::MAX_PRIORITY * sizeof(int));
    cudaGetSymbolAddress((void**)&devgQueueOffsetPtr, gQueueOffsets);
    HPQueue q0(gQueueStorage, devgQueueOffsetPtr);
    HPQueue q1(gQueueStorage + size, devgQueueOffsetPtr);
    cudaMemcpy(gHPQueue[0], &q0, sizeof(HPQueue), cudaMemcpyHostToDevice);
    cudaMemcpy(gHPQueue[1], &q1, sizeof(HPQueue), cudaMemcpyHostToDevice);
  }

  TaskQueue::~TaskQueue()
  {
    cudaFree(gQueueStorage);
    cudaFree(gHPQueue[0]);
    cudaFree(gHPQueue[1]);
    cudaFree(gBlockPriority);
  }
} // namespace dt