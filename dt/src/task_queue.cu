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
    void priorityCDF(const std::vector<int>& h_priorities, int* output, int K)
    {
      const int size = h_priorities.size();
      if (size == 0)
        return;

      output[0] = 0;
      std::vector<int> counts(K, 0);

      // Count how many blocks have priority < p for each p
      for (int p : h_priorities)
      {
        if (p >= 0 && p < K)
        {
          for (int i = p + 1; i <= K; ++i)
            counts[i]++;
        }
      }

      for (int i = 1; i < K; i++)
      {
        output[i] = counts[i];
        if (output[i] == 0)
        {
          output[i] = output[i - 1];
        }
        else
        {
          output[i] = (output[i] + 1) / 2; // We have two queues
        }
        assert(output[i] >= output[i - 1]);
      }
    }
  } // namespace

  TaskQueue::TaskQueue(int gridDimX, int gridDimY, const dt::image2d_view<std::uint8_t>& mask, int block_size)
    : gActiveBlocks(gridDimX * gridDimY)
  {
    m_gridDimX = gridDimX;
    m_gridDimY = gridDimY;
    m_round    = 0;

    // Allocate memory for the task queues
    int size = gridDimX * gridDimY;
    cudaMalloc(&gQueueStorage, size * sizeof(int));
    cudaMalloc(&gHPQueue[0], sizeof(HPQueue));
    cudaMalloc(&gHPQueue[1], sizeof(HPQueue));
    cudaMalloc(&gBlockPriority, size);

    // Compute block priorities from mask using the new library
    auto d_priorities = dt::compute_block_priorities(mask, block_size, HPQueue::MAX_PRIORITY);

    // Copy priorities to gBlockPriority
    auto             priorities = std::make_unique_for_overwrite<std::uint8_t[]>(size);
    std::vector<int> h_priorities(size);
    thrust::copy(d_priorities.begin(), d_priorities.end(), h_priorities.begin());
    for (int i = 0; i < size; ++i)
      priorities[i] = static_cast<std::uint8_t>(h_priorities[i]);
    cudaMemcpy(gBlockPriority, priorities.get(), size, cudaMemcpyHostToDevice);

    std::vector<int> offsets(HPQueue::MAX_PRIORITY + 1);
    priorityCDF(h_priorities, offsets.data(), HPQueue::MAX_PRIORITY);

    int* devgQueueOffsetPtr;
    cudaMemcpyToSymbol(gQueueOffsets, offsets.data(), HPQueue::MAX_PRIORITY * sizeof(int));
    cudaGetSymbolAddress((void**)&devgQueueOffsetPtr, gQueueOffsets);
    HPQueue q0(gQueueStorage, devgQueueOffsetPtr);
    HPQueue q1(gQueueStorage + (size + 1) / 2, devgQueueOffsetPtr);
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