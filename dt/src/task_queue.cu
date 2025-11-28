#include <cuda_runtime.h>

#include <vector>

#include "task_queue.cuh"

__constant__ int gQueueOffsets[HPQueue::MAX_PRIORITY];

HPQueue::HPQueue(int* storage, int* offsets)
{
  memset(this, 0, sizeof(HPQueue));
  m_base    = storage;
  m_offsets = offsets;
}

namespace
{
  void priorityCDF(int width, int height, int* output, int K)
  {
    int dmax  = std::max(width + 1, height + 1) / 2;
    int total = width * height + 1;
    output[0] = 0;
    for (int d = 0; d < dmax; ++d)
    {
      // Number of blocks at distance ≤ d-1 (ie < d)
      int count    = PriorityTools::distanceCDF(d, width, height);
      int priority = count * K / total;

      // We want the number of blocks at priority < p
      //                             ie  distance < min distance that gives priority p
      output[priority + 1] = count;
    }
    for (int i = 1; i < K; i++)
    {
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


TaskQueue::TaskQueue(int gridDimX, int gridDimY)
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


  std::vector<int> offsets(HPQueue::MAX_PRIORITY + 1);
  priorityCDF(gridDimX, gridDimY, offsets.data(), HPQueue::MAX_PRIORITY);


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
}