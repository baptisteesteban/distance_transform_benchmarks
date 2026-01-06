#pragma once

#include <dt/image2d_view.hpp>

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>

#ifndef __CUDA_ARCH__
#define __ffsll ffsll
#endif

#include "bitset.cuh"

#include <cstdio>

namespace dt
{
  class HPQueue
  {
  public:
    static constexpr int MAX_PRIORITY = 64;
    static_assert(MAX_PRIORITY <= sizeof(uint64_t) * CHAR_BIT,
                  "MAX_PRIORITY must be less than the number of bits in uint64_t");

    HPQueue(int* storage, std::uint32_t* offsets);

    __device__ void enqueue(int task, int priority)
    {
      assert(priority >= 0 && priority < MAX_PRIORITY);
      int  pos   = m_count[priority]++;
      int* queue = m_base + m_offsets[priority];
      // printf("(%d) p %d: %d\n", task, priority, m_offsets[priority]);
      if (priority < MAX_PRIORITY - 1)
        assert(0 <= pos && (m_offsets[priority] + pos) < m_offsets[priority + 1]);
      queue[pos] = task;
      m_flags |= (1UL << priority);
    }

    __device__ int dequeue()
    {
      do
      {
        uint64_t flags = m_flags;
        if (flags == 0)
          return -1;

        // Get the position of the highest priority task
        int priority = __ffsll(flags) - 1;
        int count    = m_count[priority]--;
        if (count <= 1)
        {
          m_flags &= ~(1UL << priority);
          m_count[priority] = 0;
        }
        if (count > 0)
        {
          int* queue      = m_base + m_offsets[priority];
          int  blockIndex = queue[count - 1];
          return blockIndex;
        }
      } while (true);
    }

    __device__ uint64_t getFlags() const { return m_flags; }

    uint64_t isEmpty() const
    {
      uint64_t flags;
      cudaMemcpy(&flags, &m_flags, sizeof(uint64_t), cudaMemcpyDeviceToHost);
      return flags;
    }


  private:
    cuda::atomic<uint64_t, cuda::thread_scope_device> m_flags;
    cuda::atomic<int, cuda::thread_scope_device>      m_count[MAX_PRIORITY];
    int*                                              m_base;
    std::uint32_t*                                    m_offsets;
  };


  struct DeviceTaskQueue
  {
    int            gridDimX;
    int            gridDimY;
    Bitset         block_status;
    HPQueue*       current_queue;
    HPQueue*       next_queue;
    std::uint8_t*  priorities;
    std::uint32_t* cdf;

    __device__ bool enqueueTask(int x, int y)
    {
      int blockIndex = y * gridDimX + x;

      if (block_status.test_and_set(blockIndex))
        return false;

      next_queue->enqueue(blockIndex, priorities[blockIndex]);
      return true;
    }

    __device__ int popTask()
    {
      int blockIndex = current_queue->dequeue();
      if (blockIndex >= 0)
        block_status.clear(blockIndex);
      return blockIndex;
    }


    __device__ uint64_t finishRound()
    {
      std::swap(current_queue, next_queue);
      return current_queue->getFlags();
    }

    __forceinline__ __device__ std::uint32_t level0_work_size() const { return cdf[0]; }
  };


  class TaskQueue
  {
  public:
    TaskQueue(const dt::image2d_view<std::uint8_t>& mask, int gridDimX, int gridDimY);
    ~TaskQueue();

    DeviceTaskQueue get_device_queue()
    {
      return {m_gridDimX, m_gridDimY, m_block_status, m_hpqueues[0], m_hpqueues[1], m_priorities, m_cdf};
    }

  private:
    int            m_gridDimX;   // Number of columns of the grid
    int            m_gridDimY;   // Number of rows of the grid
    std::uint8_t*  m_priorities; // Priority of each block of the grid
    std::uint32_t* m_cdf;        // Cumulative Distribution Function of the priorities

    Bitset   m_block_status;   // Bitset indicating the status of a block (active or not)
    int*     m_queues_storage; // Storage of the queue
    HPQueue* m_hpqueues[2];    // Hierarchical priority queue
  };
} // namespace dt