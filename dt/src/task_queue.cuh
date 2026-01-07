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
  class FarStack
  {
  public:
    FarStack(int* storage);

    __device__ void push(int id)
    {
      int current        = m_size++;
      m_storage[current] = id;
    }
    __device__ int pop()
    {
      while (true)
      {
        int old_size = m_size.load(cuda::memory_order_acquire);
        if (old_size <= 0)
          return -1;
        if (m_size.compare_exchange_weak(old_size, old_size - 1, cuda::memory_order_release,
                                         cuda::memory_order_relaxed))
        {
          return m_storage[old_size - 1];
        }
      }
    }

  private:
    cuda::atomic<int, cuda::thread_scope_device> m_size;
    int*                                         m_storage;
  };

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
    int           gridDimX;
    int           gridDimY;
    Bitset        block_status;
    HPQueue*      current_queue;
    HPQueue*      next_queue;
    std::uint8_t* priorities;
    int           level0_worksize;

    FarStack* far_stack;
    Bitset    far_status;

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
      return blockIndex;
    }

    __device__ void clearBlockStatus(int blockIndex) { block_status.clear(blockIndex); }

    __device__ void enqueueFar(int x, int y)
    {
      int bid = y * gridDimX + x;
      if (far_status.test_and_set(bid))
        return;
      far_stack->push(bid);
    }

    __device__ uint64_t finishRound()
    {
      std::swap(current_queue, next_queue);
      for (int id = far_stack->pop(); id >= 0; id = far_stack->pop())
      {
        far_status.clear(id);
        enqueueTask(id % gridDimX, id / gridDimX);
      }
      return current_queue->getFlags() | next_queue->getFlags();
    }
  };


  class TaskQueue
  {
  public:
    TaskQueue(const dt::image2d_view<std::uint8_t>& mask, int gridDimX, int gridDimY);
    ~TaskQueue();

    DeviceTaskQueue get_device_queue()
    {
      return {m_gridDimX,   m_gridDimY,        m_block_status, m_hpqueues[0], m_hpqueues[1],
              m_priorities, m_level0_worksize, m_far_stack,    m_far_status};
    }

  private:
    int            m_gridDimX;   // Number of columns of the grid
    int            m_gridDimY;   // Number of rows of the grid
    std::uint8_t*  m_priorities; // Priority of each block of the grid
    std::uint32_t* m_cdf;        // Cumulative Distribution Function of the priorities
    int            m_level0_worksize;

    Bitset   m_block_status;   // Bitset indicating the status of a block (active or not)
    int*     m_queues_storage; // Storage of the queue
    HPQueue* m_hpqueues[2];    // Hierarchical priority queue

    // Far Stack
    int*      m_far_storage;
    FarStack* m_far_stack;
    Bitset    m_far_status;
  };
} // namespace dt