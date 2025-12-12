#pragma once

#include <dt/image2d.hpp>

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>

#ifndef __CUDA_ARCH__
#define __ffsll ffsll
#endif

#include "bitset.cuh"

namespace dt
{
  class HPQueue
  {
  public:
    static constexpr int MAX_PRIORITY = 64;
    static_assert(MAX_PRIORITY <= sizeof(uint64_t) * CHAR_BIT,
                  "MAX_PRIORITY must be less than the number of bits in uint64_t");

    // Initialize from host
    HPQueue(int* storage, int* offsets);

    /// @brief Schedule a task with a given priority
    /// @param task
    /// @param priory
    /// @details Enqueue operations can be called concurrently from multiple threads but should not be called
    /// concurrently with dequeue
    __device__ void enqueue(int task, int priority)
    {
      assert(priority >= 0 && priority < MAX_PRIORITY);
      int  pos   = m_count[priority]++;
      int* queue = m_base + m_offsets[priority];
      if (priority < MAX_PRIORITY - 1)
        assert(0 <= pos && (m_offsets[priority] + pos) < m_offsets[priority + 1]);
      queue[pos] = task;
      m_flags |= (1UL << priority);
    }


    /// @brief Pop the next task to be executed
    /// @details Dequeue operations can be called concurrently from multiple threads but should not be called
    /// concurrently with enqueue
    /// @return -1 if the queue is empty, otherwise the task index
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
    cuda::atomic<uint64_t, cuda::thread_scope_device> m_flags; // bitset of queue status
    cuda::atomic<int, cuda::thread_scope_device>      m_count[MAX_PRIORITY];
    int*                                              m_base;    // Pointer to the storage (grid constant)
    int*                                              m_offsets; // Array of offsets (grid constant)
  };


  struct DeviceTaskQueue
  {
    int           gridDimX;
    int           gridDimY;
    Bitset        blockStatus; // Current active blocks
    HPQueue*      currentQueue;
    HPQueue*      nextQueue;
    std::uint8_t* blockPriorities;


    // Schedule the block (x,y) for execution in the next round
    // Can be called concurrently from multiple threads
    __device__ bool enqueueTask(int x, int y)
    {
      int blockIndex = y * gridDimX + x;

      // If the block was already active, do not enqueue it
      if (blockStatus.test_and_set(blockIndex))
        return false;

      nextQueue->enqueue(blockIndex, blockPriorities[blockIndex]);
      return true;
    }


    /// @brief Pop the next block to be executed in the current round
    /// @details Can be called concurrently from multiple threads
    /// @return -1 if the queue is empty, otherwise the block index
    ///
    /// @note Pop and enqueue operations on the same queue is not supported
    __device__ int popTask()
    {
      int blockIndex = currentQueue->dequeue();
      if (blockIndex >= 0)
        blockStatus.clear(blockIndex);
      return blockIndex;
    }


    __device__ uint64_t finishRound()
    {
      std::swap(currentQueue, nextQueue);
      return currentQueue->getFlags();
    }
  };


  class TaskQueue
  {
  public:
    TaskQueue(int gridDimX, int gridDimY, const dt::image2d_view<std::uint8_t>& mask, int block_size);
    ~TaskQueue();


    // Host function to be called at the beginning of each round
    // Returns the status of the queues (1 bit per queue). If the return value is 0, there are no more tasks to be
    // executed
    uint64_t finishRound();

    DeviceTaskQueue getDeviceQueue()
    {
      return {m_gridDimX, m_gridDimY, gActiveBlocks, gHPQueue[m_round], gHPQueue[!m_round], gBlockPriority};
    }

  private:
    bool                m_round;    // Indicates if the current round is even or odd
    int                 m_gridDimX; // Number of blocks in the grid in the x direction
    int                 m_gridDimY; // Number of blocks in the grid in the y direction
    cudaTextureObject_t m_offsets;  // Array of offsets for the HPQueue (of type int32)


    // Data allocated on the device
    Bitset   gActiveBlocks; // Priority of each block in the grid
    int*     gQueueStorage; // Storage for the queue data
    HPQueue* gHPQueue[2]; // Queues of blocks to be executed in the EVEN round (gHPQueue[0]) and ODD round (gHPQueue[1])
    std::uint8_t* gBlockPriority;
  };


  inline uint64_t TaskQueue::finishRound()
  {
    m_round = !m_round;
    return gHPQueue[m_round]->isEmpty();
  }
} // namespace dt