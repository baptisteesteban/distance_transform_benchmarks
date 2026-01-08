#include <dt/block_priority.hpp>

#include <cuda_runtime.h>

#include "task_queue.cuh"

namespace dt
{
  HPQueue::HPQueue(int* storage, std::uint32_t* offsets)
  {
    memset(this, 0, sizeof(HPQueue));
    m_base    = storage;
    m_offsets = offsets;
  }

  TaskQueue::TaskQueue(const dt::image2d_view<std::uint8_t>& mask, int gridDimX, int gridDimY)
  {
    m_gridDimX = gridDimX;
    m_gridDimY = gridDimY;
    int size   = gridDimX * gridDimY;

    cudaMalloc(&m_priorities, size);
    cudaMalloc(&m_cdf, HPQueue::MAX_PRIORITY * sizeof(std::uint32_t));
    m_level0_worksize = static_cast<int>(compute_block_priorities(mask, m_priorities, m_cdf));

    // HPQueues
    cudaMalloc(&m_queues_storage, 2 * size * sizeof(int));
    cudaMalloc(&m_hpqueues[0], sizeof(HPQueue));
    cudaMalloc(&m_hpqueues[1], sizeof(HPQueue));
    HPQueue q0(m_queues_storage, m_cdf);
    HPQueue q1(m_queues_storage + size, m_cdf);
    cudaMemcpy(m_hpqueues[0], &q0, sizeof(HPQueue), cudaMemcpyHostToDevice);
    cudaMemcpy(m_hpqueues[1], &q1, sizeof(HPQueue), cudaMemcpyHostToDevice);

    // Bitsets
    const int bitset_size = ((size + 31) / 32);
    cudaMalloc(&m_bitset_storage, 2 * bitset_size * sizeof(std::uint32_t));
    cudaMalloc(&m_block_status[0], sizeof(BitsetView));
    cudaMalloc(&m_block_status[1], sizeof(BitsetView));
    BitsetView bs0(m_bitset_storage);
    BitsetView bs1(m_bitset_storage + bitset_size);
    cudaMemcpy(m_block_status[0], &bs0, sizeof(BitsetView), cudaMemcpyHostToDevice);
    cudaMemcpy(m_block_status[1], &bs1, sizeof(BitsetView), cudaMemcpyHostToDevice);
  }

  TaskQueue::~TaskQueue()
  {
    cudaFree(m_queues_storage);
    cudaFree(m_hpqueues[0]);
    cudaFree(m_hpqueues[1]);

    cudaFree(m_priorities);
    cudaFree(m_cdf);

    cudaFree(m_bitset_storage);
    cudaFree(m_block_status[0]);
    cudaFree(m_block_status[1]);
  }
} // namespace dt