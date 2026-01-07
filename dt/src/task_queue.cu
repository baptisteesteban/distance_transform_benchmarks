#include <dt/block_priority.hpp>

#include <cuda_runtime.h>

#include "task_queue.cuh"

namespace dt
{
  FarStack::FarStack(int* storage)
  {
    memset(this, 0, sizeof(FarStack));
    m_storage = storage;
  }

  HPQueue::HPQueue(int* storage, std::uint32_t* offsets)
  {
    memset(this, 0, sizeof(HPQueue));
    m_base    = storage;
    m_offsets = offsets;
  }

  TaskQueue::TaskQueue(const dt::image2d_view<std::uint8_t>& mask, int gridDimX, int gridDimY)
    : m_block_status(gridDimX * gridDimY)
    , m_far_status(gridDimX * gridDimY)
  {
    m_gridDimX = gridDimX;
    m_gridDimY = gridDimY;

    int size = gridDimX * gridDimY;
    cudaMalloc(&m_queues_storage, 2 * size * sizeof(int));
    cudaMalloc(&m_hpqueues[0], sizeof(HPQueue));
    cudaMalloc(&m_hpqueues[1], sizeof(HPQueue));

    cudaMalloc(&m_priorities, size);
    cudaMalloc(&m_cdf, HPQueue::MAX_PRIORITY * sizeof(std::uint32_t));
    m_level0_worksize = static_cast<int>(compute_block_priorities(mask, m_priorities, m_cdf));

    HPQueue q0(m_queues_storage, m_cdf);
    HPQueue q1(m_queues_storage + size, m_cdf);
    cudaMemcpy(m_hpqueues[0], &q0, sizeof(HPQueue), cudaMemcpyHostToDevice);
    cudaMemcpy(m_hpqueues[1], &q1, sizeof(HPQueue), cudaMemcpyHostToDevice);

    // Far stack
    cudaMalloc(&m_far_storage, size * sizeof(int));
    FarStack st(m_far_storage);
    cudaMalloc(&m_far_stack, sizeof(FarStack));
    cudaMemcpy(m_far_stack, &st, sizeof(FarStack), cudaMemcpyHostToDevice);
  }

  TaskQueue::~TaskQueue()
  {
    cudaFree(m_queues_storage);
    cudaFree(m_hpqueues[0]);
    cudaFree(m_hpqueues[1]);

    cudaFree(m_priorities);
    cudaFree(m_cdf);

    m_block_status.release();

    cudaFree(m_far_stack);
    cudaFree(m_far_storage);
    m_far_status.release();
  }
} // namespace dt