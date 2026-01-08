#pragma once
#include <cuda/atomic>
#include <cuda_runtime.h>

#ifndef __device__
#define __device__
#endif

class BitsetView
{
public:
  BitsetView(std::uint32_t* data);

  __device__ bool test_and_set(int index);
  __device__ void clear(int index);

protected:
  uint32_t* data;
};

class Bitset final : public BitsetView
{
  Bitset(int size);
  void release();
};

inline BitsetView::BitsetView(std::uint32_t* data)
  : data(data)
{
}

inline __device__ bool BitsetView::test_and_set(int index)
{
  int      word  = index / 32;
  int      bit   = index % 32;
  uint32_t mask  = 1 << bit;
  auto     w     = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(this->data[word]);
  uint32_t value = w.fetch_or(mask);
  return value & mask;
}

inline __device__ void BitsetView::clear(int index)
{
  int      word = index / 32;
  int      bit  = index % 32;
  uint32_t mask = ~(1 << bit);
  auto     w    = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(this->data[word]);
  w.fetch_and(mask);
}

inline Bitset::Bitset(int size)
  : BitsetView(nullptr)
{
  cudaMalloc(&this->data, sizeof(uint32_t) * ((size + 31) / 32));
  cudaMemset(this->data, 0, sizeof(uint32_t) * ((size + 31) / 32));
}

inline void Bitset::release()
{
  if (this->data)
    cudaFree(this->data);
}