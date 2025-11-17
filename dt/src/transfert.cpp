#include <dt/transfert.hpp>

#include <cuda_runtime.h>

#include <format>
#include <stdexcept>

namespace dt
{
  void host_to_device(const image2d_view<void>& src, image2d_view<void>& dst)
  {
    assert(src.memory_kind() == e_memory_kind::CPU && dst.memory_kind() == e_memory_kind::GPU);
    assert(src.width() == dst.width() && src.height() == dst.height());
    auto err = cudaMemcpy2D(dst.buffer(), dst.pitch(), src.buffer(), src.pitch(), src.width(), src.height(),
                            cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw std::runtime_error(std::format("Unable to transfert host to device: {}", cudaGetErrorString(err)));
  }

  void device_to_host(const image2d_view<void>& src, image2d_view<void>& dst)
  {
    assert(src.memory_kind() == e_memory_kind::GPU && dst.memory_kind() == e_memory_kind::CPU);
    assert(src.width() == dst.width() && src.height() == dst.height());
    auto err = cudaMemcpy2D(dst.buffer(), dst.pitch(), src.buffer(), src.pitch(), src.width(), src.height(),
                            cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw std::runtime_error(std::format("Unable to transfert device to host: {}", cudaGetErrorString(err)));
  }
} // namespace dt