// Some code implemented here come from
// https://raw.githubusercontent.com/masadcv/FastGeodis/refs/heads/master/FastGeodis/fastgeodis_cuda.cu

#include <dt/geodesic_distance_transform.hpp>

static constexpr int N_THREADS = 256;

__constant__ float local_dist2d[3];

namespace dt
{
  // Left -> Right
  template <bool Forward>
  __global__ static void pass(const image2d_view<std::uint8_t>& img, image2d_view<float>& D, float l_grad, float l_eucl)
  {
  }

  // Top -> Down
  template <bool Forward>
  __global__ static void pass_T(const image2d_view<std::uint8_t>& img, image2d_view<float>& D, float l_grad,
                                float l_eucl)
  {
  }

  __global__ static void initialize_distance_map(const image2d_view<std::uint8_t>& mask, image2d_view<float>& D,
                                                 float v)
  {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < D.width() && y < D.height())
      D(x, y) = v * (mask(x, y) > 0);
  }

  void geodesic_distance_transform(const image2d_view<std::uint8_t>& img, const image2d_view<std::uint8_t>& mask,
                                   image2d_view<float>& D, float v, float l_grad, float l_eucl, int iterations)
  {
    assert(img.width() == D.width() && img.height() == D.height() && img.width() == mask.width() &&
           img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && D.memory_kind() == e_memory_kind::GPU);

    const float local_dist[] = {std::sqrt(2.f), 1.f, std::sqrt(2.f)};
    cudaMemcpyToSymbol(local_dist2d, local_dist, 3 * sizeof(float));

    const int n_blocks = (img.width() + 1) / N_THREADS + 1;

    {
      dim3 gridDim(D.width() + 31 / 32, D.height() + 31 / 32);
      dim3 blockDim(32, 32);
      initialize_distance_map<<<gridDim, blockDim>>>(mask, D, v);
    }

    for (int i = 0; i < iterations; i++)
    {
      pass<true><<<n_blocks, N_THREADS>>>(img, D, l_grad, l_eucl);
      pass<false><<<n_blocks, N_THREADS>>>(img, D, l_grad, l_eucl);
      pass_T<true><<<n_blocks, N_THREADS>>>(img, D, l_grad, l_eucl);
      pass_T<false><<<n_blocks, N_THREADS>>>(img, D, l_grad, l_eucl);
    }
  }

  image2d<float> geodesic_distance_transform(const image2d_view<std::uint8_t>& img,
                                             const image2d_view<std::uint8_t>& mask, float v, float l_grad,
                                             float l_eucl, int iterations)
  {
    assert(img.width() == mask.width() && img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && mask.memory_kind() == e_memory_kind::GPU);
    image2d<float> D(img.width(), img.height(), e_memory_kind::GPU);
    geodesic_distance_transform(img, mask, D, v, l_grad, l_eucl, iterations);
    return D;
  }
} // namespace dt