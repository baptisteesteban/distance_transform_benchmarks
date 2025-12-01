// Some code implemented here come from
// https://raw.githubusercontent.com/masadcv/FastGeodis/refs/heads/master/FastGeodis/fastgeodis_cuda.cu

#include <dt/geodesic_distance_transform.hpp>

#include "utils.cuh"

static constexpr int N_THREADS = 256;

__constant__ float local_dist2d[3];

namespace dt
{
  // Left -> Right
  template <bool Forward>
  __global__ static void pass(const image2d_view<std::uint8_t>& img, image2d_view<float>& D, float l_grad, float l_eucl)
  {
    const int     start_x = Forward ? 1 : img.width() - 2;
    const int     end_x   = Forward ? img.width() : -1;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;

    // Current line
    const int y     = blockDim.x * blockIdx.x + threadIdx.x;
    const int inf_y = blockDim.x * blockIdx.x;
    const int sup_y = (blockIdx.x + 1) * blockDim.x;

    if (y >= img.height())
      return;

    for (int x = start_x; x != end_x; x += inc)
    {
      float new_dist = D(x, y);

      for (int dy = -1; dy < 2; dy++)
      {
        const int ny = y + dy;
        if (ny < inf_y || ny >= sup_y || ny >= img.height())
          continue;

        const float l_dist   = l1distance_cuda(img(x, y), img(x + dx, ny));
        const float cur_dist = D(x + dx, ny) + l_eucl * local_dist2d[dy + 1] + l_grad * l_dist;
        new_dist             = std::min(new_dist, cur_dist);
      }

      if (new_dist < D(x, y))
        D(x, y) = new_dist;

      __syncthreads();
    }
  }

  // Top -> Down
  template <bool Forward>
  __global__ static void pass_T(const image2d_view<std::uint8_t>& img, image2d_view<float>& D, float l_grad,
                                float l_eucl)
  {
    const int     start_y = Forward ? 1 : img.height() - 2;
    const int     end_y   = Forward ? img.height() : -1;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dy      = -1 * inc;

    // Current column
    const int x     = blockDim.x * blockDim.x + threadIdx.x;
    const int inf_x = blockDim.x * blockIdx.x;
    const int sup_x = (blockIdx.x + 1) * blockDim.x;

    if (x >= img.width())
      return;

    for (int y = start_y; y != end_y; y += inc)
    {
      float new_dist = D(x, y);
      for (int dx = -1; dx < 2; dx++)
      {
        const int nx = x + dx;
        if (nx < inf_x || nx >= sup_x || nx >= img.width())
          continue;

        const float l_dist   = l1distance_cuda(img(x, y), img(nx, y + dy));
        const float cur_dist = D(nx, y + dy) + l_eucl * local_dist2d[dx + 1] + l_grad * l_dist;
        new_dist             = std::min(new_dist, cur_dist);
      }

      if (new_dist < D(x, y))
        D(x, y) = new_dist;

      __syncthreads();
    }
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

    {
      dim3 gridDim((D.width() + 31) / 32, (D.height() + 31) / 32);
      dim3 blockDim(32, 32);
      initialize_distance_map<<<gridDim, blockDim>>>(mask, D, v);
    }

    const int n_blocks_w = (img.width() + N_THREADS - 1) / N_THREADS;
    const int n_blocks_h = (img.height() + N_THREADS - 1) / N_THREADS;
    for (int i = 0; i < iterations; i++)
    {
      pass<true><<<n_blocks_h, N_THREADS>>>(img, D, l_grad, l_eucl);
      pass<false><<<n_blocks_h, N_THREADS>>>(img, D, l_grad, l_eucl);
      pass_T<true><<<n_blocks_w, N_THREADS>>>(img, D, l_grad, l_eucl);
      pass_T<false><<<n_blocks_w, N_THREADS>>>(img, D, l_grad, l_eucl);
      cudaDeviceSynchronize();
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