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
  __global__ static void pass(const image2d_view<std::uint8_t>& img, image2d_view<float>& D, float l_grad, float l_eucl,
                              bool* changed)
  {
    const int     start_x = Forward ? 1 : img.width() - 2;
    const int     end_x   = Forward ? img.width() : -1;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;

    // Current line
    const int y     = blockDim.x * blockIdx.x + threadIdx.x;
    const int inf_y = blockDim.x * blockIdx.x;
    const int sup_y = std::min<int>((blockIdx.x + 1) * blockDim.x, img.height());

    if (y >= img.height())
      return;

    for (int x = start_x; x != end_x; x += inc)
    {
      float new_dist = D(x, y);

      for (int dy = -1; dy < 2; dy++)
      {
        const int ny = y + dy;
        if (ny < inf_y || ny >= sup_y)
          continue;

        const float l_dist   = minus_abs(img(x, y), img(x + dx, ny));
        const float cur_dist = D(x + dx, ny) + l_eucl * local_dist2d[dy + 1] + l_grad * l_dist;
        new_dist             = std::min(new_dist, cur_dist);
      }

      if (new_dist < D(x, y))
      {
        D(x, y)  = new_dist;
        *changed = true;
      }

      __syncthreads();
    }
  }

  // Top -> Down
  template <bool Forward>
  __global__ static void pass_T(const image2d_view<std::uint8_t>& img, image2d_view<float>& D, float l_grad,
                                float l_eucl, bool* changed)
  {
    const int     start_y = Forward ? 1 : img.height() - 2;
    const int     end_y   = Forward ? img.height() : -1;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dy      = -1 * inc;

    // Current column
    const int x     = blockDim.x * blockIdx.x + threadIdx.x;
    const int inf_x = blockDim.x * blockIdx.x;
    const int sup_x = std::min<int>((blockIdx.x + 1) * blockDim.x, img.width());

    if (x >= img.width())
      return;

    for (int y = start_y; y != end_y; y += inc)
    {
      float new_dist = D(x, y);
      for (int dx = -1; dx < 2; dx++)
      {
        const int nx = x + dx;
        if (nx < inf_x || nx >= sup_x)
          continue;

        const float l_dist   = minus_abs(img(x, y), img(nx, y + dy));
        const float cur_dist = D(nx, y + dy) + l_eucl * local_dist2d[dx + 1] + l_grad * l_dist;
        new_dist             = std::min(new_dist, cur_dist);
      }

      if (new_dist < D(x, y))
      {
        D(x, y)  = new_dist;
        *changed = true;
      }

      __syncthreads();
    }
  }

  void geodesic_distance_transform(const image2d_view<std::uint8_t>& img, const image2d_view<std::uint8_t>& mask,
                                   image2d_view<float>& D, float v, float lambda)
  {
    assert(img.width() == D.width() && img.height() == D.height() && img.width() == mask.width() &&
           img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && D.memory_kind() == e_memory_kind::GPU);
    assert(lambda >= 0 && lambda <= 1);
    const float l_grad = lambda;
    const float l_eucl = 1 - lambda;

    const float local_dist[] = {std::sqrt(2.f), 1.f, std::sqrt(2.f)};
    cudaMemcpyToSymbol(local_dist2d, local_dist, 3 * sizeof(float));

    {
      dim3 gridDim((D.width() + 31) / 32, (D.height() + 31) / 32);
      dim3 blockDim(32, 32);
      initialize_geodesic_distance_map<<<gridDim, blockDim>>>(mask, D, v);
      cudaDeviceSynchronize();
    }

    const int n_blocks_w = (img.width() + N_THREADS - 1) / N_THREADS;
    const int n_blocks_h = (img.height() + N_THREADS - 1) / N_THREADS;
    bool*     changed;
    cudaMallocManaged(&changed, sizeof(bool));
    *changed = true;
    while (*changed)
    {
      *changed = false;
      pass<true><<<n_blocks_h, N_THREADS>>>(img, D, l_grad, l_eucl, changed);
      pass<false><<<n_blocks_h, N_THREADS>>>(img, D, l_grad, l_eucl, changed);
      pass_T<true><<<n_blocks_w, N_THREADS>>>(img, D, l_grad, l_eucl, changed);
      pass_T<false><<<n_blocks_w, N_THREADS>>>(img, D, l_grad, l_eucl, changed);
      cudaDeviceSynchronize();
    }
    cudaFree(changed);
    if (const auto err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(
          std::format("Error while running level lines distance transform: {}", cudaGetErrorString(err)));
  }

  image2d<float> geodesic_distance_transform(const image2d_view<std::uint8_t>& img,
                                             const image2d_view<std::uint8_t>& mask, float v, float lambda)
  {
    assert(img.width() == mask.width() && img.height() == mask.height());
    assert(img.memory_kind() == e_memory_kind::GPU && mask.memory_kind() == e_memory_kind::GPU);
    image2d<float> D(img.width(), img.height(), e_memory_kind::GPU);
    geodesic_distance_transform(img, mask, D, v, lambda);
    return D;
  }
} // namespace dt