#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/normalize.hpp>
#include <dt/transfert.hpp>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <iostream>

static constexpr int BLOCK_SIZE = 32;

__global__ static void initialize_distance(const dt::image2d_view<std::uint8_t>& mask, std::uint32_t* distance)
{
  const int bx      = blockIdx.x;
  const int by      = blockIdx.y;
  const int y       = by * blockDim.x + threadIdx.x;
  const int start_x = bx * blockDim.x;
  const int end_x   = std::min<int>(start_x + blockDim.x, mask.width());

  __shared__ int block_has_mask;
  if (threadIdx.x == 0)
    block_has_mask = 0;

  int line_has_mask = 0;
  if (y < mask.height())
  {
    for (int x = start_x; x != end_x; x++)
      line_has_mask |= mask(x, y);
  }
  __syncthreads();

  atomicOr(&block_has_mask, line_has_mask);
  __syncthreads();

  if (threadIdx.x == 0)
    distance[by * gridDim.x + bx] = (block_has_mask == 0) * distance[by * gridDim.x + bx];
}

template <bool Forward>
__global__ static void pass_manathan(std::uint32_t* distance, int width, int height)
{
  const int     y       = blockIdx.x * blockDim.x + threadIdx.x;
  const int     start_x = Forward ? 1 : width - 2;
  const int     end_x   = Forward ? width : -1;
  constexpr int inc     = Forward ? 1 : -1;
  constexpr int dx      = -1 * inc;

  if (y < height)
  {
    for (int x = start_x; x != end_x; x += inc)
      distance[y * width + x] = std::min(distance[y * width + x + dx] + 1, distance[y * width + x]);
  }
}

template <bool Forward>
__global__ static void pass_T_manathan(std::uint32_t* distance, int width, int height)
{
  const int     x       = blockIdx.x * blockDim.x + threadIdx.x;
  const int     start_y = Forward ? 1 : height - 2;
  const int     end_y   = Forward ? height : -1;
  constexpr int inc     = Forward ? 1 : -1;
  constexpr int dy      = -1 * inc;

  if (x < width)
  {
    for (int y = start_y; y != end_y; y += inc)
      distance[y * width + x] = std::min(distance[(y + dy) * width + x] + 1, distance[y * width + x]);
  }
}

void compute_block_priorities(const dt::image2d_view<std::uint8_t>& mask, std::uint8_t* priorities, std::uint32_t* cdf)
{
  assert(mask.memory_kind() == dt::e_memory_kind::GPU);

  const int      grid_width  = (mask.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int      grid_height = (mask.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int      N           = grid_width * grid_height;
  std::uint32_t* distance    = nullptr;
  cudaMalloc(&distance, N * sizeof(std::uint32_t));
  // Fill distance with a large value
  {
    auto ptr = thrust::device_pointer_cast(distance);
    thrust::fill(ptr, ptr + N, std::numeric_limits<std::uint32_t>::max() / 2);
  }
  // Initialize the distance (0: block has seed points, large value otherwise)
  {
    dim3 grid_dim(grid_width, grid_height);
    dim3 block_dim(BLOCK_SIZE);
    initialize_distance<<<grid_dim, block_dim>>>(mask, distance);
    cudaDeviceSynchronize();
  }

  // Four passes manathan distance computation
  {
    dim3 grid_dim((grid_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_dim(BLOCK_SIZE);
    pass_manathan<true><<<grid_dim, block_dim>>>(distance, grid_width, grid_height);
    pass_manathan<false><<<grid_dim, block_dim>>>(distance, grid_width, grid_height);
  }
  {
    dim3 grid_dim((grid_width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_dim(BLOCK_SIZE);
    pass_T_manathan<true><<<grid_dim, block_dim>>>(distance, grid_width, grid_height);
    pass_T_manathan<false><<<grid_dim, block_dim>>>(distance, grid_width, grid_height);
  }
  cudaDeviceSynchronize();

  // Normalization and CDF computation
  {
    const auto ptr            = thrust::device_pointer_cast(distance);
    auto       priorities_ptr = thrust::device_pointer_cast(priorities);
    // Normalization and priority computation from manathan distance
    const auto max_dist = *thrust::max_element(ptr, ptr + N);
    thrust::transform(ptr, ptr + N, priorities_ptr,
                      [max_dist] __device__(const auto& v) { return (static_cast<float>(v) / max_dist) * 63; });
    // CDF
    auto cdf_ptr = thrust::device_pointer_cast(cdf);
    cudaMemset(cdf, 0, 64 * sizeof(std::uint32_t));
    thrust::for_each(priorities_ptr, priorities_ptr + N, [cdf] __device__(const auto v) { atomicAdd(&cdf[v], 1); });
    thrust::inclusive_scan(cdf_ptr, cdf_ptr + 64, cdf_ptr);
  }
  cudaFree(distance);
}

template <typename T>
__global__ static void block_attribute_to_image(const T* attribute, dt::image2d_view<T>& img)
{
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int x  = bx * blockDim.x + threadIdx.x;
  const int y  = by * blockDim.y + threadIdx.y;

  if (x < img.width() && y < img.height())
    img(x, y) = attribute[by * gridDim.x + bx];
}

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " mask.png\n";
    return 1;
  }
  const auto mask   = dt::imread<std::uint8_t>(argv[1]);
  const auto d_mask = dt::host_to_device(mask);

  const int grid_width  = (mask.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int grid_height = (mask.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int N           = grid_width * grid_height;
  std::cout << "Grid with " << N << " blocks\n";
  std::uint8_t*  priorities = nullptr;
  std::uint32_t* cdf        = nullptr;
  cudaMalloc(&priorities, N * sizeof(std::uint8_t)); // Priority of each block
  cudaMalloc(&cdf, 64 * sizeof(std::uint32_t)); // Cumulative distribution function of the priorities in a cuda grid
  compute_block_priorities(d_mask, priorities, cdf);

  dt::image2d<std::uint8_t> d_attr_img(mask.width(), mask.height(), dt::e_memory_kind::GPU);
  {
    dim3 grid_dim(grid_width, grid_height);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    block_attribute_to_image<<<grid_dim, block_dim>>>(priorities, d_attr_img);
  }
  {
    std::uint32_t* h_cdf = (std::uint32_t*)std::malloc(64 * sizeof(std::uint32_t));
    cudaMemcpy(h_cdf, cdf, 64 * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 63; i++)
      std::cout << std::format("{:>3} ", h_cdf[i]);
    std::cout << std::format("{:>3}\n", h_cdf[63]);
  }
  const auto attr_img = dt::device_to_host(d_attr_img);
  const auto colored  = dt::inferno(dt::normalize<std::uint8_t>(attr_img));
  dt::imsave("priorities.png", colored);

  cudaFree(priorities);
  cudaFree(cdf);

  return 0;
}