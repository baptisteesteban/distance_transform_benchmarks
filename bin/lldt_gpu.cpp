#include <dt/border.hpp>
#include <dt/immersion.hpp>
#include <dt/imprint.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/level_lines_distance_transform.hpp>
#include <dt/normalize.hpp>
#include <dt/propagation.hpp>
#include <dt/transfert.hpp>

#include <cstdio>

void save_colored(const char* filename, const dt::image2d_view<std::uint32_t>& D)
{
  const auto normalized = dt::normalize<std::uint8_t>(D);
  const auto colored    = dt::inferno(normalized);
  dt::imsave(filename, colored);
}

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cout << std::format("Usage: {} filename\n", argv[0]);
    return 1;
  }
  const auto _img = dt::add_median_border(dt::imread<std::uint8_t>(argv[1]));

  // CPU
  const auto [m, M] = dt::immersion(_img);
  const auto D_cpu  = dt::propagation<uint32_t>(m, M);
  save_colored("out_cpu.png", D_cpu);

  // GPU
  {
    const auto img            = dt::host_to_device(_img);
    const auto [m_gpu, M_gpu] = dt::immersion_gpu(img);
    const auto _D             = dt::level_lines_distance_transform_chessboard_gpu(m_gpu, M_gpu);
    const auto D              = dt::device_to_host(_D);

    auto diff      = dt::image2d<std::uint8_t>(D.width(), D.height());
    int  n_invalid = 0;
    for (int y = 0; y < D.height(); y++)
    {
      for (int x = 0; x < D.width(); x++)
      {
        if (D(x, y) != D_cpu(x, y))
        {
          diff(x, y) = 255;
          std::cout << std::format("Invalid value in ({}, {})\n", x, y);
          ++n_invalid;
        }
        else
        {
          diff(x, y) = 0;
        }
      }
    }
    if (!n_invalid)
    {
      std::cout << "Perfect match\n";
    }
    else
    {
      std::cout << "Invalid: " << (static_cast<float>(n_invalid) / (D.width() * D.height())) * 100 << " %\n";
    }
    save_colored("out_gpu.png", D);
    dt::imsave("diff.png", diff);
  }
}