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

void save_colored(const char* filename, const dt::image2d_view<std::uint32_t>& D)
{
  const auto normalized = dt::normalize<std::uint8_t>(D);
  const auto colored    = dt::inferno(normalized);
  dt::imsave(filename, colored);
}

int main(void)
{
  const auto _img = dt::add_median_border(dt::imread<std::uint8_t>("/data/amazigh.pgm"));

  // CPU
  {
    const auto [m, M] = dt::immersion(_img);
    const auto D      = dt::propagation<uint32_t>(m, M);
    save_colored("out_cpu.png", D);
  }

  // GPU
  {
    const auto img    = dt::host_to_device(_img);
    const auto [m, M] = dt::immersion_gpu(img);
    const auto _D     = dt::level_lines_distance_transform_fg_gpu(m, M);
    const auto D      = dt::device_to_host(_D);
    save_colored("out_gpu.png", D);
  }
}