#include <dt/immersion.hpp>
#include <dt/imprint.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/level_lines_distance_transform.hpp>
#include <dt/normalize.hpp>
#include <dt/transfert.hpp>

int main(void)
{
  const auto _img       = dt::imread<std::uint8_t>("/data/amazigh.pgm");
  const auto img        = dt::host_to_device(_img);
  const auto [m, M]     = dt::immersion_gpu(img);
  const auto _D         = dt::level_lines_distance_transform_fg_gpu(m, M);
  const auto D          = dt::device_to_host(_D);
  const auto normalized = dt::normalize<std::uint8_t>(D);
  const auto colored    = dt::inferno(normalized);
  dt::imsave("out.png", colored);
}