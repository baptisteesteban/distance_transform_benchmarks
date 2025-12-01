#include <dt/fill.hpp>
#include <dt/geodesic_distance_transform.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/normalize.hpp>
#include <dt/transfert.hpp>

int main(int argc, char* argv[])
{
  auto img  = dt::image2d<std::uint8_t>(50, 40);
  auto mask = dt::image2d<std::uint8_t>(50, 40);
  dt::fill(img, std::uint8_t(0));
  dt::fill(mask, std::uint8_t(1));
  for (int y = 20; y < 30; y++)
  {
    for (int x = 20; x < 43; x++)
      mask(x, y) = 0;
  }
  const auto d_img   = dt::host_to_device(img);
  const auto d_mask  = dt::host_to_device(mask);
  const auto d_dist  = dt::geodesic_distance_transform(d_img, d_mask, 1e10, 0, 1, 100);
  const auto dist    = dt::device_to_host(d_dist);
  const auto norm    = dt::normalize<std::uint8_t>(dist);
  const auto colored = dt::inferno(norm);
  dt::imsave("out.png", colored);
}