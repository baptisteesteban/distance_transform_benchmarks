#include <dt/border.hpp>
#include <dt/immersion.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/normalize.hpp>
#include <dt/propagation.hpp>

#include <iostream>

int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << "input_filename output_filename[png]\n";
    return 1;
  }

  auto img      = dt::imread<std::uint8_t>(argv[1]);
  auto bordered = dt::add_median_border(img);
  auto [m, M]   = dt::immersion(bordered);
  auto dt       = dt::propagation<std::uint16_t>(m, M);
  auto norm     = dt::normalize<std::uint8_t>(dt);
  auto colored  = dt::inferno(norm);

  dt::imsave(argv[2], colored);

  return 0;
}