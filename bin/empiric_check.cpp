#include <dt/fill.hpp>
#include <dt/generalised_distance_transform.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/invert_mask.hpp>
#include <dt/normalize.hpp>
#include <dt/transfert.hpp>

#include <format>
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc < 4)
  {
    std::cerr << std::format("Usage: {} input_filename lambda mask_filename\n", argv[0]);
    return 1;
  }

  auto                      img  = dt::imread<std::uint8_t>(argv[1]);
  dt::image2d<std::uint8_t> mask = dt::image2d<std::uint8_t>(img.width(), img.height());


  const auto temp_mask = dt::imread<std::uint8_t>(argv[3]);
  if (temp_mask.width() != img.width() || temp_mask.height() != img.height())
  {
    std::cerr << std::format("Invalid mask shape (Got (w: {} h: {}), expected (w: {} h: {}))", temp_mask.width(),
                             temp_mask.height(), img.width(), img.height());
    return 1;
  }
  for (int y = 0; y < img.height(); ++y)
  {
    for (int x = 0; x < img.width(); ++x)
      mask(x, y) = temp_mask(x, y);
  }

  const float lambda = std::atof(argv[2]);
  if (lambda < 0 || lambda > 1)
  {
    std::cerr << std::format("Lambda argument must be in the range [0.0 - 1.0] (Got {})", lambda);
    return 1;
  }

  const auto d_img      = dt::host_to_device(img);
  const auto d_mask     = dt::host_to_device(mask);
  const auto d_dist_ref = dt::generalised_distance_transform(d_img, d_mask, lambda);
  const auto d_dist     = dt::generalised_distance_transform_task(d_img, d_mask, lambda);
  const auto dist_ref   = dt::device_to_host(d_dist_ref);
  const auto dist       = dt::device_to_host(d_dist);

  int diff_count = 0;
  for (int y = 0; y < img.height(); y++)
  {
    for (int x = 0; x < img.width(); x++)
    {
      if (dist(x, y) != dist_ref(x, y))
      {
        if (diff_count < 5)
          std::cerr << "Different value at (" << x << ", " << y << "): got " << dist(x, y) << ", expected "
                    << dist_ref(x, y) << "\n";
        diff_count++;
      }
    }
  }
  if (diff_count > 0)
  {
    std::cerr << "Total differences: " << diff_count << " / " << (img.width() * img.height()) << "\n";
    return 1;
  }
  return 0;
}