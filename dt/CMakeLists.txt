add_library(stb_image OBJECT src/stb_image_implementation.c)

add_library(dt)
target_sources(dt PRIVATE
    src/circular_bucket_queue.cpp
    src/image2d_data.cpp
    src/image2d_view.cpp
    src/imprint.cpp
    src/imread.cpp
    src/imsave.cpp
    src/inferno.cpp
)
target_include_directories(dt PUBLIC include)
target_compile_features(dt PUBLIC cxx_std_23)
target_link_libraries(dt PRIVATE stb_image)