add_library(dt)
target_sources(dt PRIVATE
    src/circular_bucket_queue.cpp
    src/image2d_data.cpp
    src/image_view.cpp
    src/imprint.cpp
)
target_include_directories(dt PUBLIC include)
target_compile_features(dt PUBLIC cxx_std_23)