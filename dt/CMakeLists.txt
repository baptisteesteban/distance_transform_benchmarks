add_library(dt-core)
add_library(dt::core ALIAS dt-core)
target_sources(dt-core PRIVATE
    src/circular_bucket_queue.cpp
    src/image2d_data.cpp
    src/image2d_view.cpp
    src/imprint.cpp
    src/inferno.cpp
)
target_include_directories(dt-core PUBLIC include)
target_compile_features(dt-core PUBLIC cxx_std_23)

if (BUILD_IO)
    add_library(dt-io)
    add_library(dt::io ALIAS dt-io)
    target_sources(dt-io PRIVATE
        src/imread.cpp
        src/imsave.cpp
        src/stb_image_implementation.c
    )
    target_link_libraries(dt-io PUBLIC dt-core)
endif(BUILD_IO)