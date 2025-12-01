find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 90)

add_library(dt-core)
add_library(dt::core ALIAS dt-core)
target_sources(dt-core PRIVATE
    # CPP source code
    src/circular_bucket_queue.cpp
    src/image2d_data.cpp
    src/image2d_view.cpp
    src/imprint.cpp
    src/inferno.cpp
    src/transfert.cpp

    # CUDA source code
    src/utils.cu
    src/task_queue.cu
    src/immersion.cu
    src/level_lines_distance_transform_fg.cu
    src/level_lines_distance_transform_chessboard.cu
    src/level_lines_distance_transform_task_priority.cu
    src/geodesic_distance_transform.cu
)
target_include_directories(dt-core PUBLIC include)
target_compile_features(dt-core PUBLIC cxx_std_23)
target_link_libraries(dt-core PUBLIC CUDA::cudart)

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