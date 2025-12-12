find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 90)

add_library(dt-core)
add_library(dt::core ALIAS dt-core)
target_sources(dt-core PRIVATE
    # CPP source code
    src/image2d_data.cpp
    src/image2d_view.cpp
    src/imprint.cpp
    src/inferno.cpp
    src/transfert.cpp
    src/invert_mask.cpp

    # IO
    src/imread.cpp
    src/imsave.cpp
    src/stb_image_implementation.c

    # CUDA source code
    src/block_priority.cu
    src/utils.cu
    src/task_queue.cu
    src/generalised_distance_transform.cu
    src/generalised_distance_transform_chessboard.cu
    src/generalised_distance_transform_task.cu
)
target_include_directories(dt-core PUBLIC include)
target_compile_features(dt-core PUBLIC cxx_std_23)
target_link_libraries(dt-core PUBLIC CUDA::cudart)