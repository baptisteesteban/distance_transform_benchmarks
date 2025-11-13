add_library(dt)
target_sources(dt PRIVATE
    src/image_view.cpp
)
target_include_directories(dt PUBLIC include)
target_compile_features(dt PUBLIC cxx_std_23)