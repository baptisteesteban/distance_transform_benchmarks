add_library(dt)
target_sources(dt PRIVATE
    src/image_view.cpp
)
target_include_directories(dt PUBLIC include)