set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

pybind11_add_module(_dt_cxx
    module.cpp
    level_lines_distance_transform.cpp
)
target_link_libraries(_dt_cxx PRIVATE dt ${TORCH_LIBRARIES})
set_target_properties(_dt_cxx PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH ON
)

install(TARGETS _dt_cxx DESTINATION ./dt)