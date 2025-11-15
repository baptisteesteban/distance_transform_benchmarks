set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

pybind11_add_module(_dt_cxx module.cpp)
target_link_libraries(_dt_cxx PRIVATE dt ${TORCH_LIBRARIES})

install(TARGETS _dt_cxx DESTINATION ./dt)