find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_dt_cxx module.cpp)
target_link_libraries(_dt_cxx PRIVATE dt)

install(TARGETS _dt_cxx DESTINATION ./dt)