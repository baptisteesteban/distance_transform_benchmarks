#include <pybind11/pybind11.h>

PYBIND11_MODULE(_dt_cxx, m)
{
  namespace py = pybind11;

  m.def("test", []() { py::print("Test"); });
}