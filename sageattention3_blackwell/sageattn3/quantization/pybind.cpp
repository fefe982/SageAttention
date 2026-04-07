// Pybind11 bindings for fp4 quantization (compiled by MSVC, not nvcc)
#include <torch/all.h>
#include <torch/python.h>
#include "fp4_quantization_4d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scaled_fp4_quant", &scaled_fp4_quant);
  m.def("scaled_fp4_quant_permute", &scaled_fp4_quant_permute);
  m.def("scaled_fp4_quant_trans", &scaled_fp4_quant_trans);
}
