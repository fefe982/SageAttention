#pragma once
#include <torch/types.h>

void scaled_fp4_quant(torch::Tensor const& input,
                      torch::Tensor const& output,
                      torch::Tensor const& output_sf,
                      int tensor_layout);

void scaled_fp4_quant_permute(torch::Tensor const& input,
                              torch::Tensor const& output,
                              torch::Tensor const& output_sf,
                              int tensor_layout);

void scaled_fp4_quant_trans(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout);
