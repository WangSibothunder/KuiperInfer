// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-12-25.
#include "cat.hpp"
#include <cstring>
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
CatLayer::CatLayer(int32_t dim) : NonParamLayer("cat"), dim_(dim) {}

StatusCode CatLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                             std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  StatusCode status_code = Check(inputs, outputs);
  if (status_code != StatusCode::kSuccess) {
    return status_code;
  }

  const uint32_t output_size = outputs.size();
  const uint32_t packet_size = inputs.size() / output_size;
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    std::vector<std::shared_ptr<Tensor<float>>> packet_inputs;
    packet_inputs.reserve(packet_size);
    for (uint32_t j = i; j < inputs.size(); j += output_size) {
      packet_inputs.push_back(inputs.at(j));
    }

    auto infer_axis = [&](const std::shared_ptr<Tensor<float>>& out) -> int {
      if (packet_inputs.empty()) {
        return -1;
      }
      auto norm_shapes = [](const std::vector<uint32_t>& s,
                            std::array<uint32_t, 3>& out) -> bool {
        if (s.size() == 3) {
          out = {s[0], s[1], s[2]};
          return true;
        }
        if (s.size() == 2) {
          out = {1, s[0], s[1]};
          return true;
        }
        if (s.size() == 1) {
          out = {1, 1, s[0]};
          return true;
        }
        return false;
      };

      std::array<uint32_t, 3> base_shapes{};
      if (!packet_inputs[0] || !norm_shapes(packet_inputs[0]->raw_shapes(), base_shapes)) {
        return -1;
      }

      std::vector<std::array<uint32_t, 3>> input_shapes;
      input_shapes.reserve(packet_inputs.size());
      for (const auto& in : packet_inputs) {
        if (!in) {
          return -1;
        }
        std::array<uint32_t, 3> s{};
        if (!norm_shapes(in->raw_shapes(), s)) {
          return -1;
        }
        input_shapes.push_back(s);
      }

      std::vector<size_t> diff_axes;
      for (size_t k = 0; k < 3; ++k) {
        for (const auto& s : input_shapes) {
          if (s[k] != base_shapes[k]) {
            diff_axes.push_back(k);
            break;
          }
        }
      }
      if (diff_axes.size() == 1) {
        return static_cast<int>(diff_axes.front());
      }

      std::array<uint32_t, 3> sum = base_shapes;
      for (size_t k = 0; k < 3; ++k) {
        uint32_t total = 0;
        for (const auto& s : input_shapes) {
          total += s[k];
        }
        sum[k] = total;
      }

      std::array<uint32_t, 3> out_shapes{};
      const bool has_out_shapes =
          (out && !out->empty()) && norm_shapes(out->raw_shapes(), out_shapes);

      for (size_t k = 0; k < 3; ++k) {
        bool other_equal = true;
        for (const auto& s : input_shapes) {
          for (size_t d = 0; d < 3; ++d) {
            if (d == k) continue;
            if (s[d] != base_shapes[d]) {
              other_equal = false;
              break;
            }
          }
          if (!other_equal) break;
        }
        if (!other_equal) continue;

        if (has_out_shapes) {
          if (out_shapes[k] == sum[k] && out_shapes[(k + 1) % 3] == base_shapes[(k + 1) % 3] &&
              out_shapes[(k + 2) % 3] == base_shapes[(k + 2) % 3]) {
            return static_cast<int>(k);
          }
        }
      }

      for (size_t k = 0; k < 3; ++k) {
        if (sum[k] != base_shapes[k]) {
          bool other_equal = true;
          for (size_t d = 0; d < 3; ++d) {
            if (d == k) continue;
            if (sum[d] != base_shapes[d]) {
              other_equal = false;
              break;
            }
          }
          if (other_equal) {
            return static_cast<int>(k);
          }
        }
      }

      return -1;
    };

    // Special path for rank<=2 tensors on dim=1 (e.g. DeiT class token cat).
    // Keep C-order linear layout for downstream reshape/permute/select kernels.
    bool low_rank_dim1 = (dim_ == 1);
    for (const auto& in : packet_inputs) {
      if (!in) {
        low_rank_dim1 = false;
        break;
      }
      const auto& s = in->raw_shapes();
      if (s.empty() || s.size() > 3) {
        low_rank_dim1 = false;
        break;
      }
      if (s.size() == 3 && !(s[0] == 1 && s[1] == 1)) {
        low_rank_dim1 = false;
        break;
      }
    }
    if (low_rank_dim1) {
      bool has_single_row = false;
      bool has_multi_row = false;
      for (const auto& in : packet_inputs) {
        const auto& s = in->raw_shapes();
        uint32_t rows = 1;
        if (s.size() == 2) {
          rows = s[0];
        } else if (s.size() == 3) {
          rows = s[1];
        }
        has_single_row = has_single_row || (rows == 1);
        has_multi_row = has_multi_row || (rows > 1);
      }
      if (!(has_single_row && has_multi_row)) {
        low_rank_dim1 = false;
      }
    }

    if (low_rank_dim1) {
      uint32_t out_rows = 0;
      uint32_t out_cols = 0;
      for (size_t idx = 0; idx < packet_inputs.size(); ++idx) {
        const auto& in = packet_inputs[idx];
        const auto& s = in->raw_shapes();
        uint32_t rows = 1;
        uint32_t cols = 1;
        if (s.size() == 1) {
          rows = 1;
          cols = s[0];
        } else if (s.size() == 2) {
          rows = s[0];
          cols = s[1];
        } else {  // size == 3 and s[0] == 1
          rows = s[1];
          cols = s[2];
        }
        if (idx == 0) {
          out_cols = cols;
        } else {
          CHECK_EQ(cols, out_cols);
        }
        out_rows += rows;
      }

      if (output == nullptr || output->empty() ||
          output->size() != static_cast<size_t>(out_rows) * out_cols) {
        output = std::make_shared<Tensor<float>>(1, out_rows, out_cols);
        outputs.at(i) = output;
      }
      output->Reshape({out_rows, out_cols});

      float* out_ptr = output->raw_ptr();
      uint32_t row_offset = 0;
      for (const auto& in : packet_inputs) {
        const auto& s = in->raw_shapes();
        uint32_t rows = 1;
        uint32_t cols = 1;
        if (s.size() == 1) {
          rows = 1;
          cols = s[0];
        } else if (s.size() == 2) {
          rows = s[0];
          cols = s[1];
        } else {  // size == 3 and s[0] == 1
          rows = s[1];
          cols = s[2];
        }
        const float* in_ptr = in->raw_ptr();
        for (uint32_t r = 0; r < rows; ++r) {
          std::memcpy(out_ptr + (row_offset + r) * out_cols, in_ptr + r * cols,
                      sizeof(float) * cols);
        }
        row_offset += rows;
      }
      continue;
    }

    const int axis = infer_axis(output);
    if (axis == 1 || axis == 2) {
      const auto& first = packet_inputs.front();
      const uint32_t in_channels = first->channels();
      const uint32_t in_rows = first->rows();
      const uint32_t in_cols = first->cols();

      uint32_t total_rows = 0;
      uint32_t total_cols = 0;
      if (axis == 1) {
        for (const auto& in : packet_inputs) {
          CHECK(in->channels() == in_channels && in->cols() == in_cols);
          total_rows += in->rows();
        }
        if (output == nullptr || output->empty()) {
          output = std::make_shared<Tensor<float>>(in_channels, total_rows, in_cols);
          outputs.at(i) = output;
        }
        if (output->channels() != in_channels || output->rows() != total_rows ||
            output->cols() != in_cols) {
          output = std::make_shared<Tensor<float>>(in_channels, total_rows, in_cols);
          outputs.at(i) = output;
        }

        for (uint32_t c = 0; c < in_channels; ++c) {
          uint32_t row_offset = 0;
          for (const auto& in : packet_inputs) {
            const uint32_t rows = in->rows();
            output->slice(c).submat(row_offset, 0, row_offset + rows - 1, in_cols - 1) =
                in->slice(c);
            row_offset += rows;
          }
        }
      } else {  // axis == 2
        for (const auto& in : packet_inputs) {
          CHECK(in->channels() == in_channels && in->rows() == in_rows);
          total_cols += in->cols();
        }
        if (output == nullptr || output->empty()) {
          output = std::make_shared<Tensor<float>>(in_channels, in_rows, total_cols);
          outputs.at(i) = output;
        }
        if (output->channels() != in_channels || output->rows() != in_rows ||
            output->cols() != total_cols) {
          output = std::make_shared<Tensor<float>>(in_channels, in_rows, total_cols);
          outputs.at(i) = output;
        }

        for (uint32_t c = 0; c < in_channels; ++c) {
          uint32_t col_offset = 0;
          for (const auto& in : packet_inputs) {
            const uint32_t cols = in->cols();
            output->slice(c).submat(0, col_offset, in_rows - 1, col_offset + cols - 1) =
                in->slice(c);
            col_offset += cols;
          }
        }
      }
    } else {
      uint32_t copy_channel_offset = 0;
      for (const auto& input : packet_inputs) {
        const uint32_t in_rows = input->rows();
        const uint32_t in_cols = input->cols();
        const uint32_t in_channels = input->channels();

        if (output == nullptr || output->empty()) {
          output = std::make_shared<Tensor<float>>(in_channels * packet_size, in_rows, in_cols);
          outputs.at(i) = output;
        }
        if (output->channels() != in_channels * packet_size || output->rows() != in_rows ||
            output->cols() != in_cols) {
          output = std::make_shared<Tensor<float>>(in_channels * packet_size, in_rows, in_cols);
          outputs.at(i) = output;
        }

        const uint32_t plane_size = in_rows * in_cols;
        memcpy(output->raw_ptr(copy_channel_offset * plane_size), input->raw_ptr(),
               sizeof(float) * plane_size * in_channels);
        copy_channel_offset += input->channels();
      }
    }
  }
  return StatusCode::kSuccess;
}

StatusCode CatLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                    std::shared_ptr<Layer<float>>& cat_layer) {
  if (!op) {
    LOG(ERROR) << "The cat operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the cat layer is empty.";
    return StatusCode::kParseParamError;
  }

  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Can not find the dim parameter";
    return StatusCode::kParseParamError;
  }

  auto dim_param = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("dim"));
  if (!dim_param) {
    LOG(ERROR) << "Can not find the dim parameter";
    return StatusCode::kParseParamError;
  }
  const int32_t dim = dim_param->value;
  cat_layer = std::make_shared<CatLayer>(dim);
  return StatusCode::kSuccess;
}

StatusCode CatLayer::Check(const std::vector<sftensor>& inputs,
                           const std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the cat layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  for (const auto& input_data : inputs) {
    if (input_data == nullptr || inputs.empty()) {
      return StatusCode::kInferInputsEmpty;
    }
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the cat layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (dim_ != 1 && dim_ != -3) {
    LOG(ERROR) << "The dimension parameter of cat layer is error";
    return StatusCode::kInferParamError;
  }

  const uint32_t output_size = outputs.size();
  if (inputs.size() % output_size != 0) {
    LOG(ERROR) << "The input and output tensor array size of cat layer do not match";
    return StatusCode::kInferDimMismatch;
  }
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kCatCreateInstance(CatLayer::CreateInstance, "torch.cat");
}  // namespace kuiper_infer
