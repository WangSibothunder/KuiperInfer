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

// Created by fss on 22-11-15.
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {

const std::vector<std::shared_ptr<Tensor<float>>>& Layer<float>::weights() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

const std::vector<std::shared_ptr<Tensor<float>>>& Layer<float>::bias() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer<float>::set_bias(const std::vector<float>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer<float>::set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer<float>::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer<float>::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

StatusCode Layer<float>::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
  return StatusCode::kFunctionNotImplement;
}

StatusCode Layer<float>::Forward() {
  LOG_IF(FATAL, this->runtime_operator_.expired()) << "Runtime operator is expired or nullptr";
  const auto& runtime_operator = this->runtime_operator_.lock();
  std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;
  // 记录每个输入 tensor 对应的 operand 名称和 batch 索引，方便调试
  std::vector<std::pair<std::string, size_t>> layer_input_meta;
  const bool unbind_single_input_mode = (runtime_operator->type == "torch.unbind");
  for (size_t idx = 0; idx < runtime_operator->input_operands_seq.size(); ++idx) {
    const auto& input_operand_data = runtime_operator->input_operands_seq[idx];
    if (input_operand_data == nullptr) {
      LOG(ERROR) << runtime_operator->name << " input_operand[" << idx << "] is nullptr";
      return StatusCode::kInferInputsEmpty;
    }
    if (input_operand_data->datas.empty()) {
      LOG(ERROR) << runtime_operator->name << " input_operand[" << idx
                 << "] datas is empty, producer=" << input_operand_data->name;
      return StatusCode::kInferInputsEmpty;
    }
    if (unbind_single_input_mode) {
      // torch.unbind 在 DeiT 路径上是单输入多输出；当上游误把首维当成 batch 时，
      // datas 可能被扩成多个槽位。这里仅取第一个非空输入，避免把它错误展平成多 batch。
      size_t selected_index = 0;
      for (size_t b = 0; b < input_operand_data->datas.size(); ++b) {
        if (input_operand_data->datas[b] != nullptr && !input_operand_data->datas[b]->empty()) {
          selected_index = b;
          break;
        }
      }
      layer_input_datas.push_back(input_operand_data->datas[selected_index]);
      layer_input_meta.emplace_back(input_operand_data->name, selected_index);
      continue;
    }
    for (size_t b = 0; b < input_operand_data->datas.size(); ++b) {
      layer_input_datas.push_back(input_operand_data->datas[b]);
      layer_input_meta.emplace_back(input_operand_data->name, b);
    }
  }

  if (layer_input_datas.empty()) {
    LOG(ERROR) << runtime_operator->name << " Layer input data is empty";
    return StatusCode::kInferInputsEmpty;
  }

  for (size_t i = 0; i < layer_input_datas.size(); ++i) {
    const sftensor& layer_input_data = layer_input_datas[i];
    if (layer_input_data == nullptr || layer_input_data->empty()) {
      const auto& meta = (i < layer_input_meta.size()) ? layer_input_meta[i]
                                                       : std::make_pair(std::string("unknown"), size_t(0));
      LOG(ERROR) << runtime_operator->name << " layer input tensor[" << i
                 << "] is empty, from operand=" << meta.first
                 << ", batch=" << meta.second;
      return StatusCode::kInferInputsEmpty;
    }
  }

  const std::shared_ptr<RuntimeOperand>& output_operand_datas = runtime_operator->output_operands;
  if (output_operand_datas == nullptr || output_operand_datas->datas.empty()) {
    LOG(ERROR) << "Layer output data is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  StatusCode status =
      runtime_operator->layer->Forward(layer_input_datas, output_operand_datas->datas);
  if (status != StatusCode::kSuccess) {
    LOG(ERROR) << "Forward the layer " << runtime_operator->name << " get a error status";
  }
  return status;
}

StatusCode Layer<float>::Check(const std::vector<sftensor>& inputs,
                               const std::vector<sftensor>& outputs) {
  return StatusCode::kFunctionNotImplement;
}

void Layer<float>::set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  CHECK(runtime_operator != nullptr);
  this->runtime_operator_ = runtime_operator;
}

}  // namespace kuiper_infer
