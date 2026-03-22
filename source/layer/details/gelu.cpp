// MIT License
// Copyright (c) 2022 - 王思博
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified by User for DeiT-Tiny support

#include "gelu.hpp"
#include <cmath>
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

GELULayer::GELULayer() : Layer("GELU") {}

StatusCode GELULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the GELU layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the GELU layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the GELU layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  const uint32_t batch_size = inputs.size();

  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    CHECK(input != nullptr && !input->empty()) << "The input tensor at index " << b << " is empty";

    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty() || output->size() != input->size()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(b) = output;
    }

    CHECK(output->shapes() == input->shapes()) 
        << "The input and output tensor shapes of the GELU layer do not match at index " << b;

    // GELU 是逐元素的，所以我们直接遍历所有元素即可，不用关心维度
    // 使用 transform 配合 lambda 或者直接循环
    // 公式: 0.5 * x * (1 + erf(x / sqrt(2)))
    // 预计算常量 sqrt(2)
    const float kSqrt2 = std::sqrt(2.0f);
    
    // 获取每个 Tensor 的数据量
    uint32_t element_size = input->size();
    
    // 如果 Tensor 数据量很大，这里也可以开二级并行，或者让外层 Batch 并行即可
    // 对于 DeiT-Tiny, size 约为 197*192 ≈ 3.8w，单线程处理足矣
    for (uint32_t i = 0; i < element_size; ++i) {
      float x = input->index(i);
      
      // 标准 GELU 实现
      // 注意：erf 是 C++11 标准库函数
      float y = 0.5f * x * (1.0f + std::erf(x / kSqrt2));
      
      output->index(i) = y;
    }
  }
  return StatusCode::kSuccess;
}

StatusCode GELULayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& gelu_layer) {
  if (!op) {
    LOG(ERROR) << "The operator parameter in the GELU layer is null";
    return StatusCode::kParseNullOperator;
  }
  
  // GELU 通常没有可配置参数 (除了近似版本，但这里默认标准版)
  gelu_layer = std::make_shared<GELULayer>();
  return StatusCode::kSuccess;
}

// 注册算子：对应 PNNX 导出的类型 "GELU"
LayerRegistererWrapper kGELUCreateInstance(GELULayer::CreateInstance, "nn.GELU");

}  // namespace kuiper_infer
