// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#include "reshape.hpp"
#include <numeric>
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

ReshapeLayer::ReshapeLayer(std::vector<int32_t> target_shapes)
    : ParamLayer("Tensor.reshape"), target_shapes_(std::move(target_shapes)) {}

StatusCode ReshapeLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the Reshape layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the Reshape layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  const uint32_t batch_size = inputs.size();

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    CHECK(input != nullptr && !input->empty()) << "The input tensor at index " << b << " is empty";

    // 1. 计算总元素数量
    const uint32_t total_elements = input->size();
    
    // 2. 解析目标形状 (处理 -1 和 0)
    std::vector<uint32_t> final_shapes;
    int32_t infer_idx = -1;
    uint32_t current_size = 1;

    for (size_t i = 0; i < target_shapes_.size(); ++i) {
        int32_t dim = target_shapes_[i];
        if (dim == -1) {
            CHECK_EQ(infer_idx, -1) << "Reshape can only have one -1 dimension";
            infer_idx = i;
            final_shapes.push_back(0); // 占位
        } else {
            if (dim == 0) {
                // PNNX 语义：0 意味着 copy input dim (通常用于 batch)，暂简化处理
                // 如果遇到，通常意味着我们需要参考 input->shapes()[i]
                // 这里假设 PNNX 导出的都是静态形状或非 0
                dim = 1; 
            }
            final_shapes.push_back(dim);
            current_size *= dim;
        }
    }

    // 3. 填充推断的维度 (-1)
    if (infer_idx != -1) {
        CHECK_EQ(total_elements % current_size, 0) 
            << "Total elements " << total_elements << " not divisible by known dimensions size " << current_size;
        final_shapes[infer_idx] = total_elements / current_size;
    } else {
        CHECK_EQ(total_elements, current_size)
            << "Total elements " << total_elements << " does not match target shape size " << current_size;
    }

    // 4. 准备输出 Tensor
    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      // 先创建一个临时的 1D/2D Tensor 容纳数据
      output = std::make_shared<Tensor<float>>(1, total_elements, 1);
      outputs.at(b) = output;
    }
    
    // 5. 核心步骤：重设形状 (Reshape)
    // 注意：先 Reshape 确定维度和内存结构，再拷贝数据，防止 Reshape 重置数据
    // Tensor::Reshape 负责更新 raw_shapes_ 和底层的 Armadillo Cube 维度
    output->Reshape(final_shapes);

    // 6. 数据拷贝
    // 假设内存是连续的 (Armadillo 是列主序连续，只要我们不改变数据的相对顺序，memcpy 是安全的)
    // 检查尺寸一致性
    CHECK_EQ(input->size(), output->size());
    
    // 执行拷贝
    memcpy(output->raw_ptr(), input->raw_ptr(), total_elements * sizeof(float));
  }
  return StatusCode::kSuccess;
}

StatusCode ReshapeLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& layer) {
  if (!op) {
    return StatusCode::kParseNullOperator;
  }

  std::vector<int32_t> target_shapes;
  if (op->params.find("shape") != op->params.end()) {
    auto param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(op->params.at("shape"));
    if (param) {
        target_shapes = param->value;
    }
  }

  layer = std::make_shared<ReshapeLayer>(target_shapes);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kReshapeCreateInstance(ReshapeLayer::CreateInstance, "Tensor.reshape");

}  // namespace kuiper_infer