// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#include "reshape.hpp"
#include <atomic>
#include <numeric>
#include <sstream>
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
  static std::atomic<uint32_t> reshape_call_count{0};
  const bool debug_this = (reshape_call_count.fetch_add(1) == 0);

  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    CHECK(input != nullptr && !input->empty()) << "The input tensor at index " << b << " is empty";

    // 1. 计算总元素数量
    const size_t total_elements = input->size();
    const std::vector<uint32_t>& input_raw_shapes = input->raw_shapes();
    const int input_rank = static_cast<int>(input_raw_shapes.size());
    const int target_rank = static_cast<int>(target_shapes_.size());
    
    // 2. 解析目标形状 (处理 -1 和 0)
    std::vector<uint32_t> final_shapes;
    int32_t infer_idx = -1;
    size_t current_size = 1;

    auto resolve_zero_dim = [&](int index) -> uint32_t {
      if (input_raw_shapes.empty()) {
        return 1;
      }
      if (input_rank == target_rank) {
        return input_raw_shapes.at(index);
      }
      if (input_rank + 1 == target_rank) {
        if (index == 0) {
          return 1;
        }
        const int in_idx = index - 1;
        if (in_idx >= 0 && in_idx < input_rank) {
          return input_raw_shapes.at(in_idx);
        }
        return 1;
      }
      const int in_idx = index - target_rank + input_rank;
      if (in_idx >= 0 && in_idx < input_rank) {
        return input_raw_shapes.at(in_idx);
      }
      return 1;
    };

    for (size_t i = 0; i < target_shapes_.size(); ++i) {
      int32_t dim = target_shapes_[i];
      if (dim == -1) {
        CHECK_EQ(infer_idx, -1) << "Reshape can only have one -1 dimension";
        infer_idx = static_cast<int32_t>(i);
        final_shapes.push_back(0);
      } else {
        uint32_t resolved_dim = 1;
        if (dim == 0) {
          resolved_dim = resolve_zero_dim(static_cast<int>(i));
        } else {
          CHECK_GT(dim, 0) << "Invalid reshape dim: " << dim;
          resolved_dim = static_cast<uint32_t>(dim);
        }
        final_shapes.push_back(resolved_dim);
        current_size *= resolved_dim;
      }
    }

    // 3. 填充推断的维度 (-1)
    if (infer_idx != -1) {
      CHECK_EQ(total_elements % current_size, 0)
          << "Total elements " << total_elements
          << " not divisible by known dimensions size " << current_size;
      final_shapes[infer_idx] = static_cast<uint32_t>(total_elements / current_size);
    } else {
      CHECK_EQ(total_elements, current_size)
          << "Total elements " << total_elements << " does not match target shape size " << current_size;
    }

    // 4. 准备输出 Tensor
    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty() || output->size() != total_elements) {
      // 先创建一个临时的 1D/2D Tensor 容纳数据
      output = std::make_shared<Tensor<float>>(1, static_cast<uint32_t>(total_elements), 1);
      outputs.at(b) = output;
    }

    if (debug_this) {
      std::ostringstream target_ss;
      target_ss << "(";
      for (size_t si = 0; si < final_shapes.size(); ++si) {
        if (si > 0) {
          target_ss << ",";
        }
        target_ss << final_shapes[si];
      }
      target_ss << ")";
      LOG(INFO) << ">>> [ReshapeDebugPre] input_size=" << total_elements
                << " output_size=" << output->size()
                << " target=" << target_ss.str();
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
    if (debug_this) {
      std::ostringstream shape_stream;
      shape_stream << "(";
      for (size_t si = 0; si < final_shapes.size(); ++si) {
        if (si > 0) {
          shape_stream << ",";
        }
        shape_stream << final_shapes[si];
      }
      shape_stream << ")";
      LOG(INFO) << ">>> [ReshapeDebug] in_raw_rank=" << input_raw_shapes.size()
                << " out_raw=" << shape_stream.str()
                << " size=" << output->size();
    }
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
