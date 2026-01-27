// MIT License
// Copyright (c) 2022 - 傅莘莘
// Modified by WangSibo 2026: Fix Batch Mismatch Reuse & Dynamic Shape

#include "runtime/runtime_op.hpp"
#include "data/tensor_util.hpp"
#include <numeric>

namespace kuiper_infer {

// ---------------------------------------------------------
// [FIX] 辅助函数：创建 Tensor
// ---------------------------------------------------------
static sftensor CreateTensor(const std::vector<int32_t>& operand_shapes) {
  if (operand_shapes.empty()) {
    return nullptr;
  }
  
  uint32_t total_size = 1;
  // 跳过 Batch 维度 (index 0)
  for (size_t i = 1; i < operand_shapes.size(); ++i) {
      if (operand_shapes[i] <= 0) {
          continue; // 忽略动态维度
      }
      total_size *= static_cast<uint32_t>(operand_shapes[i]);
  }
  
  return TensorCreate<float>(1, 1, total_size);
}

// ---------------------------------------------------------
// [FIX] 辅助函数：Reshape (安全版)
// ---------------------------------------------------------
static void CheckAndReshapeTensor(sftensor& output_tensor,
                                  const std::vector<int32_t>& operand_shapes) {
  if (!output_tensor) return;

  // 1. 检查动态形状 (-1)
  for (const auto& dim : operand_shapes) {
      if (dim < 0) return; // 跳过 Reshape
  }
  
  // 2. 构造目标形状 (剔除 Batch)
  std::vector<uint32_t> target_shapes;
  for (size_t i = 1; i < operand_shapes.size(); ++i) {
      target_shapes.push_back(static_cast<uint32_t>(operand_shapes[i]));
  }
  
  if (target_shapes.empty()) {
      target_shapes.push_back(1);
  }

  // 3. 执行 Reshape
  // 如果这里依然报错，说明 InitOperatorOutput 的复用筛选没拦住，但本次修复应该已解决
  output_tensor->Reshape(target_shapes);
}

// ---------------------------------------------------------
// InitOperatorInput
// ---------------------------------------------------------
void RuntimeOperatorUtils<float>::InitOperatorInput(
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
  if (operators.empty()) {
    LOG(ERROR) << "Operators for init input shapes is empty!";
    return;
  }

  for (const auto& op : operators) {
    if (op->input_operands.empty()) {
      continue;
    } else {
      const std::map<std::string, std::shared_ptr<RuntimeOperand>>& input_operands_map =
          op->input_operands;
      for (const auto& [_, input_operand] : input_operands_map) {
        if (!input_operand) {
          continue;
        }
        const auto& type = input_operand->type;
        auto& input_datas = input_operand->datas;
        CHECK(type == RuntimeDataType::kTypeFloat32) << "The graph only support float32 yet!";
        const auto& input_operand_shape = input_operand->shapes;

        CHECK(!input_operand_shape.empty());
        const int32_t batch = input_operand_shape.at(0);
        CHECK(batch > 0) << "Dynamic batch size is not supported!";
        
        // 5D check
        CHECK(input_operand_shape.size() >= 2 && input_operand_shape.size() <= 5)
            << "Unsupported tensor shape sizes: " << input_operand_shape.size();

        if (!input_datas.empty()) {
          CHECK_EQ(input_datas.size(), batch);
        } else {
          input_datas.resize(batch);
        }
      }
    }
  }
}

// ---------------------------------------------------------
// InitOperatorOutput (支持多输出算子，如 unbind)
// ---------------------------------------------------------
void RuntimeOperatorUtils<float>::InitOperatorOutput(
    const std::vector<pnnx::Operator*>& pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
  CHECK(!pnnx_operators.empty() && !operators.empty() && pnnx_operators.size() == operators.size());
  CHECK(pnnx_operators.size() == operators.size());
  
  for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
    // 获取当前算子的所有输出操作数
    const std::vector<pnnx::Operand*> operands = pnnx_operators[i]->outputs;
    if (operands.empty()) continue;
    
    // [FIX] 移除 "Only support one node one output" 的报错
    // if (operands.size() > 1) { LOG(FATAL) ... }

    // 使用第一个输出作为主参考 (Shape/Type)
    pnnx::Operand* operand = operands.front();
    CHECK(operand != nullptr && !operand->shape.empty()) << "Operand output is null or empty!";
    
    std::vector<int32_t> operand_shapes;
    std::copy_if(operand->shape.begin(), operand->shape.end(), std::back_inserter(operand_shapes),
                 [](int32_t dim) { return dim > 0; });

    const auto& runtime_op = operators[i];
    auto& output_tensors = runtime_op->output_operands;
    
    CHECK(operand_shapes.size() >= 2 && operand_shapes.size() <= 5)
        << "Unsupported shape sizes: " << operand_shapes.size();

    size_t operand_size =
        std::accumulate(operand_shapes.begin(), operand_shapes.end(), 1, std::multiplies<size_t>());

    const int32_t batch = operand_shapes[0];
    CHECK_EQ(operand->type, 1) << "The type of pnnx operand is not float32";

    if (!output_tensors) {
      // -------------------------------------------
      // 内存复用逻辑 (Memory Reuse Logic)
      // [FIX] 仅针对单输出算子启用复用，多输出算子(如 unbind) 比较复杂，直接分配新内存更安全
      // -------------------------------------------
      bool has_found = false;
      
      if (operands.size() == 1) { 
          for (uint32_t j = 0; j < i; ++j) {
            if (has_found) break;

            const auto& prev_runtime_op = operators.at(j);
            if (!prev_runtime_op->output_operands || prev_runtime_op->occur_end_time != -1) {
              continue;
            }
            if (runtime_op->start_time > prev_runtime_op->occur_end_time) {
              prev_runtime_op->occur_end_time = -1;
            }

            if (runtime_op->start_time > prev_runtime_op->end_time) {
              // 复用检查
              if (prev_runtime_op->output_operands->size() == operand_size) {
                 // 物理检查
                 if (!prev_runtime_op->output_operands->datas.empty()) {
                     if (prev_runtime_op->output_operands->datas.size() != batch) continue;
                     size_t physical_size = prev_runtime_op->output_operands->datas[0]->size();
                     if (physical_size != operand_size) continue;
                 }

                has_found = true;
                const auto& prev_output_operand = prev_runtime_op->output_operands;
                runtime_op->output_operands = std::make_shared<RuntimeOperand>(
                    prev_output_operand->name + "_output", operand_shapes, batch,
                    RuntimeDataType::kTypeFloat32);
                
                const auto& prev_runtime_op_tensors = prev_output_operand->datas;
                for (uint32_t b = 0; b < batch; ++b) {
                  sftensor prev_output_tensor = prev_runtime_op_tensors.at(b);
                  sftensor output_tensor = std::make_shared<ftensor>(prev_output_tensor->raw_ptr(),
                                                                     prev_output_tensor->shapes());
                  CheckAndReshapeTensor(output_tensor, operand_shapes);
                  output_tensors->datas[b] = output_tensor;
                }
                prev_runtime_op->occur_end_time = runtime_op->end_time;
              }
            }
          }
      }

      // -------------------------------------------
      // 分配新内存 (支持多输出)
      // -------------------------------------------
      if (!has_found) {
        std::vector<sftensor> output_operand_datas;
        
        // [FIX] 遍历所有 outputs，为每一个分配空间
        // 注意：这里假设所有输出的 shape 是一样的 (对于 unbind/split 通常是成立的)
        // 如果 shape 不一样，RuntimeOperand 目前的结构只能存一个 shape，这可能是一个限制
        // 但对于 DeiT 的 unbind (QKV) 来说，三个输出 shape 是一致的。
        for (size_t k = 0; k < operands.size(); ++k) {
            for (uint32_t j = 0; j < batch; ++j) {
              output_operand_datas.push_back(CreateTensor(operand_shapes));
            }
        }
        
        runtime_op->output_operands =
            std::make_shared<RuntimeOperand>(operand->name + "_output", operand_shapes,
                                             output_operand_datas, RuntimeDataType::kTypeFloat32);
      }
    } else {
      // output_tensors 已存在的情况
      // 假设 Attribute 节点不会有多输出
      CHECK(batch == output_tensors->datas.size());
      CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
      for (uint32_t b = 0; b < batch; ++b) {
        sftensor output_tensor = output_tensors->datas[b];
        CheckAndReshapeTensor(output_tensor, operand_shapes);
      }
    }
  }
}

}  // namespace kuiper_infer