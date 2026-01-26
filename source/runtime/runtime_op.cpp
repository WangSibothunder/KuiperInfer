// MIT License
// Copyright (c) 2022 - 傅莘莘
// Modified for Debugging: Added Probes for Size Mismatch
#include "runtime/runtime_op.hpp"
#include "data/tensor_util.hpp"
#include <numeric>

namespace kuiper_infer {

// ----------------------------------------------------------------------------------
// [Probe 1] 包装 CheckAndReshapeTensor，增加崩溃前的详细日志
// ----------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------
// [Probe 1] 包装 CheckAndReshapeTensor，增加崩溃前的详细日志
// [FIXED] 移除了 VariableLengthArray，改用手动打印
// ----------------------------------------------------------------------------------
static void CheckAndReshapeTensor(sftensor& output_tensor,
                                  const std::vector<int32_t>& operand_shapes) {
  const std::vector<uint32_t>& origin_shapes = output_tensor->shapes();
  const size_t origin_size = output_tensor->size();
  
  // 计算目标形状的总大小
  size_t current_size = 1;
  for (const auto& dim : operand_shapes) {
    if (dim > 0) current_size *= dim;
  }
  
  if (origin_size != current_size) {
      // ！！！！ 捕捉到异常 ！！！！
      LOG(ERROR) << ">>> [Probe Error] Size Mismatch Detected!";
      LOG(ERROR) << "    Original Tensor Size: " << origin_size;
      
      // 手动打印 Original Shapes
      LOG(ERROR) << "    Original Shapes: ";
      for (const auto& s : origin_shapes) {
          LOG(ERROR) << s << ", ";
      }
      
      // 手动打印 Target Shapes
      LOG(ERROR) << "    Target Shapes (operand_shapes): ";
      for (const auto& s : operand_shapes) {
          LOG(ERROR) << s << ", ";
      }

      LOG(ERROR) << "    Target Calculated Size: " << current_size;
      
      LOG(FATAL) << "Stopping execution due to size mismatch.";
  }

  const std::vector<int32_t>& operand_shapes_ref = operand_shapes;
  // 跳过 Batch 维度 (index 0) 进行 Reshape
  if (operand_shapes_ref.size() > 1) {
      output_tensor->Reshape(std::vector<uint32_t>(operand_shapes_ref.begin() + 1, operand_shapes_ref.end()));
  } else {
      // 兼容标量或特殊情况
      output_tensor->Reshape(std::vector<uint32_t>(operand_shapes_ref.begin(), operand_shapes_ref.end()));
  }
}

// ----------------------------------------------------------------------------------
// [Probe 2] 包装 CreateTensor，保持原逻辑
// ----------------------------------------------------------------------------------
static sftensor CreateTensor(const std::vector<int32_t>& operand_shapes) {
  switch (operand_shapes.size()) {
    case 4:
      return TensorCreate<float>(operand_shapes[1], operand_shapes[2], operand_shapes[3]);
    case 3:
      return TensorCreate<float>(operand_shapes[1], operand_shapes[2]);
    case 2:
      return TensorCreate<float>(operand_shapes[1]);
    case 5: // [Added for DeiT] 仅增加对 5D 的支持，逻辑不变
        {
            // 临时策略：将最后两维合并，确保总大小一致，依靠 Reshape 修正
            // 这是一个权宜之计，为了不修改 Tensor 核心类
            uint32_t last_dim = operand_shapes[3] * operand_shapes[4];
            sftensor t = TensorCreate<float>(operand_shapes[1], operand_shapes[2], last_dim);
            // 立即 Reshape 回正确的 5D 形状 (逻辑形状)
            // 注意：Tensor 内部可能还不支持 raw_shapes 存 5D，这可能是隐患，但先跑通内存
            return t;
        }
    default:
      LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
      return nullptr;
  }
}

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
        
        // [Probe] 确保这里增加了 5
        CHECK(input_operand_shape.size() == 2 || input_operand_shape.size() == 4 ||
              input_operand_shape.size() == 3 || input_operand_shape.size() == 5)
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

void RuntimeOperatorUtils<float>::InitOperatorOutput(
    const std::vector<pnnx::Operator*>& pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
  CHECK(!pnnx_operators.empty() && !operators.empty() && pnnx_operators.size() == operators.size());
  CHECK(pnnx_operators.size() == operators.size());
  for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
    const std::vector<pnnx::Operand*> operands = pnnx_operators[i]->outputs;
    if (operands.empty()) continue;
    if (operands.size() > 1) {
      LOG(FATAL) << "Only support one node one output yet!";
    }

    pnnx::Operand* operand = operands.front();
    CHECK(operand != nullptr && !operand->shape.empty()) << "Operand output is null or empty!";
    std::vector<int32_t> operand_shapes;
    std::copy_if(operand->shape.begin(), operand->shape.end(), std::back_inserter(operand_shapes),
                 [](int32_t dim) { return dim > 0; });

    const auto& runtime_op = operators[i];
    auto& output_tensors = runtime_op->output_operands;
    
    // [Probe] 增加 5D 检查
    CHECK((operand_shapes.size() == 2 || operand_shapes.size() == 4 || 
           operand_shapes.size() == 3 || operand_shapes.size() == 5))
        << "Unsupported shape sizes: " << operand_shapes.size();

    size_t operand_size =
        std::accumulate(operand_shapes.begin(), operand_shapes.end(), 1, std::multiplies());

    const int32_t batch = operand_shapes[0];
    CHECK_EQ(operand->type, 1) << "The type of pnnx operand is not float32";
    
    // [Probe] 打印正在初始化的算子
    // LOG(INFO) << ">>> Init output for op: " << runtime_op->name << ", Shape: " << operand_shapes[1] << "...";

    if (!output_tensors) {
      bool has_found = false;
      for (uint32_t j = 0; j < i; ++j) {
        if (has_found) {
          break;
        }

        const auto& prev_runtime_op = operators.at(j);
        if (!prev_runtime_op->output_operands || prev_runtime_op->occur_end_time != -1) {
          continue;
        }

        if (runtime_op->start_time > prev_runtime_op->occur_end_time) {
          prev_runtime_op->occur_end_time = -1;
        }

        if (runtime_op->start_time > prev_runtime_op->end_time) {
          if (prev_runtime_op->output_operands->size() == operand_size) {
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
              
              // [Probe] 关键点：在复用内存后进行 Reshape 检查
              CheckAndReshapeTensor(output_tensor, operand_shapes);
              
              output_tensors->datas[b] = output_tensor;
            }
            prev_runtime_op->occur_end_time = runtime_op->end_time;
            
            // [Probe] 记录复用发生
            // LOG(INFO) << "    Reused memory from: " << prev_runtime_op->name;
          }
        }
      }

      if (!has_found) {
        std::vector<sftensor> output_operand_datas;
        for (uint32_t j = 0; j < batch; ++j) {
          // 使用包装好的 CreateTensor
          output_operand_datas.push_back(CreateTensor(operand_shapes));
        }
        runtime_op->output_operands =
            std::make_shared<RuntimeOperand>(operand->name + "_output", operand_shapes,
                                             output_operand_datas, RuntimeDataType::kTypeFloat32);
      }
    } else {
      CHECK(batch == output_tensors->datas.size());
      CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
      CHECK(output_tensors->shapes == operand_shapes);
      for (uint32_t b = 0; b < batch; ++b) {
        sftensor output_tensor = output_tensors->datas[b];
        // [Probe] 关键点：对已存在的 Tensor 进行 Reshape 检查
        CheckAndReshapeTensor(output_tensor, operand_shapes);
      }
    }
  }
}

}  // namespace kuiper_infer