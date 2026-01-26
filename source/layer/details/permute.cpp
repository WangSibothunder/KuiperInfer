// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support (Generic N-Dim Permute)

#include "permute.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <numeric>

namespace kuiper_infer {

PermuteLayer::PermuteLayer(std::vector<int32_t> dims)
    : NonParamLayer("Tensor.permute"), dims_(std::move(dims)) {}

StatusCode PermuteLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the Permute layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the Permute layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  const uint32_t batch_size = inputs.size();

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    CHECK(input != nullptr && !input->empty()) << "The input tensor at index " << b << " is empty";

    // 1. 获取输入形状 (支持任意维度)
    const std::vector<uint32_t>& in_shapes = input->raw_shapes();
    const uint32_t rank = in_shapes.size();
    
    // 检查 dims 参数是否合法
    CHECK_EQ(dims_.size(), rank) 
        << "Permute dims size (" << dims_.size() << ") must match input tensor rank (" << rank << ")";

    // 2. 计算输出形状
    std::vector<uint32_t> out_shapes(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        int32_t d = dims_[i];
        CHECK(d >= 0 && d < static_cast<int32_t>(rank)) << "Invalid permute dimension index";
        out_shapes[i] = in_shapes[d];
    }

    // 3. 准备输出 Tensor
    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty()) {
        // 创建一个足够大的容器，初始形状设为 flatten 后的 1D，随后 Reshape
        output = std::make_shared<Tensor<float>>(1, input->size(), 1);
        outputs.at(b) = output;
    }
    // 更新为正确的逻辑形状
    output->Reshape(out_shapes);
    
    // 4. 计算 Strides (步长) 用于坐标映射
    // 假设数据是 Row-Major (C-style) 连续的，这与我们 LoadData 的方式一致
    std::vector<uint32_t> in_strides(rank);
    std::vector<uint32_t> out_strides(rank);
    
    uint32_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        in_strides[i] = stride;
        stride *= in_shapes[i];
    }
    
    stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        out_strides[i] = stride;
        stride *= out_shapes[i];
    }

    // 5. 执行重排 (线性遍历输出，计算对应的输入偏移)
    const float* in_ptr = input->raw_ptr();
    float* out_ptr = output->raw_ptr();
    const uint32_t total_elements = input->size();

    // 这里的循环可以进一步并行化，但最外层已有 batch 并行
    for (uint32_t i = 0; i < total_elements; ++i) {
        // A. 将线性索引 i 转换为 输出坐标 (out_coords)
        // 例如 5D: [d0, d1, d2, d3, d4]
        uint32_t temp_idx = i;
        uint32_t in_offset = 0;
        
        // 这是一个通用的坐标映射：
        // Output_Coord[k] = (temp_idx / out_strides[k]) % out_shapes[k]
        // 对应的 Input_Coord 来自 dims_[k]
        // Input_Offset += Input_Coord * in_strides[dims_[k]]
        
        for (uint32_t k = 0; k < rank; ++k) {
            uint32_t out_coord_k = (temp_idx / out_strides[k]) % out_shapes[k];
            // 核心逻辑: Output 的第 k 维 对应 Input 的第 dims_[k] 维
            in_offset += out_coord_k * in_strides[dims_[k]];
        }
        
        // B. 拷贝数据
        out_ptr[i] = in_ptr[in_offset];
    }
  }
  return StatusCode::kSuccess;
}

StatusCode PermuteLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& layer) {
  if (!op) {
    LOG(ERROR) << "The operator parameter in the Permute layer is null";
    return StatusCode::kParseNullOperator;
  }

  std::vector<int32_t> dims;
  if (op->params.find("dims") != op->params.end()) {
    auto param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(op->params.at("dims"));
    if (param) {
        dims = param->value;
    }
  }

  if (dims.empty()) {
      LOG(ERROR) << "PermuteLayer missing 'dims' parameter";
      return StatusCode::kParseParamError;
  }

  layer = std::make_shared<PermuteLayer>(dims);
  return StatusCode::kSuccess;
}

// 新增: torch.transpose 的 CreateInstance 函数
StatusCode TransposeCreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& layer) {
    if (!op) return StatusCode::kParseNullOperator;
    
    int dim0 = 0;
    int dim1 = 0;
    
    if (op->params.find("dim0") != op->params.end())
        dim0 = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("dim0"))->value;
    if (op->params.find("dim1") != op->params.end())
        dim1 = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("dim1"))->value;

    // 此时我们不知道 Tensor 的总维度 (Rank)，无法构建完整的 dims 数组 [0, 1, 2...]
    // 这是一个棘手的问题。PermuteLayer 需要知道完整的 dims。
    // 但是，torch.transpose 只交换两个维度。
    
    // 解决方案：我们需要一个专门的 TransposeLayer，或者让 PermuteLayer 支持动态 rank。
    // 鉴于时间，我们快速实现一个 TransposeLayer，它在 Forward 时动态生成 dims 并调用 Permute 逻辑。
    
    // 既然我们已经在 PermuteLayer 实现了通用逻辑，不如直接让 TransposeLayer 继承 PermuteLayer
    // 但是 PermuteLayer 构造函数需要 dims。
    
    // 简单粗暴方案：创建一个新类 TransposeLayer
    // 请将以下代码放在 permute.cpp 内部 (或者新建 transpose.cpp)
    // 为了方便，建议直接新建 source/layer/details/transpose.cpp
    return StatusCode::kFunctionNotImplement; 
}

LayerRegistererWrapper kPermuteCreateInstance(PermuteLayer::CreateInstance, "Tensor.permute");

}  // namespace kuiper_infer

