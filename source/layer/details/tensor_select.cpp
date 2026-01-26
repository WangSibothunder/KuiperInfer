// 2026-王思博
// MIT License
#include "tensor_select.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

SelectLayer::SelectLayer(int32_t dim, int32_t index)
    : NonParamLayer("Tensor.select"), dim_(dim), index_(index) {}

StatusCode SelectLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) return StatusCode::kInferInputsEmpty;

  const uint32_t batch_size = inputs.size();

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    const auto& in_shapes = input->raw_shapes();
    
    // 1. 处理维度索引 (支持负数索引)
    int32_t real_dim = dim_;
    if (real_dim < 0) real_dim += in_shapes.size();
    
    CHECK(real_dim >= 0 && real_dim < in_shapes.size()) << "Select dim out of range";

    // 2. 计算输出形状
    // Select 操作会移除被选中的那个维度
    std::vector<uint32_t> out_shapes;
    for (int i = 0; i < in_shapes.size(); ++i) {
        if (i != real_dim) out_shapes.push_back(in_shapes[i]);
    }
    // 如果结果是标量或空，至少保留为 (1)
    if (out_shapes.empty()) out_shapes.push_back(1);

    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(1, input->size() / in_shapes[real_dim], 1);
        outputs.at(b) = output;
    }
    output->Reshape(out_shapes);

    // 3. 数据拷贝
    // 这是一个切片操作。对于 DeiT (Batch, Seq, Dim) select(dim=1, index=0)
    // 相当于取 Seq=0 的那一行。
    // 为了通用性，我们需要计算 stride。
    
    // 简易实现：假设是取 Class Token (dim=1, index=0) 且 input 为 (B, S, D)
    // 则变成 (B, D)。Armadillo 是列主序，直接取可能不连续。
    // 使用最稳健的通用坐标映射。
    
    // 计算 Input Strides
    std::vector<uint32_t> in_strides(in_shapes.size(), 1);
    for (int i = in_shapes.size() - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_shapes[i + 1];
    }
    
    // 计算选定维度的偏移基准
    uint32_t select_stride = in_strides[real_dim];
    uint32_t select_offset_base = index_ * select_stride;
    
    // 遍历输出并映射回输入
    const float* in_ptr = input->raw_ptr();
    float* out_ptr = output->raw_ptr();
    
    // 这里的逻辑稍微复杂：由于我们移除了一维，输出是连续的，但输入是跳跃的（除非 select 的是第0维）
    // 为了避开复杂的递归，我们直接遍历所有元素，判断其在 select 维度上的坐标是否等于 index_
    // 这种方法效率较低但绝对正确。
    // 更高效的方法：由于我们已经算出 select_stride，我们可以分块拷贝。
    
    // 针对 DeiT 常见情况: (1, 197, 192) select(1, 0)
    // 维度 1 是 Seq。stride[1] = 192. stride[0] = 197*192.
    // 我们要取第 0 个 Seq。
    // Block size = 192 (stride of dim+1 which is dim 2).
    // Loop outer dims.
    
    // 采用通用全遍历方案 (安全第一)
    uint32_t total = input->size();
    uint32_t out_idx = 0;
    
    for (uint32_t i = 0; i < total; ++i) {
        // 计算当前元素在 real_dim 上的坐标
        uint32_t current_dim_idx = (i / select_stride) % in_shapes[real_dim];
        if (current_dim_idx == index_) {
            out_ptr[out_idx++] = in_ptr[i];
        }
    }
  }
  return StatusCode::kSuccess;
}

StatusCode SelectLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                       std::shared_ptr<Layer<float>>& layer) {
  if (!op) return StatusCode::kParseNullOperator;
  
  int32_t dim = 0;
  int32_t index = 0;
  
  if (op->params.find("dim") != op->params.end()) 
     dim = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("dim"))->value;
  if (op->params.find("index") != op->params.end()) 
     index = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("index"))->value;

  layer = std::make_shared<SelectLayer>(dim, index);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kSelectCreateInstance(SelectLayer::CreateInstance, "Tensor.select");

} // namespace kuiper_infer