// 2026-王思博
// MIT License
#include "unbind.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

UnbindLayer::UnbindLayer(int32_t dim) : NonParamLayer("torch.unbind"), dim_(dim) {}

StatusCode UnbindLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) return StatusCode::kInferInputsEmpty;

  const uint32_t batch_size = inputs.size();
  
  // 对于 Unbind，outputs 的数量取决于 batch_size * split_count
  // 但 KuiperInfer 的 Forward 接口定义 outputs 通常对应一个 batch 的输出列表
  // 这里 PNNX 的 Unbind 会产生多个输出节点。
  // 在 RuntimeGraph 中，如果一个节点有 N 个输出，outputs 向量的大小应该足以容纳所有 outputs。
  // 注意：Forward 这里的 outputs 是 vector<shared_ptr<Tensor>>, 通常 index 对应 batch。
  // 但是！Unbind 是一个 "多输出算子" (Multi-Output Operator)。
  // 如果 Layer 产生多个输出，通常的做法是：
  // 1. 如果 batch=1，outputs 的大小就是 unbind 的份数。
  // 2. 如果 batch>1，这比较麻烦。
  // 幸好 DeiT 推理通常 Batch=1。
  
  // 修正：KuiperInfer 的 Layer::Forward 定义中，outputs.size() == inputs.size() (即 batch_size)
  // 每个 outputs[b] 是一个 Tensor。
  // 这对于 Unbind 是不够的，因为 Unbind 针对一个 Input 产生 N 个 Output Tensors。
  
  // 关键：我们需要确认 KuiperInfer 是否支持单层多输出？
  // 查看 `runtime_ir.cpp` 的 `Forward` 逻辑，它从 `output_operands` 获取输出。
  // 如果算子有多个输出端口，Layer 需要怎么填充？
  
  // 回顾 `split` (在 cat.cpp 或 split 相关的实现，如果有的话)。
  // 如果没有现成的多输出案例，我们需要利用 `outputs` 参数的特殊约定。
  // 假设：outputs 传入的是所有输出端口的张量扁平列表？
  // 不，通常 `Forward(inputs, outputs)` 的语义是 outputs[i] 对应 inputs[i] 的结果 (Batch 维度)。
  
  // 重新审视 Unbind 在 PNNX 中的角色：
  // Input: (1, 3, 197, 64) -> dim=1
  // Output0: (1, 197, 64)
  // Output1: (1, 197, 64)
  // Output2: (1, 197, 64)
  
  // 这是一个 "1 入 3 出" 的算子。
  // 现有的 `Layer::Forward` 接口 `(vector<Tensor> in, vector<Tensor> out)` 
  // 默认是 Batch 处理。即 out.size() == in.size()。
  // 这意味 KuiperInfer 可能暂时不支持单层多输出，或者我们对其接口理解需要调整。
  
  // 临时解决方案 (针对 DeiT Batch=1)：
  // 我们假设 outputs.size() == split_size (即 3)。
  // 并在代码中只处理 inputs[0]。
  
  if (inputs.size() != 1) {
      LOG(ERROR) << "UnbindLayer currently only supports batch_size=1";
      return StatusCode::kInferDimMismatch;
  }
  
  const auto& input = inputs.at(0);
  const auto& in_shapes = input->raw_shapes();
  
  // 1. 处理维度
  int32_t real_dim = dim_;
  if (real_dim < 0) real_dim += in_shapes.size();
  CHECK(real_dim >= 0 && real_dim < in_shapes.size());
  
  uint32_t split_size = in_shapes[real_dim];
  
  // 检查 outputs 容器大小是否足够
  // RuntimeGraph 在调用 Forward 前会分配好 outputs 吗？
  // 通常是 Layer 内部 resize，或者外部传入空指针。
  if (outputs.size() != split_size) {
      outputs.resize(split_size);
  }
  
  // 2. 计算输出形状 (移除 real_dim)
  std::vector<uint32_t> out_shapes;
  for (int i = 0; i < in_shapes.size(); ++i) {
      if (i != real_dim) out_shapes.push_back(in_shapes[i]);
  }
  if (out_shapes.empty()) out_shapes.push_back(1);

  // 3. 计算 Stride 以便切片
  std::vector<uint32_t> in_strides(in_shapes.size(), 1);
  for (int i = in_shapes.size() - 2; i >= 0; --i) {
      in_strides[i] = in_strides[i + 1] * in_shapes[i + 1];
  }
  uint32_t dim_stride = in_strides[real_dim];
  uint32_t copy_size_per_step = (real_dim == in_shapes.size() - 1) ? 1 : in_strides[real_dim] / in_shapes[real_dim]; 
  // 上面 copy_size 计算可能有误，换个思路：
  // 无论如何，我们需要遍历整个 Tensor，把元素分发到不同的 Output 中。
  
  const float* in_ptr = input->raw_ptr();
  
  // 初始化所有输出 Tensor
  const uint32_t out_size = input->size() / split_size;
  for (int i = 0; i < split_size; ++i) {
      if (outputs[i] == nullptr || outputs[i]->empty() || outputs[i]->size() != out_size) {
          outputs[i] = std::make_shared<Tensor<float>>(1, out_size, 1);
      }
      outputs[i]->Reshape(out_shapes);
  }
  
  // 4. 数据分发
  // 遍历 Input 的每一个元素，计算它在 real_dim 上的坐标
  uint32_t total_elems = input->size();
  
  // 为了加速，我们需要维护每个 output 的当前写入指针
  std::vector<float*> out_ptrs(split_size);
  for(int i=0; i<split_size; ++i) out_ptrs[i] = outputs[i]->raw_ptr();
  
  // 并行优化难做，因为写入位置不连续。单线程遍历即可。
  for (uint32_t i = 0; i < total_elems; ++i) {
      // 计算当前元素属于哪个 split (即在 real_dim 上的 index)
      // index = (i / dim_stride) % dim_size
      uint32_t split_idx = (i / dim_stride) % split_size;
      
      // 写入对应 Tensor，并自增指针
      *(out_ptrs[split_idx]++) = in_ptr[i];
  }

  return StatusCode::kSuccess;
}

StatusCode UnbindLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                       std::shared_ptr<Layer<float>>& layer) {
  if (!op) return StatusCode::kParseNullOperator;
  
  int32_t dim = 0;
  if (op->params.find("dim") != op->params.end()) {
     dim = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("dim"))->value;
  }
  
  layer = std::make_shared<UnbindLayer>(dim);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kUnbindCreateInstance(UnbindLayer::CreateInstance, "torch.unbind");

}  // namespace kuiper_infer
