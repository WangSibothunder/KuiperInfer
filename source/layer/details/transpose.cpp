// MIT License
// Copyright (c) 2022 - 傅莘莘
// Modified by WangSibo 2026: Optimized TransposeLayer to bypass Permute checks

#include "transpose.hpp"
#include "permute.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "data/tensor_util.hpp"
#include <atomic>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>

namespace kuiper_infer {

TransposeLayer::TransposeLayer(int dim0, int dim1) 
    : NonParamLayer("torch.transpose"), dim0_(dim0), dim1_(dim1) {}

StatusCode TransposeLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    return StatusCode::kInferInputsEmpty;
  }

  const uint32_t batch_size = inputs.size();
  if (outputs.size() != batch_size) {
    outputs.resize(batch_size);
  }
  static std::atomic<uint32_t> transpose_call_count{0};
  const bool debug_this = (transpose_call_count.fetch_add(1) == 0);

  // 并行处理每个 Batch，支持任意 Rank，安全兜底，不再触发 FATAL
  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    if (input == nullptr || input->empty()) {
      continue;
    }

    // 使用 raw_shapes 与 Permute 保持一致（每个 Tensor 表示单个样本，不含 batch 维）
    const std::vector<uint32_t>& in_shapes = input->raw_shapes();
    const uint32_t rank = in_shapes.size();
    if (rank == 0) {
      outputs.at(b) = input;
      continue;
    }

    auto norm_dim = [](int d, int r) -> int {
      if (d < 0) {
        d += r;
      }
      return d;
    };

    // 优先按 raw rank 直接解释（当 raw_shapes 含 batch 维时最准确）
    int d0 = norm_dim(dim0_, static_cast<int>(rank));
    int d1 = norm_dim(dim1_, static_cast<int>(rank));
    bool mapped = (d0 >= 0 && d0 < static_cast<int>(rank) && d1 >= 0 &&
                   d1 < static_cast<int>(rank));

    // pnnx 里常见的 (dim0=1, dim1=2) 在 rank=2 的张量上，
    // 需要交换逻辑形状，但保持线性存储次序不变。
    if (rank == 2 && dim0_ == 1 && dim1_ == 2) {
      std::shared_ptr<Tensor<float>>& output = outputs.at(b);
      const uint32_t total_elements = input->size();
      if (output == nullptr || output->empty() || output->size() != total_elements) {
        output = TensorCreate<float>({total_elements});
      }
      output->Reshape({in_shapes[1], in_shapes[0]});
      std::memcpy(output->raw_ptr(), input->raw_ptr(), sizeof(float) * total_elements);
      continue;
    }

    // 若 raw rank 不可解释，则退化为“逻辑 rank 含 batch”映射
    if (!mapped) {
      const int logical_rank = static_cast<int>(rank) + 1;
      int d0_logical = norm_dim(dim0_, logical_rank);
      int d1_logical = norm_dim(dim1_, logical_rank);
      if (d0_logical < 0 || d0_logical >= logical_rank || d1_logical < 0 ||
          d1_logical >= logical_rank) {
        LOG(ERROR) << "Transpose dim out of range. dim0=" << dim0_ << " dim1=" << dim1_
                   << " raw_rank=" << rank << " logical_rank=" << logical_rank;
        outputs.at(b) = input;
        continue;
      }
      if (d0_logical == 0 || d1_logical == 0) {
        outputs.at(b) = input;
        continue;
      }
      d0 = d0_logical - 1;
      d1 = d1_logical - 1;
    }

    if (d0 == d1) {
      // 交换同一维度，相当于 no-op
      outputs.at(b) = input;
      continue;
    }

    // 构造 permute 数组：先 [0,1,2,...]，再交换 d0/d1
    std::vector<int32_t> perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[d0], perm[d1]);

    // 计算输出形状
    std::vector<uint32_t> out_shapes(rank);
    for (uint32_t i = 0; i < rank; ++i) {
      out_shapes[i] = in_shapes[perm[i]];
    }

    // 准备输出 Tensor：容量按元素个数分配，逻辑形状之后 Reshape
    std::shared_ptr<Tensor<float>>& output = outputs.at(b);
    const uint32_t total_elements = input->size();
    if (output == nullptr || output->empty() || output->size() != total_elements) {
      output = TensorCreate<float>({total_elements});
    }
    output->Reshape(out_shapes);

    // 计算输入/输出 strides（行主序）
    std::vector<uint32_t> in_strides(rank);
    std::vector<uint32_t> out_strides(rank);

    uint32_t stride = 1;
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
      in_strides[i] = stride;
      stride *= in_shapes[i];
    }

    stride = 1;
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
      out_strides[i] = stride;
      stride *= out_shapes[i];
    }

    const float* in_ptr = input->raw_ptr();
    float* out_ptr = output->raw_ptr();

    // 通用 N 维坐标映射：线性遍历输出，反推输入偏移
    for (uint32_t idx = 0; idx < total_elements; ++idx) {
      uint32_t tmp = idx;
      uint32_t in_offset = 0;

      for (uint32_t k = 0; k < rank; ++k) {
        uint32_t out_coord_k = (tmp / out_strides[k]) % out_shapes[k];
        int src_dim = perm[k];  // 输出第 k 维来自输入的 src_dim 维
        in_offset += out_coord_k * in_strides[src_dim];
      }

      CHECK_LT(in_offset, total_elements);
      out_ptr[idx] = in_ptr[in_offset];
    }

    if (debug_this) {
      std::ostringstream out_shape_ss;
      out_shape_ss << "(";
      for (size_t si = 0; si < out_shapes.size(); ++si) {
        if (si > 0) {
          out_shape_ss << ",";
        }
        out_shape_ss << out_shapes[si];
      }
      out_shape_ss << ")";
      LOG(INFO) << ">>> [TransposeDebug] in_rank=" << rank << " out_raw=" << out_shape_ss.str()
                << " size=" << output->size();
    }
  }

  return StatusCode::kSuccess;
}

StatusCode TransposeLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                          std::shared_ptr<Layer<float>>& layer) {
  if (!op) return StatusCode::kParseNullOperator;
  int dim0 = 0;
  int dim1 = 0;
  if (op->params.find("dim0") != op->params.end()) {
      dim0 = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("dim0"))->value;
  }
  if (op->params.find("dim1") != op->params.end()) {
      dim1 = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("dim1"))->value;
  }
  layer = std::make_shared<TransposeLayer>(dim0, dim1);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kTransposeCreateInstance(TransposeLayer::CreateInstance, "torch.transpose");

}  // namespace kuiper_infer
