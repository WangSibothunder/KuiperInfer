// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#include "scaled_dot_product_attention.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <cmath>
#include <armadillo>

namespace kuiper_infer {

SDPALayer::SDPALayer(float scale, float dropout, bool is_causal)
    : NonParamLayer("SDPA"), scale_(scale), dropout_(dropout), is_causal_(is_causal) {}

StatusCode SDPALayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.size() != 3) {
    LOG(ERROR) << "SDPA layer requires 3 inputs (Query, Key, Value)";
    return StatusCode::kInferInputsEmpty;
  }

  // Inputs: 0=Query, 1=Key, 2=Value
  // Shape: (Batch, Head, Seq, Dim)
  const auto& q_tensor = inputs.at(0);
  const auto& k_tensor = inputs.at(1);
  const auto& v_tensor = inputs.at(2);

  if (outputs.empty()) {
    LOG(ERROR) << "Output tensor array is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  // 维度检查
  const auto& q_shape = q_tensor->raw_shapes();
  
  CHECK_EQ(q_shape.size(), 4) << "SDPA expects 4D input (Batch, Head, Seq, Dim)";
  
  // 提取维度
  uint32_t batch = q_shape[0];
  uint32_t head = q_shape[1];
  uint32_t seq = q_shape[2];
  uint32_t head_dim = q_shape[3];

  // 自动计算 Scale
  float scale_factor = scale_;
  if (scale_factor == 0.0f) {
      scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
  }

  // 准备输出
  std::shared_ptr<Tensor<float>> output = outputs.at(0);
  if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(1, q_tensor->size(), 1);
      outputs.at(0) = output;
  }
  output->Reshape(q_shape); // 输出形状与 Q 一致

  const float* q_ptr_base = q_tensor->raw_ptr();
  const float* k_ptr_base = k_tensor->raw_ptr();
  const float* v_ptr_base = v_tensor->raw_ptr();
  float* out_ptr_base = output->raw_ptr();

  // 步长计算 (Batch * Head 个矩阵乘法)
  uint32_t matrix_size = seq * head_dim;
  uint32_t total_matrices = batch * head;

#pragma omp parallel for num_threads(total_matrices)
  for (uint32_t i = 0; i < total_matrices; ++i) {
      uint32_t offset = i * matrix_size;
      
      // 巧妙利用 Armadillo 包装器
      // 内存是 Row-Major 的 (Seq, Dim)
      // arma::fmat 包装后变成 (Dim, Seq) 的矩阵，即原矩阵的转置 M^T
      arma::fmat Q_arma(const_cast<float*>(q_ptr_base + offset), head_dim, seq, false, true);
      arma::fmat K_arma(const_cast<float*>(k_ptr_base + offset), head_dim, seq, false, true);
      arma::fmat V_arma(const_cast<float*>(v_ptr_base + offset), head_dim, seq, false, true);
      arma::fmat Out_arma(out_ptr_base + offset, head_dim, seq, false, true);

      // 计算公式推导: K * Q^T (对应逻辑上的 Q * K^T)
      arma::fmat scores = K_arma.t() * Q_arma;
      
      // Scale
      scores *= scale_factor;
      
      // 3. Softmax (手动实现替代 arma::softmax)
      // 我们需要对每一列 (dim=0) 进行 Softmax
      // 步骤: Max -> Subtract -> Exp -> Sum -> Divide
      
      // [FIXED] 使用 arma::frowvec (float类型) 替代 arma::rowvec (double类型)
      arma::frowvec max_val = arma::max(scores, 0);
      scores.each_row() -= max_val;
      
      // B. 求指数
      scores = arma::exp(scores);
      
      // [FIXED] 使用 arma::frowvec
      arma::frowvec sum_val = arma::sum(scores, 0);
      
      // D. 归一化
      scores.each_row() /= sum_val;

      // 4. 计算 Output: V * scores (对应逻辑上的 scores * V)
      Out_arma = V_arma * scores;
  }

  return StatusCode::kSuccess;
}

StatusCode SDPALayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& layer) {
  if (!op) return StatusCode::kParseNullOperator;

  float scale = 0.0f;
  float dropout = 0.0f;
  
  if (op->params.find("scale") != op->params.end()) {
     scale = std::dynamic_pointer_cast<RuntimeParameterFloat>(op->params.at("scale"))->value;
  }
  
  layer = std::make_shared<SDPALayer>(scale, dropout);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kSDPACreateInstance(SDPALayer::CreateInstance, "F.scaled_dot_product_attention");

}  // namespace kuiper_infer