// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#include "scaled_dot_product_attention.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <limits>
#include <vector>

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
  if (output == nullptr || output->empty() || output->size() != q_tensor->size()) {
      output = std::make_shared<Tensor<float>>(1, q_tensor->size(), 1);
      outputs.at(0) = output;
  }
  output->Reshape(q_shape); // 输出形状与 Q 一致

  const float* q_ptr_base = q_tensor->raw_ptr();
  const float* k_ptr_base = k_tensor->raw_ptr();
  const float* v_ptr_base = v_tensor->raw_ptr();
  float* out_ptr_base = output->raw_ptr();

  // 步长计算 (Batch * Head 个注意力矩阵)
  const uint32_t matrix_size = seq * head_dim;
  const uint32_t total_matrices = batch * head;

  for (uint32_t i = 0; i < total_matrices; ++i) {
      const uint32_t offset = i * matrix_size;
      const float* q_ptr = q_ptr_base + offset;
      const float* k_ptr = k_ptr_base + offset;
      const float* v_ptr = v_ptr_base + offset;
      float* out_ptr = out_ptr_base + offset;

      std::vector<float> logits(seq * seq, 0.f);
      std::vector<float> probs(seq * seq, 0.f);

      // logits = (Q * K^T) * scale
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  static_cast<int>(seq), static_cast<int>(seq), static_cast<int>(head_dim),
                  scale_factor,
                  q_ptr, static_cast<int>(head_dim),
                  k_ptr, static_cast<int>(head_dim),
                  0.f, logits.data(), static_cast<int>(seq));

      // Row-wise softmax
      for (uint32_t q_idx = 0; q_idx < seq; ++q_idx) {
        const float* logit_row = logits.data() + q_idx * seq;
        float* prob_row = probs.data() + q_idx * seq;

        float max_logit = -std::numeric_limits<float>::infinity();
        for (uint32_t k_idx = 0; k_idx < seq; ++k_idx) {
          max_logit = std::max(max_logit, logit_row[k_idx]);
        }

        float sum_exp = 0.0f;
        for (uint32_t k_idx = 0; k_idx < seq; ++k_idx) {
          const float exp_v = std::exp(logit_row[k_idx] - max_logit);
          prob_row[k_idx] = exp_v;
          sum_exp += exp_v;
        }
        const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (uint32_t k_idx = 0; k_idx < seq; ++k_idx) {
          prob_row[k_idx] *= inv_sum;
        }
      }

      // out = probs * V
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  static_cast<int>(seq), static_cast<int>(head_dim), static_cast<int>(seq),
                  1.f,
                  probs.data(), static_cast<int>(seq),
                  v_ptr, static_cast<int>(head_dim),
                  0.f, out_ptr, static_cast<int>(head_dim));
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
