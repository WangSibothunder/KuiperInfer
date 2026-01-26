// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#ifndef KUIPER_INFER_SOURCE_LAYER_SDPA_HPP_
#define KUIPER_INFER_SOURCE_LAYER_SDPA_HPP_

#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {

class SDPALayer : public NonParamLayer {
 public:
  explicit SDPALayer(float scale = 0.0f, float dropout = 0.0f, bool is_causal = false);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& layer);

 private:
  float scale_ = 0.0f;     // 0.0f 表示自动计算 1/sqrt(dim)
  float dropout_ = 0.0f;
  bool is_causal_ = false; // DeiT 通常为 false
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_SDPA_HPP_