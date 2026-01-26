// MIT License
// Copyright (c) 2022 - 王思博
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified by User for DeiT-Tiny support

#ifndef KUIPER_INFER_SOURCE_LAYER_GELU_HPP_
#define KUIPER_INFER_SOURCE_LAYER_GELU_HPP_

#include "layer/abstract/layer.hpp"

namespace kuiper_infer {

class GELULayer : public Layer<float> {
 public:
  explicit GELULayer();

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& gelu_layer);
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_GELU_HPP_