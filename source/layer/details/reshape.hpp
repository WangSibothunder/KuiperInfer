// 2026-王思博
// MIT License
// Copyright (c) 2026 - 王思博
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#ifndef KUIPER_INFER_SOURCE_LAYER_RESHAPE_HPP_
#define KUIPER_INFER_SOURCE_LAYER_RESHAPE_HPP_

#include "layer/abstract/param_layer.hpp"

namespace kuiper_infer {

class ReshapeLayer : public ParamLayer {
 public:
  /**
   * @brief ReshapeLayer 构造函数
   * @param target_shapes 目标形状 (例如 [1, 197, 3, 3, 64])，允许包含 -1
   */
  explicit ReshapeLayer(std::vector<int32_t> target_shapes);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& layer);

 private:
  std::vector<int32_t> target_shapes_;
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_RESHAPE_HPP_