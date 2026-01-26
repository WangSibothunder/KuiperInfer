// MIT License
// Copyright (c) 2022 - 王思博
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#ifndef KUIPER_INFER_SOURCE_LAYER_PERMUTE_HPP_
#define KUIPER_INFER_SOURCE_LAYER_PERMUTE_HPP_

#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {

class PermuteLayer : public NonParamLayer {
 public:
  /**
   * @brief PermuteLayer 构造函数
   * @param dims PNNX 传递的维度顺序 (例如 [0, 2, 1, 3])
   */
  explicit PermuteLayer(std::vector<int32_t> dims);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& layer);

 private:
  std::vector<int32_t> dims_;
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_PERMUTE_HPP_