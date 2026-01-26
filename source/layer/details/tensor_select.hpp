// 2026-王思博
// MIT License
#ifndef KUIPER_INFER_SOURCE_LAYER_TENSOR_SELECT_HPP_
#define KUIPER_INFER_SOURCE_LAYER_TENSOR_SELECT_HPP_

#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {

class SelectLayer : public NonParamLayer {
 public:
  explicit SelectLayer(int32_t dim, int32_t index);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& select_layer);

 private:
  int32_t dim_ = 0;
  int32_t index_ = 0;
};
}  // namespace kuiper_infer

#endif