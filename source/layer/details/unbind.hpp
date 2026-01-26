// 2026-王思博
// MIT License
#ifndef KUIPER_INFER_SOURCE_LAYER_UNBIND_HPP_
#define KUIPER_INFER_SOURCE_LAYER_UNBIND_HPP_

#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {

class UnbindLayer : public NonParamLayer {
 public:
  explicit UnbindLayer(int32_t dim);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& unbind_layer);

 private:
  int32_t dim_ = 0;
};
}  // namespace kuiper_infer

#endif