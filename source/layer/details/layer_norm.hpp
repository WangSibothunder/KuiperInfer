// 2026-王思博
// MIT License
#ifndef KUIPER_INFER_SOURCE_LAYER_LAYER_NORM_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LAYER_NORM_HPP_

#include "layer/abstract/param_layer.hpp"

namespace kuiper_infer {

class LayerNormLayer : public ParamLayer {
 public:
  explicit LayerNormLayer(int32_t normalized_shape, float eps, bool affine);
  
  // 重载构造函数以支持多维 normalized_shape (虽然 DeiT 通常是 1D)
  explicit LayerNormLayer(std::vector<int32_t> normalized_shape, float eps, bool affine);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& layer);

 private:
  float eps_ = 1e-5f;
  bool affine_ = true;
  std::vector<int32_t> normalized_shape_;
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_LAYER_NORM_HPP_