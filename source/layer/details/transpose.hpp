#ifndef KUIPER_INFER_SOURCE_LAYER_TRANSPOSE_HPP_
#define KUIPER_INFER_SOURCE_LAYER_TRANSPOSE_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {
class TransposeLayer : public NonParamLayer {
public:
    explicit TransposeLayer(int dim0, int dim1);
    StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                       std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;
    static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& layer);
private:
    int dim0_;
    int dim1_;
};
}
#endif