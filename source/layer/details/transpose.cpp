#include "transpose.hpp"
#include "permute.hpp" // 复用 PermuteLayer
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

TransposeLayer::TransposeLayer(int dim0, int dim1) 
    : NonParamLayer("torch.transpose"), dim0_(dim0), dim1_(dim1) {}

StatusCode TransposeLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
    if (inputs.empty()) return StatusCode::kInferInputsEmpty;
    
    // 1. 获取 Rank
    auto& input = inputs.at(0);
    uint32_t rank = input->raw_shapes().size();
    
    // 2. 构建 Permute dims
    // 默认是 [0, 1, 2, ... rank-1]
    std::vector<int32_t> dims(rank);
    for(int i=0; i<rank; ++i) dims[i] = i;
    
    // 交换 dim0 和 dim1
    int d0 = dim0_ < 0 ? dim0_ + rank : dim0_;
    int d1 = dim1_ < 0 ? dim1_ + rank : dim1_;
    
    CHECK(d0 >= 0 && d0 < rank && d1 >= 0 && d1 < rank);
    std::swap(dims[d0], dims[d1]);
    
    // 3. 借用 PermuteLayer 的逻辑
    // 这里我们临时创建一个 PermuteLayer 实例来执行计算
    // (这比复制粘贴 Permute 的代码要好)
    PermuteLayer permute_op(dims);
    return permute_op.Forward(inputs, outputs);
}

StatusCode TransposeLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                          std::shared_ptr<Layer<float>>& layer) {
    int dim0 = 0;
    int dim1 = 0;
    if (op->params.count("dim0")) 
        dim0 = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("dim0"))->value;
    if (op->params.count("dim1")) 
        dim1 = std::dynamic_pointer_cast<RuntimeParameterInt>(op->params.at("dim1"))->value;
        
    layer = std::make_shared<TransposeLayer>(dim0, dim1);
    return StatusCode::kSuccess;
}

LayerRegistererWrapper kTransposeCreateInstance(TransposeLayer::CreateInstance, "torch.transpose");

}