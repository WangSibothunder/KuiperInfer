// 2026-王思博
// MIT License
#include "layer_norm.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <atomic>
#include <numeric>
#include <cmath>

namespace kuiper_infer {

LayerNormLayer::LayerNormLayer(int32_t normalized_shape, float eps, bool affine)
    : ParamLayer("nn.LayerNorm"), eps_(eps), affine_(affine) {
    this->normalized_shape_ = {normalized_shape};
    if (affine_ && normalized_shape > 0) {
      this->InitWeightParam(1, 1, 1, static_cast<uint32_t>(normalized_shape));
      this->InitBiasParam(1, 1, 1, static_cast<uint32_t>(normalized_shape));
    }
}

LayerNormLayer::LayerNormLayer(std::vector<int32_t> normalized_shape, float eps, bool affine)
    : ParamLayer("nn.LayerNorm"), eps_(eps), affine_(affine), normalized_shape_(std::move(normalized_shape)) {
    if (affine_ && !normalized_shape_.empty()) {
      int32_t total = 1;
      for (int32_t d : normalized_shape_) {
        total *= d;
      }
      if (total > 0) {
        this->InitWeightParam(1, 1, 1, static_cast<uint32_t>(total));
        this->InitBiasParam(1, 1, 1, static_cast<uint32_t>(total));
      }
    }
}

StatusCode LayerNormLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) return StatusCode::kInferInputsEmpty;

  const uint32_t batch_size = inputs.size();
  
  // LayerNorm 通常在最内层维度进行 (DeiT: dim=192)
  // 如果 normalized_shape_ 是 [192]，则对最后一维归一化
  int32_t norm_dim_size = 1;
  for(int32_t dim : normalized_shape_) norm_dim_size *= dim;

  static std::atomic<uint32_t> ln_call_count{0};
  const bool debug_this = (ln_call_count.fetch_add(1) == 0);
  
  // 检查 inputs[0] 的最后一维是否匹配
  // 简单实现：假设是对最后一维归一化
  // TODO: 支持多维归一化
  
  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(b) = output;
    }
    if (output->shapes() != input->shapes() || output->size() != input->size()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(b) = output;
    }

    const uint32_t total_elems = input->size();
    const uint32_t outer_size = total_elems / norm_dim_size;
    
    // 检查维度是否整除
    CHECK_EQ(total_elems % norm_dim_size, 0);
    CHECK_EQ(output->size(), total_elems);

    if (debug_this) {
      LOG(INFO) << ">>> [LayerNormDebug] total_elems=" << total_elems
                << " norm_dim_size=" << norm_dim_size
                << " outer_size=" << outer_size;
    }

    const float* in_ptr = input->raw_ptr();
    float* out_ptr = output->raw_ptr();
    
    // 遍历每一个需要归一化的块 (例如每个 token)
    for (uint32_t i = 0; i < outer_size; ++i) {
        const float* current_in = in_ptr + i * norm_dim_size;
        float* current_out = out_ptr + i * norm_dim_size;
        
        // 1. 计算 Mean
        float sum = 0.f;
        for (uint32_t j = 0; j < norm_dim_size; ++j) sum += current_in[j];
        float mean = sum / norm_dim_size;
        
        // 2. 计算 Var
        float sum_sq_diff = 0.f;
        for (uint32_t j = 0; j < norm_dim_size; ++j) {
            float diff = current_in[j] - mean;
            sum_sq_diff += diff * diff;
        }
        float var = sum_sq_diff / norm_dim_size;
        float inv_std = 1.0f / std::sqrt(var + eps_);
        
        // 3. Normalize + Affine
        for (uint32_t j = 0; j < norm_dim_size; ++j) {
            float val = (current_in[j] - mean) * inv_std;
            
            if (affine_) {
                // weights_ 存放 gamma, bias_ 存放 beta
                // 假设 weights_ 已加载且大小为 norm_dim_size (PNNX 格式)
                // weights_ 通常是一个 Tensor，我们按线性索引取
                float gamma = 1.0f;
                float beta = 0.0f;
                
                if (!this->weights_.empty() && this->weights_[0]->size() == norm_dim_size) {
                    gamma = this->weights_[0]->index(j);
                }
                
                if (!this->bias_.empty() && this->bias_[0]->size() == norm_dim_size) {
                    beta = this->bias_[0]->index(j);
                }
                
                val = val * gamma + beta;
            }
            current_out[j] = val;
        }
    }
  }
  return StatusCode::kSuccess;
}

StatusCode LayerNormLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                          std::shared_ptr<Layer<float>>& layer) {
  if (!op) return StatusCode::kParseNullOperator;

  // 1. 解析 eps
  float eps = 1e-5f;
  if (op->params.find("eps") != op->params.end()) {
      auto p = std::dynamic_pointer_cast<RuntimeParameterFloat>(op->params.at("eps"));
      if (p) eps = p->value;
  }

  // 2. 解析 elementwise_affine
  bool affine = true;
  if (op->params.find("elementwise_affine") != op->params.end()) {
      auto p = std::dynamic_pointer_cast<RuntimeParameterBool>(op->params.at("elementwise_affine"));
      if (p) affine = p->value;
  }

  // 3. 解析 normalized_shape (关键崩溃点修复)
  std::vector<int32_t> normalized_shape;
  if (op->params.find("normalized_shape") != op->params.end()) {
      auto& param = op->params.at("normalized_shape");
      
      // 分情况处理 Int 和 IntArray
      if (param->type == RuntimeParameterType::kParameterIntArray) {
          auto p = std::dynamic_pointer_cast<RuntimeParameterIntArray>(param);
          if (p) normalized_shape = p->value;
      } else if (param->type == RuntimeParameterType::kParameterInt) {
          auto p = std::dynamic_pointer_cast<RuntimeParameterInt>(param);
          if (p) normalized_shape.push_back(p->value);
      } else {
          LOG(ERROR) << "Unknown type for normalized_shape in LayerNorm";
      }
  }

  layer = std::make_shared<LayerNormLayer>(normalized_shape, eps, affine);
  auto ln_layer = std::dynamic_pointer_cast<LayerNormLayer>(layer);
  if (affine) {
      if (op->has_attribute("weight")) {
          auto weight = op->attribute.at("weight");
          ln_layer->set_weights(weight->get<float>());
      }
      if (op->has_attribute("bias")) {
          auto bias = op->attribute.at("bias");
          ln_layer->set_bias(bias->get<float>());
      }
      if (op->name == "ln_0" && !ln_layer->weights().empty() && !ln_layer->bias().empty()) {
          LOG(INFO) << ">>> [LayerNormParamProbe] ln_0 gamma0=" << ln_layer->weights()[0]->index(0)
                    << " beta0=" << ln_layer->bias()[0]->index(0)
                    << " gamma_size=" << ln_layer->weights()[0]->size();
      }
  }
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kLayerNormCreateInstance(LayerNormLayer::CreateInstance, "nn.LayerNorm");

}  // namespace kuiper_infer
