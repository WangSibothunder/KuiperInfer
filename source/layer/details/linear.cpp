// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-11-13.

#include "linear.hpp"
#include <cblas.h>
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

LinearLayer::LinearLayer(int32_t in_features, int32_t out_features, bool use_bias, bool row_major_io)
    : ParamLayer("Linear"),
      use_bias_(use_bias),
      in_features_(in_features),
      out_features_(out_features),
      row_major_io_(row_major_io) {
  CHECK_GT(in_features_, 0);
  CHECK_GT(out_features_, 0);
  this->InitWeightParam(1, 1, in_features_, out_features_);
  if (use_bias) {
    this->InitBiasParam(1, 1, 1, out_features);
  }
}

void LinearLayer::set_weights(const std::vector<float>& weights) {
  const size_t elem_size = weights.size();
  const uint32_t batch_size = this->weights_.size();
  CHECK_EQ(weights.size(), in_features_ * out_features_);
  CHECK_EQ(elem_size % batch_size, 0);

  const uint32_t blob_size = elem_size / batch_size;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    const uint32_t end_offset = start_offset + blob_size;
    const auto& sub_values =
        std::vector<float>{weights.begin() + start_offset, weights.begin() + end_offset};
    this->weights_.at(idx)->Fill(sub_values, false);
  }
}

void LinearLayer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  return ParamLayer::set_weights(weights);
}

StatusCode LinearLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  StatusCode check_status = Check(inputs, outputs);
  if (check_status != StatusCode::kSuccess) {
    return check_status;
  }

  uint32_t batch = inputs.size();
  const std::shared_ptr<Tensor<float>>& weight = weights_.front();
  CHECK(weight != nullptr && !weight->empty()) << "The weight tensor in linear layer is empty";
  CHECK_EQ(weight->size(), static_cast<size_t>(in_features_) * out_features_)
      << "The weight tensor size is not match to in/out features";
  const float* weight_ptr = weight->raw_ptr();

  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the linear layer has an empty tensor " << i << " th";
    const std::vector<uint32_t>& input_shapes = input->shapes();

    uint32_t feature_dims = input_shapes.at(1);
    uint32_t in_features = input_shapes.at(2);
    CHECK_GT(out_features_, 0);
    CHECK_GT(in_features_, 0);

    const uint32_t total_elements = input->size();
    if (in_features != in_features_ || total_elements != feature_dims * in_features) {
      CHECK_EQ(total_elements % in_features_, 0)
          << "Input tensor size is not compatible with in_features";
      in_features = in_features_;
      feature_dims = total_elements / in_features_;
    }
    CHECK_GT(feature_dims, 0);

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty() || output->rows() != feature_dims ||
        output->cols() != out_features_ || output->channels() != 1) {
      output = std::make_shared<Tensor<float>>(1, feature_dims, out_features_);
      outputs.at(i) = output;
    }

    const float* input_ptr = input->raw_ptr();
    float* output_ptr = output->raw_ptr();
    if (row_major_io_) {
      // A[M,K] * W^T[K,N] => C[M,N], where weight memory is [N,K] row-major.
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  static_cast<int>(feature_dims), out_features_, in_features_, 1.f,
                  input_ptr, in_features_,
                  weight_ptr, in_features_,
                  0.f, output_ptr, out_features_);
    } else {
      for (uint32_t out_col = 0; out_col < static_cast<uint32_t>(out_features_); ++out_col) {
        for (uint32_t row = 0; row < feature_dims; ++row) {
          float sum = 0.f;
          for (uint32_t in_col = 0; in_col < static_cast<uint32_t>(in_features_); ++in_col) {
            const float in_val = input_ptr[row + in_col * feature_dims];
            const float w_val = weight_ptr[in_col + out_col * static_cast<uint32_t>(in_features_)];
            sum += in_val * w_val;
          }
          output_ptr[row + out_col * feature_dims] = sum;
        }
      }
    }

    if (use_bias_) {
      CHECK(!this->bias_.empty() && this->bias_.size() == 1)
          << "The bias tensor is empty, but \"use bias\" is true";

      const auto& bias_data = bias_.front()->data();
      CHECK(!bias_data.empty() && bias_data.n_slices == 1 && bias_data.n_cols == out_features_)
          << "The col of bias tensor is not same to output features";
      const float* bias_ptr = bias_.front()->raw_ptr();
      if (row_major_io_) {
        for (uint32_t row = 0; row < feature_dims; ++row) {
          const uint32_t out_base = row * static_cast<uint32_t>(out_features_);
          for (uint32_t out_col = 0; out_col < static_cast<uint32_t>(out_features_); ++out_col) {
            output_ptr[out_base + out_col] += bias_ptr[out_col];
          }
        }
      } else {
        for (uint32_t out_col = 0; out_col < static_cast<uint32_t>(out_features_); ++out_col) {
          const float b = bias_ptr[out_col];
          for (uint32_t row = 0; row < feature_dims; ++row) {
            output_ptr[row + out_col * feature_dims] += b;
          }
        }
      }
    }
  }
  return StatusCode::kSuccess;
}

StatusCode LinearLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                       std::shared_ptr<Layer<float>>& linear_layer) {
  if (!op) {
    LOG(ERROR) << "The linear operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the linear layer is empty.";
    return StatusCode::kParseParamError;
  }

  if (!op->has_parameter("bias")) {
    LOG(ERROR) << "Can not find the use bias parameter in the parameter list.";
    return StatusCode::kParseParamError;
  }
  auto use_bias_param = std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
  if (use_bias_param == nullptr) {
    LOG(ERROR) << "Can not find the use bias parameter in the parameter list.";
    return StatusCode::kParseParamError;
  }

  const auto& attr = op->attribute;
  if (attr.empty()) {
    LOG_IF(ERROR, attr.empty()) << "The attributes of the operator is empty.";
    return StatusCode::kParseWeightError;
  }

  if (!op->has_attribute("weight")) {
    LOG(ERROR) << "Can not find the weight parameter in the parameter list.";
    return StatusCode::kParseWeightError;
  }

  if (use_bias_param->value) {
    if (!op->has_attribute("bias")) {
      LOG(ERROR) << "Can not find the bias parameter in the parameter list.";
      return StatusCode::kParseWeightError;
    }
  }

  const auto& weight = attr.at("weight");
  const auto& bias = attr.at("bias");
  const auto& shapes = weight->shape;
  if ((shapes.size() < 2)) {
    LOG(ERROR) << "The dimension of the linear weight parameter should be 2.";
    return StatusCode::kParseWeightError;
  }

  int32_t out_features = shapes.at(0);
  int32_t in_features = shapes.at(1);
  const bool use_bias = use_bias_param->value;

  const bool row_major_io =
      (op->name.find("blocks.") == 0 || op->name == "head" || op->name == "blocks");
  linear_layer = std::make_shared<LinearLayer>(in_features, out_features, use_bias, row_major_io);
  if (use_bias) {
    linear_layer->set_bias(bias->get<float>());
  }

  // load weights
  linear_layer->set_weights(weight->get<float>());
  return StatusCode::kSuccess;
}

StatusCode LinearLayer::Check(const std::vector<sftensor>& inputs,
                              const std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the linear layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the linear layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the linear "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  if (this->weights_.empty()) {
    LOG(ERROR) << "The weight tensor in the linear layer is empty";
    return StatusCode::kInferParamError;
  } else {
    if (this->use_bias_ && this->weights_.size() != this->bias_.size()) {
      LOG(ERROR) << "The size of the weight and bias tensor do not match";
      return StatusCode::kInferParamError;
    }
  }

  if (weights_.size() != 1) {
    LOG(ERROR) << "Need one weight tensor in the linear layer";
    return StatusCode::kInferParamError;
  }

  if (use_bias_ && this->bias_.size() != 1) {
    LOG(ERROR) << "Need one bias tensor in the linear layer";
    return StatusCode::kInferParamError;
  }
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kLinearCreateInstance(LinearLayer::CreateInstance, "nn.Linear");

}  // namespace kuiper_infer
