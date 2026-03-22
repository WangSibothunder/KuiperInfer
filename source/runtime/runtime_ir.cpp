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

#include "runtime/runtime_ir.hpp"
#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "utils/time/time_logging.hpp"
#include "data/tensor_util.hpp" // <--- 必须确认有这行

namespace kuiper_infer {
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_bin_path(const std::string& bin_path) { this->bin_path_ = bin_path; }

void RuntimeGraph::set_param_path(const std::string& param_path) { this->param_path_ = param_path; }

const std::string& RuntimeGraph::param_path() const { return this->param_path_; }

const std::string& RuntimeGraph::bin_path() const { return this->bin_path_; }

static bool IsQuantizeOp(const pnnx::Operator* op) { return false; }

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  int32_t load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Can not find the param path or bin path: " << param_path_ << " " << bin_path_;
    return false;
  }

  std::vector<pnnx::Operator*> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  operators_.clear();
  for (const pnnx::Operator* op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node in the model";
      continue;
    } else {
      if (!IsQuantizeOp(op)) {
        std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
        // 初始化算子的名称
        runtime_operator->name = op->name;
        runtime_operator->type = op->type;

        // 初始化算子中的input
        InitGraphOperatorsInput(op->inputs, runtime_operator);

        // 记录输出operand中的名称
        InitGraphOperatorsOutput(op->outputs, runtime_operator);

        // 初始化算子中的attribute(权重)
        InitGraphAttrs(op->attrs, runtime_operator);

        // 初始化算子中的parameter
        InitGraphParams(op->params, runtime_operator);
        this->operators_.push_back(runtime_operator);
      } else {
        LOG(FATAL) << "UnSupported quantize operator in the model " << op->name
                   << " type: " << op->type;
      }
    }
  }

  graph_state_ = GraphState::NeedBuild;
  return true;
}

// [FIXED] RuntimeGraph::Build with Auto-Reallocation for Attributes
void RuntimeGraph::Build() {
  if (graph_state_ == GraphState::Complete) {
    LOG(INFO) << "Model has been built already!";
    return;
  }

  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph || graph_state_ == GraphState::NeedInit) << "Init graph failed!";
  }

  CHECK(graph_state_ >= GraphState::NeedBuild);
  LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty";

  // 1. 构建节点关系
  CreateNodeRelation();

  // 2. 拓扑排序
  ReverseTopoSort();

  // 3. 初始化输入输出
  RuntimeOperatorUtils<float>::InitOperatorInput(operators_);
  RuntimeOperatorUtils<float>::InitOperatorOutput(graph_->ops, operators_);

  // 4. 填充 Attribute 数据 (关键修复)
  for (const auto& op : operators_) {
    if (op->type == "pnnx.Attribute") {
      if (op->attribute.empty()) {
          LOG(ERROR) << "Attribute node " << op->name << " has no attributes!";
          continue;
      }
      
      // 找到数据 payload
      std::shared_ptr<RuntimeAttribute> attr_data = nullptr;
      if (op->attribute.find("data") != op->attribute.end()) {
          attr_data = op->attribute.at("data");
      } else if (op->attribute.find("weight") != op->attribute.end()) {
          attr_data = op->attribute.at("weight");
      } else {
          attr_data = op->attribute.begin()->second;
      }
      
      if (!attr_data) {
          LOG(ERROR) << "Failed to find data for attribute node " << op->name;
          continue;
      }

      // 计算属性的真实总大小
      uint32_t total_size = 1;
      std::vector<uint32_t> attr_shapes;
      for(int i : attr_data->shape) {
          attr_shapes.push_back(i);
          total_size *= i;
      }
      
      // 确保 Attribute 输出空间存在
      if (op->output_operands == nullptr || op->output_operands->datas.empty()) {
          std::vector<int32_t> attr_shapes_i32;
          attr_shapes_i32.reserve(attr_shapes.size());
          for (auto s : attr_shapes) {
            attr_shapes_i32.push_back(static_cast<int32_t>(s));
          }
          const uint32_t batch = attr_shapes.empty() ? 1 : attr_shapes.at(0);
          op->output_operands = std::make_shared<RuntimeOperand>(
              op->name + "_output", attr_shapes_i32, batch, RuntimeDataType::kTypeFloat32);
      }

      for (auto& output_tensor : op->output_operands->datas) {
          // [FIX] 检查大小是否匹配，不匹配或为空则重新分配！
          // 这修正了 InitOperatorOutput 将属性误判为 Batch 导致的分配错误
          if (!output_tensor || output_tensor->size() != total_size) {
              // 重新分配一个足够大的 Tensor (1, 1, total_size)
              output_tensor = TensorCreate<float>(1, 1, total_size);
          }
          
          // 现在大小一定匹配了，安全 Reshape
          output_tensor->Reshape(attr_shapes);
          
          // 填充数据
          const std::vector<float>& float_data = attr_data->get<float>(); 
          // Keep raw payload order from pnnx attribute blob.
          output_tensor->Fill(float_data, false);
      }
      LOG(INFO) << "Attribute node " << op->name << " filled, shape=("
                << attr_shapes[0] << "," << (attr_shapes.size() > 1 ? attr_shapes[1] : 1)
                << "," << (attr_shapes.size() > 2 ? attr_shapes[2] : 1) << ")";
    }
  }

  graph_state_ = GraphState::Complete;
  if (graph_ != nullptr) {
    graph_.reset();
    graph_ = nullptr;
  }
}

template <typename T>
StatusCode ExecuteLayer(const std::shared_ptr<Layer<T>>& layer, const std::string& op_name,
                        const std::string& op_type, bool is_debug) {
  CHECK(layer != nullptr);
  StatusCode status;
  if (is_debug) {
    utils::LayerTimeLogging layer_time_logging(op_name, op_type);
    status = layer->Forward();
  } else {
    status = layer->Forward();
  }
  return status;
}

void RuntimeGraph::Forward(bool debug) {
  // 检查当前的执行图是否已经初始化完毕
  if (graph_state_ < GraphState::Complete) {
    LOG(FATAL) << "Graph need be build!"
               << ", current state is " << int32_t(graph_state_);
  }

  if (debug) {
    utils::LayerTimeStatesSingleton::LayerTimeStatesCollectorInit();
  }

  auto propagate_attribute = [&](const std::shared_ptr<RuntimeOperator>& current_op) {
    if (!current_op || current_op->type != "pnnx.Attribute") {
      return;
    }
    if (current_op->output_operands == nullptr || current_op->output_operands->datas.empty()) {
      LOG(ERROR) << "Attribute node " << current_op->name << " has no output datas to propagate!";
      return;
    }

    PropagateLayerOutputs(current_op, current_op->output_operands->datas, debug);
  };

  // Pre-propagate all attribute constants before executing other ops
  for (const auto& current_op : operators_) {
    propagate_attribute(current_op);
  }

  for (const auto& current_op : operators_) {
    current_op->has_forward = false;
    CHECK_GT(current_op->start_time, 0);

    // 1. 处理 Input/Output 节点 (直接跳过)
    if (is_input_op(current_op->name) || is_output_op(current_op->name)) {
      current_op->has_forward = true;
      continue;
    }

    // 2. 处理 Attribute 节点 (关键修改)
    // 虽然不执行 Layer 计算，但必须向下游传播数据！
    if (current_op->type == "pnnx.Attribute") {
      current_op->has_forward = true;
      continue;
    }

    // 3. 处理普通计算节点
    CHECK(current_op->layer != nullptr)
        << "The layer corresponding to the op " << current_op->name
        << " is empty, indicating that it may not have been created.";

    if (debug) {
      LOG(INFO) << ">>> [ForwardOp] " << current_op->name << " (Type: " << current_op->type
                << ")";
      if (current_op->name == "Tensor.select_89" || current_op->name == "head") {
        for (size_t in_idx = 0; in_idx < current_op->input_operands_seq.size(); ++in_idx) {
          const auto& in_operand = current_op->input_operands_seq[in_idx];
          if (in_operand == nullptr) {
            LOG(INFO) << ">>> [ForwardShapeProbe] op=" << current_op->name << " input[" << in_idx
                      << "] <null operand>";
            continue;
          }
          std::stringstream in_shape_ss;
          in_shape_ss << "(";
          for (size_t k = 0; k < in_operand->shapes.size(); ++k) {
            in_shape_ss << in_operand->shapes[k]
                        << (k + 1 == in_operand->shapes.size() ? "" : ",");
          }
          in_shape_ss << ")";
          LOG(INFO) << ">>> [ForwardShapeProbe] op=" << current_op->name << " input[" << in_idx
                    << "] name=" << in_operand->name << " operand_shape=" << in_shape_ss.str()
                    << " datas=" << in_operand->datas.size();
          for (size_t b = 0; b < in_operand->datas.size(); ++b) {
            const auto& t = in_operand->datas[b];
            if (t == nullptr || t->empty()) {
              LOG(INFO) << ">>> [ForwardShapeProbe] op=" << current_op->name << " input[" << in_idx
                        << "] batch=" << b << " tensor=<empty>";
              continue;
            }
            std::stringstream raw_ss;
            raw_ss << "(";
            const auto& raw = t->raw_shapes();
            for (size_t kk = 0; kk < raw.size(); ++kk) {
              raw_ss << raw[kk] << (kk + 1 == raw.size() ? "" : ",");
            }
            raw_ss << ")";
            LOG(INFO) << ">>> [ForwardShapeProbe] op=" << current_op->name << " input[" << in_idx
                      << "] batch=" << b << " raw_shape=" << raw_ss.str()
                      << " size=" << t->size();
          }
        }
      }
    }

    StatusCode status = ExecuteLayer(current_op->layer, current_op->name, current_op->type, debug);
    CHECK(status == StatusCode::kSuccess)
        << current_op->layer->layer_name()
        << " layer forward failed, error code: " << int32_t(status);

    if (debug) {
      LOG(INFO) << ">>> [ForwardOpExecDone] " << current_op->name;
      static const std::unordered_set<std::string> kStatProbeOps = {
          "patch_embed.proj",   "torch.flatten_63",   "torch.transpose_64",
          "torch.cat_62",       "pnnx_expr_234",      "ln_0",
          "blocks.0.attn.qkv",  "Tensor.reshape_38",  "Tensor.permute_26",
          "torch.unbind_77",    "F.scaled_dot_product_attention_116",
          "torch.transpose_65", "Tensor.reshape_39",  "blocks.0.attn.proj",
          "pnnx_expr_218",      "ln_1",               "blocks.0.mlp.fc1",
          "blocks.0.mlp.act",   "blocks.0.mlp.fc2",   "pnnx_expr_215"};
      if (kStatProbeOps.find(current_op->name) != kStatProbeOps.end()) {
        const auto& outs = current_op->output_operands->datas;
        if (!outs.empty() && outs[0] != nullptr && !outs[0]->empty()) {
          const auto& t = outs[0];
          const float* ptr = t->raw_ptr();
          const uint32_t n = t->size();
          double sum = 0.0;
          double sq = 0.0;
          float mn = std::numeric_limits<float>::infinity();
          float mx = -std::numeric_limits<float>::infinity();
          for (uint32_t i = 0; i < n; ++i) {
            const float v = ptr[i];
            sum += v;
            sq += static_cast<double>(v) * static_cast<double>(v);
            mn = std::min(mn, v);
            mx = std::max(mx, v);
          }
          const double mean = sum / std::max<uint32_t>(1, n);
          const double var = std::max(0.0, sq / std::max<uint32_t>(1, n) - mean * mean);
          std::ostringstream first_ss;
          first_ss << "[";
          for (uint32_t i = 0; i < std::min<uint32_t>(5, n); ++i) {
            if (i > 0) first_ss << ",";
            first_ss << ptr[i];
          }
          first_ss << "]";
          LOG(INFO) << ">>> [TensorStat] op=" << current_op->name
                    << " n=" << n
                    << " mean=" << mean
                    << " std=" << std::sqrt(var)
                    << " min=" << mn
                    << " max=" << mx
                    << " first=" << first_ss.str();
        }
      }
    }

    current_op->has_forward = true;
    PropagateLayerOutputs(current_op, current_op->output_operands->datas, debug);
    if (debug) {
      LOG(INFO) << ">>> [ForwardOpDone] " << current_op->name;
    }
  }

  if (debug) {
    utils::LayerTimeLogging::SummaryLogging();
  }

  for (const auto& op : operators_) {
    LOG_IF(FATAL, !op->has_forward) << "The operator: " << op->name << " has not been forward yet!";
  }
}

template <typename T>
std::shared_ptr<Layer<T>> RuntimeGraph::CreateLayer(
    const std::shared_ptr<RuntimeOperatorBase<T>>& op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  auto layer = LayerRegisterer::CreateLayer(op);
  LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
  return layer;
}

template <typename T>
void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand*>& inputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (inputs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const pnnx::Operand* input : inputs) {
    if (!input) {
      continue;
    }

    std::vector<int32_t> dims;
    const pnnx::Operator* producer = input->producer;

    for (int32_t dim : input->shape) {
      dims.push_back(dim);
    }
    CHECK(!dims.empty());
    std::shared_ptr<RuntimeOperandBase<T>> runtime_operand =
        std::make_shared<RuntimeOperandBase<T>>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = dims;
    runtime_operator->input_operands.insert({producer->name, runtime_operand});
    runtime_operator->input_operands_seq.push_back(runtime_operand);

    switch (input->type) {
      case 1: {
        runtime_operand->type = RuntimeDataType::kTypeFloat32;
        break;
      }
      case 7: {
        runtime_operand->type = RuntimeDataType::kTypeInt8;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown input operand type: " << input->type;
      }
    }
  }
}

template <typename T>
void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand*>& outputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (outputs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const pnnx::Operand* output : outputs) {
    if (!output) {
      continue;
    }
    const auto& consumers = output->consumers;
    for (const auto& c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

template <typename T>
void RuntimeGraph::InitGraphParams(
    const std::map<std::string, pnnx::Parameter>& params,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (params.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const auto& [name, parameter] : params) {
    const int32_t type = parameter.type;
    switch (type) {
      case int32_t(RuntimeParameterType::kParameterUnknown): {
        std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterBool): {
        std::shared_ptr<RuntimeParameterBool> runtime_parameter =
            std::make_shared<RuntimeParameterBool>(parameter.b);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterInt): {
        std::shared_ptr<RuntimeParameterInt> runtime_parameter =
            std::make_shared<RuntimeParameterInt>(parameter.i);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterFloat): {
        std::shared_ptr<RuntimeParameterFloat> runtime_parameter =
            std::make_shared<RuntimeParameterFloat>(parameter.f);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterString): {
        std::shared_ptr<RuntimeParameterString> runtime_parameter =
            std::make_shared<RuntimeParameterString>(parameter.s);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterIntArray): {
        std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
            std::make_shared<RuntimeParameterIntArray>(parameter.ai);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterFloatArray): {
        std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
            std::make_shared<RuntimeParameterFloatArray>(parameter.af);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case int32_t(RuntimeParameterType::kParameterStringArray): {
        std::shared_ptr<RuntimeParameterStringArray> runtime_parameter =
            std::make_shared<RuntimeParameterStringArray>(parameter.as);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown parameter type: " << type;
      }
    }
  }
}

template <typename T>
void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs,
                                  const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (attrs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const auto& [name, attr] : attrs) {
    switch (attr.type) {
      case 1: {
        std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>(
            attr.shape, RuntimeDataType::kTypeFloat32, attr.data);
        runtime_operator->attribute.insert({name, runtime_attribute});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown attribute type: " << attr.type;
      }
    }
  }
}

template <typename T>
void RuntimeGraph::PropagateLayerOutputs(
    const std::shared_ptr<RuntimeOperatorBase<T>>& current_op,
    const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_datas,
    bool debug) {
  auto shapes_to_string = [](const std::vector<int32_t>& shapes) {
    std::ostringstream ss;
    ss << "(";
    for (size_t i = 0; i < shapes.size(); ++i) {
      if (i > 0) ss << ",";
      ss << shapes[i];
    }
    ss << ")";
    return ss.str();
  };

  auto raw_shapes_to_string = [](const std::vector<uint32_t>& shapes) {
    std::ostringstream ss;
    ss << "(";
    for (size_t i = 0; i < shapes.size(); ++i) {
      if (i > 0) ss << ",";
      ss << shapes[i];
    }
    ss << ")";
    return ss.str();
  };

  static std::unordered_set<std::string> warned_edges;

  // For each next operator of current operator
  for (const auto& [_, output_op] : current_op->output_operators) {
    if (debug && current_op->name == "pnnx_fold_38" && output_op->name == "torch.cat_62") {
      std::ostringstream out_names;
      bool first = true;
      for (const auto& [name, __] : current_op->output_operators) {
        if (!first) out_names << ",";
        out_names << name;
        first = false;
      }

      std::ostringstream in_keys;
      first = true;
      for (const auto& [name, __] : output_op->input_operands) {
        if (!first) in_keys << ",";
        in_keys << name;
        first = false;
      }

      LOG(INFO) << "[PropagateProbe] " << current_op->name << " -> " << output_op->name
                << " output_operators={" << out_names.str() << "} input_keys={"
                << in_keys.str() << "}";

      for (const auto& operand : output_op->input_operands_seq) {
        if (!operand) {
          LOG(INFO) << "[PropagateProbe] input_operand <null>";
          continue;
        }
        uint32_t empty_cnt = 0;
        for (const auto& t : operand->datas) {
          if (!t || t->empty()) {
            empty_cnt += 1;
          }
        }
        LOG(INFO) << "[PropagateProbe] input_operand name=" << operand->name
                  << " shapes=" << shapes_to_string(operand->shapes)
                  << " datas=" << operand->datas.size()
                  << " empty=" << empty_cnt;
      }
    }

    // Get next op's input operands corresponding to current op's output
    const auto& next_input_operands = output_op->input_operands;
    const auto& next_input_op_iter = next_input_operands.find(current_op->name);
    if (next_input_op_iter != next_input_operands.end()) {
      // Get input data spaces for those operands
      std::vector<stensor<T>>& next_input_datas = next_input_op_iter->second->datas;
      // Copy current op output data to next op input data
      const size_t copy_count = std::min(next_input_datas.size(), layer_output_datas.size());
      for (size_t i = 0; i < copy_count; ++i) {
        const stensor<T>& layer_output_data = layer_output_datas[i];
        if (next_input_datas.at(i) != nullptr) {
          CHECK(next_input_datas.at(i)->shapes() == layer_output_data->shapes());
        }
        next_input_datas.at(i) = layer_output_data;
      }

      // Handle one-producer multi-output case (e.g. torch.unbind -> attention q/k/v):
      // input_operands map can only keep one entry per producer name, so distribute
      // remaining outputs to other input_operands_seq entries with the same producer name.
      if (layer_output_datas.size() > copy_count) {
        size_t cursor = copy_count;
        for (auto& input_operand : output_op->input_operands_seq) {
          if (!input_operand || input_operand->name != current_op->name ||
              input_operand.get() == next_input_op_iter->second.get()) {
            continue;
          }
          if (input_operand->datas.empty()) {
            input_operand->datas.resize(1);
          }
          const size_t assign_count =
              std::min(input_operand->datas.size(), layer_output_datas.size() - cursor);
          for (size_t i = 0; i < assign_count; ++i) {
            const auto& src = layer_output_datas[cursor + i];
            if (!src || src->empty()) {
              continue;
            }
            input_operand->datas[i] = src;
          }
          cursor += assign_count;
          if (cursor >= layer_output_datas.size()) {
            break;
          }
        }
      }

      if (debug && next_input_datas.size() != layer_output_datas.size()) {
        LOG(WARNING) << "[PropagateMismatch] edge " << current_op->name << " -> "
                     << output_op->name << " next_input_datas=" << next_input_datas.size()
                     << " layer_output_datas=" << layer_output_datas.size();
        if (current_op->name == "torch.unbind_77" &&
            output_op->name == "F.scaled_dot_product_attention_116") {
          size_t same_name_idx = 0;
          for (const auto& input_operand : output_op->input_operands_seq) {
            if (!input_operand || input_operand->name != current_op->name) {
              continue;
            }
            const auto& tensor = input_operand->datas.empty() ? nullptr : input_operand->datas[0];
            const void* ptr = tensor ? static_cast<const void*>(tensor->raw_ptr()) : nullptr;
            std::string shape_str = "null";
            if (tensor && !tensor->empty()) {
              std::ostringstream ss;
              ss << raw_shapes_to_string(tensor->raw_shapes());
              shape_str = ss.str();
            }
            LOG(INFO) << "[UnbindPropagateProbe] slot=" << same_name_idx
                      << " ptr=" << ptr << " shape=" << shape_str;
            same_name_idx += 1;
          }
        }
      }
      continue;
    }

    // Fallback path: key mismatch or missing edge in input map
    bool assigned = false;
    for (auto& input_operand : output_op->input_operands_seq) {
      if (!input_operand) {
        continue;
      }
      if (input_operand->name != current_op->name) {
        continue;
      }
      if (input_operand->datas.empty()) {
        input_operand->datas.resize(layer_output_datas.size());
      }

      const size_t count = std::min(input_operand->datas.size(), layer_output_datas.size());
      for (size_t i = 0; i < count; ++i) {
        const auto& src = layer_output_datas[i];
        if (!src || src->empty()) {
          continue;
        }
        if (input_operand->datas[i] == nullptr || input_operand->datas[i]->empty()) {
          input_operand->datas[i] = src;
          assigned = true;
        }
      }
      if (assigned) {
        break;
      }
    }

    if (assigned) {
      continue;
    }

    // Shape-based fallback: compare tensor raw shapes to operand shapes (with and without batch)
    for (auto& input_operand : output_op->input_operands_seq) {
      if (!input_operand) {
        continue;
      }
      if (input_operand->datas.empty()) {
        input_operand->datas.resize(layer_output_datas.size());
      }

      std::vector<uint32_t> operand_shapes_full;
      operand_shapes_full.reserve(input_operand->shapes.size());
      for (int32_t dim : input_operand->shapes) {
        if (dim <= 0) {
          operand_shapes_full.clear();
          break;
        }
        operand_shapes_full.push_back(static_cast<uint32_t>(dim));
      }

      std::vector<uint32_t> operand_shapes_no_batch;
      if (operand_shapes_full.size() >= 2) {
        operand_shapes_no_batch.assign(operand_shapes_full.begin() + 1, operand_shapes_full.end());
      }

      const size_t count = std::min(input_operand->datas.size(), layer_output_datas.size());
      for (size_t i = 0; i < count; ++i) {
        const auto& src = layer_output_datas[i];
        if (!src || src->empty()) {
          continue;
        }
        if (input_operand->datas[i] != nullptr && !input_operand->datas[i]->empty()) {
          continue;
        }

        const auto& raw_shapes = src->raw_shapes();
        const bool match_full =
            !operand_shapes_full.empty() && raw_shapes == operand_shapes_full;
        const bool match_no_batch =
            !operand_shapes_no_batch.empty() && raw_shapes == operand_shapes_no_batch;

        if (match_full || match_no_batch) {
          if (debug) {
            LOG(INFO) << "[PropagateFallback] " << current_op->name << " -> " << output_op->name
                      << " matched shapes raw=" << raw_shapes_to_string(raw_shapes)
                      << " operand=" << shapes_to_string(input_operand->shapes);
          }
          input_operand->datas[i] = src;
          assigned = true;
        }
      }
      if (assigned) {
        break;
      }
    }

    if (!assigned) {
      const std::string edge_key = current_op->name + "->" + output_op->name;
      if (warned_edges.insert(edge_key).second) {
        LOG(WARNING) << "Propagate fallback failed for edge " << edge_key
                     << " (missing input map key and no shape match)";
      }
    }
  }
}

void RuntimeGraph::ReverseTopoSort() {
  // 构建拓扑顺序
  for (const auto& op : operators_) {
    // 根据输入节点构建拓扑排序
    if (op != nullptr && !op->has_forward) {
      int32_t current_forward_idx = 0;
      this->ReverseTopoSortInternal(op, current_forward_idx);
    }
  }

  // 根据拓扑顺序调整算子的执行顺序
  std::sort(operators_.begin(), operators_.end(), [](const auto& op1, const auto& op2) {
    return op1->start_time > op2->start_time;
  });

  int32_t forward_index = 1;
  for (const auto& op : operators_) {
    op->start_time = forward_index;
    forward_index += 1;
  }

  for (const auto& op : operators_) {
    const auto& next_ops = op->output_operators;
    int32_t last_forward_index = -1;
    for (const auto& [_, next_op] : next_ops) {
      if (next_op->start_time >= last_forward_index) {
        last_forward_index = next_op->start_time;
      }
    }

    if (last_forward_index == -1) {
      op->end_time = op->start_time + 1;
    } else {
      op->end_time = last_forward_index;
    }
    op->occur_end_time = -1;
  }
}

template <typename T>
void RuntimeGraph::ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>>& root_op,
                                           int32_t& current_forward_idx) {
  if (!root_op) {
    LOG(INFO) << "Current operator is nullptr";
    return;
  }
  if (root_op->input_operands.empty() && !root_op->has_forward) {
    this->input_ops_.push_back(root_op);
  }
  if (root_op->output_names.empty() && !root_op->has_forward) {
    this->output_ops_.push_back(root_op);
  }

  root_op->has_forward = true;
  const auto& next_ops = root_op->output_operators;
  for (const auto& [_, op] : next_ops) {
    if (op != nullptr && !op->has_forward) {
      this->ReverseTopoSortInternal(op, current_forward_idx);
    }
  }

  for (const auto& [_, op] : next_ops) {
    CHECK_EQ(op->has_forward, true);
  }
  root_op->start_time = current_forward_idx;
  current_forward_idx += 1;
}

void RuntimeGraph::CreateNodeRelation() {
  LOG(INFO) << ">>> [NodeRelationProbe] Start...";
  
  if (this->operators_.empty()) {
      LOG(FATAL) << ">>> [NodeRelationProbe] Operators vector is empty!";
  }

  int op_index = 0;
  for (const auto& current_op : this->operators_) {
    // 1. 基础检查
    if (current_op == nullptr) {
        LOG(FATAL) << ">>> [NodeRelationProbe] Found nullptr operator at index " << op_index;
    }
    LOG(INFO) << ">>> [NodeRelationProbe] Processing Op [" << op_index << "]: " 
              << current_op->name << " (Type: " << current_op->type << ")";

    // 2. 构建输出关系 (Output Relations)
    const std::vector<std::string>& output_names = current_op->output_names;
    for (const auto& kOutputName : output_names) {
      // 遍历寻找消费者节点
      for (const auto& output_op : this->operators_) {
        if (output_op != current_op && output_op->name == kOutputName) {
          current_op->output_operators.insert({kOutputName, output_op});
        }
      }
    }

    // 2.1 Fallback: 如果 output_names 为空或未建立关系，则从输入依赖反推连接
    if (current_op->output_operators.empty()) {
      for (const auto& output_op : this->operators_) {
        if (output_op == current_op) {
          continue;
        }

        bool matched = false;
        for (const auto& [__, input_operand] : output_op->input_operands) {
          if (input_operand && input_operand->name == current_op->name) {
            current_op->output_operators.insert({output_op->name, output_op});
            matched = true;
            break;
          }
        }

        if (!matched) {
          for (const auto& input_operand : output_op->input_operands_seq) {
            if (input_operand && input_operand->name == current_op->name) {
              current_op->output_operators.insert({output_op->name, output_op});
              break;
            }
          }
        }
      }
    }

    // 3. 创建 Layer (关键崩溃点)
    // 检查是否在跳过列表中
    bool should_skip = (current_op->type == "pnnx.Input" || 
                        current_op->type == "pnnx.Output" || 
                        current_op->type == "pnnx.Attribute");

    if (!should_skip) {
      LOG(INFO) << ">>> [NodeRelationProbe] Creating Layer for: " << current_op->name;
      
      // 调用工厂创建 Layer
      auto layer = RuntimeGraph::CreateLayer(current_op);
      
      if (layer) {
        LOG(INFO) << ">>> [NodeRelationProbe] Layer created successfully.";
        current_op->layer = layer;
        layer->set_runtime_operator(current_op);
      } else {
        // 如果工厂返回空，说明算子未注册
        LOG(FATAL) << ">>> [NodeRelationProbe] Layer create failed! Operator type [" 
                   << current_op->type << "] not registered?";
      }
    } else {
        LOG(INFO) << ">>> [NodeRelationProbe] Skipping layer creation for: " << current_op->type;
    }
    
    op_index++;
  }
  LOG(INFO) << ">>> [NodeRelationProbe] Finished.";
}

RuntimeGraph::GraphState RuntimeGraph::graph_state() const { return this->graph_state_; }

void RuntimeGraph::set_inputs(const std::string& input_name, const std::vector<sftensor>& inputs) {
  CHECK(this->graph_state_ == GraphState::Complete);
  std::shared_ptr<RuntimeOperator> input_op;
  for (auto op : this->input_ops_) {
    if (op->name == input_name) {
      input_op = op;
      break;
    }
  }
  CHECK(input_op != nullptr) << "Can not find the input operator: " << input_name;
  PropagateLayerOutputs(input_op, inputs);
}

std::vector<sftensor> RuntimeGraph::get_outputs(const std::string& output_name) const {
  CHECK(this->graph_state_ == GraphState::Complete);
  std::shared_ptr<RuntimeOperator> output_op;
  for (auto op : this->output_ops_) {
    if (op->name == output_name) {
      output_op = op;
    }
  }

  CHECK(output_op != nullptr) << "Can not find the output operator: " << output_name;
  std::vector<sftensor> outputs;
  for (const auto& input_operand : output_op->input_operands_seq) {
    std::copy(input_operand->datas.begin(), input_operand->datas.end(),
              std::back_inserter(outputs));
  }
  return outputs;
}

bool RuntimeGraph::is_input_op(const std::string& op_name) const {
  for (auto op : this->input_ops_) {
    CHECK(op != nullptr);
    if (op->name == op_name) {
      return true;
    }
  }
  return false;
}

bool RuntimeGraph::is_output_op(const std::string& op_name) const {
  for (auto op : this->output_ops_) {
    CHECK(op != nullptr);
    if (op->name == op_name) {
      return true;
    }
  }
  return false;
}

}  // namespace kuiper_infer
