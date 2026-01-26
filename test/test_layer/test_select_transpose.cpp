// 2026-王思博
// MIT License
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../../source/layer/details/tensor_select.hpp"
#include "../../source/layer/details/transpose.hpp"
#include "data/load_data.hpp"
#include <vector>
#include <armadillo>

// 辅助函数：加载 CSV 到 Tensor
std::shared_ptr<kuiper_infer::Tensor<float>> LoadTensor(const std::string& path, std::vector<uint32_t> shape) {
    using namespace kuiper_infer;
    arma::fmat data = CSVDataLoader::LoadData<float>(path);
    auto tensor = std::make_shared<Tensor<float>>(1, data.n_elem, 1);
    std::vector<float> values(data.memptr(), data.memptr() + data.n_elem);
    tensor->Fill(values);
    tensor->Reshape(shape);
    return tensor;
}

// -----------------------------------------------------------
// Test Case 1: Tensor.select
// -----------------------------------------------------------
TEST(test_layer, forward_select) {
  using namespace kuiper_infer;
  
  // 1. 加载数据
  // PyTorch Input: (1, 4, 8)
  std::vector<uint32_t> in_shape = {1, 4, 8};
  auto input = LoadTensor("test_data/select_transpose/select_input.csv", in_shape);
  
  arma::fmat gt_data = CSVDataLoader::LoadData<float>("test_data/select_transpose/select_output_gt.csv");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  
  // 2. 创建 Layer
  // select(dim=1, index=0)
  SelectLayer layer(1, 0);
  
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  
  // 3. 推理
  ASSERT_EQ(layer.Forward(inputs, outputs), StatusCode::kSuccess);
  
  // 4. 验证
  auto output = outputs.at(0);
  
  // 验证形状: PyTorch select 会移除维度 -> (1, 8)
  const auto& out_shapes = output->raw_shapes();
  ASSERT_EQ(out_shapes.size(), 2);
  ASSERT_EQ(out_shapes[0], 1);
  ASSERT_EQ(out_shapes[1], 8);
  
  // 验证数据
  const float* out_ptr = output->raw_ptr();
  const float* gt_ptr = gt_data.memptr();
  for(uint32_t i=0; i<output->size(); ++i) {
      ASSERT_NEAR(out_ptr[i], gt_ptr[i], 1e-5f) << "Select mismatch at " << i;
  }
}

// -----------------------------------------------------------
// Test Case 2: torch.transpose
// -----------------------------------------------------------
TEST(test_layer, forward_transpose) {
  using namespace kuiper_infer;
  
  // 1. 加载数据
  // PyTorch Input: (1, 8, 4)
  std::vector<uint32_t> in_shape = {1, 8, 4};
  auto input = LoadTensor("test_data/select_transpose/transpose_input.csv", in_shape);
  
  arma::fmat gt_data = CSVDataLoader::LoadData<float>("test_data/select_transpose/transpose_output_gt.csv");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  
  // 2. 创建 Layer
  // transpose(dim0=1, dim1=2)
  TransposeLayer layer(1, 2);
  
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  
  // 3. 推理
  ASSERT_EQ(layer.Forward(inputs, outputs), StatusCode::kSuccess);
  
  // 4. 验证
  auto output = outputs.at(0);
  
  // 验证形状: (1, 4, 8)
  const auto& out_shapes = output->raw_shapes();
  ASSERT_EQ(out_shapes.size(), 3);
  ASSERT_EQ(out_shapes[0], 1);
  ASSERT_EQ(out_shapes[1], 4);
  ASSERT_EQ(out_shapes[2], 8);
  
  // 验证数据
  const float* out_ptr = output->raw_ptr();
  const float* gt_ptr = gt_data.memptr();
  for(uint32_t i=0; i<output->size(); ++i) {
      ASSERT_NEAR(out_ptr[i], gt_ptr[i], 1e-5f) << "Transpose mismatch at " << i;
  }
}