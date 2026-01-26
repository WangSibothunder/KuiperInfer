// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../../source/layer/details/permute.hpp"
#include "../../source/layer/details/reshape.hpp" // 需要 Reshape 来构造 5D Input
#include "data/load_data.hpp"
#include <vector>
#include <armadillo>

// ... 保留原来的 test_permute_simple 或其他测试 ...

TEST(test_layer, forward_permute_5d) {
  using namespace kuiper_infer;
  
  // 1. 加载数据
  const std::string input_path = "test_data/permute_5d/input.csv";
  const std::string output_gt_path = "test_data/permute_5d/output_gt.csv";
  
  arma::fmat input_data = CSVDataLoader::LoadData<float>(input_path);
  arma::fmat output_gt_data = CSVDataLoader::LoadData<float>(output_gt_path);
  
  ASSERT_FALSE(input_data.empty());
  ASSERT_FALSE(output_gt_data.empty());
  
  // 2. 构造 5D 输入 Tensor
  // Python Shape: (1, 4, 2, 3, 2)
  std::vector<int32_t> input_shape = {1, 4, 2, 3, 2};
  
  // 先创建一个容器 (使用 ReshapeLayer 或手动 Reshape)
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  // 初始物理内存大小要匹配
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, input_data.n_elem, 1);
  
  // 填充数据
  const std::vector<float> input_values(input_data.memptr(), input_data.memptr() + input_data.n_elem);
  input->Fill(input_values);
  
  // 强制转换为 5D 逻辑形状 (模拟上一层是 Reshape 的结果)
  // 注意：这里需要 cast 为 uint32_t
  std::vector<uint32_t> raw_shape_u32(input_shape.begin(), input_shape.end());
  input->Reshape(raw_shape_u32); 
  
  inputs.push_back(input);

  // 3. 创建 Permute Layer
  // PNNX dims: (2, 0, 3, 1, 4)
  std::vector<int32_t> dims = {2, 0, 3, 1, 4};
  PermuteLayer layer(dims);
  
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  
  // 4. 推理
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);

  // 5. 验证
  const auto& output = outputs.at(0);
  
  // 验证形状
  // Expected: (2, 1, 3, 4, 2)
  const std::vector<uint32_t>& out_shapes = output->raw_shapes();
  ASSERT_EQ(out_shapes.size(), 5);
  ASSERT_EQ(out_shapes[0], 2); // from in[2]
  ASSERT_EQ(out_shapes[1], 1); // from in[0]
  ASSERT_EQ(out_shapes[2], 3); // from in[3]
  ASSERT_EQ(out_shapes[3], 4); // from in[1]
  ASSERT_EQ(out_shapes[4], 2); // from in[4]
  
  // 验证数据
  const float* out_ptr = output->raw_ptr();
  const float* gt_ptr = output_gt_data.memptr();
  
  for (uint32_t i = 0; i < output->size(); ++i) {
      ASSERT_NEAR(out_ptr[i], gt_ptr[i], 1e-5f) 
        << "Data mismatch at index " << i;
  }
}