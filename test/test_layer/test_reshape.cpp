// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../../source/layer/details/reshape.hpp"
#include "data/load_data.hpp"
#include <vector>
#include <armadillo>

TEST(test_layer, forward_reshape_5d) {
  using namespace kuiper_infer;
  
  // 1. 加载数据 (用于填充 Input)
  const std::string input_path = "test_data/reshape/input.csv";
  arma::fmat input_data = CSVDataLoader::LoadData<float>(input_path);
  ASSERT_FALSE(input_data.empty());
  
  // 2. 准备输入 Tensor
  uint32_t b = 1;
  uint32_t c = 1;
  uint32_t h = 4;
  uint32_t w = 18;
  
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(c, h, w);
  
  // 将数据填入 Input (Fill 会自动处理 Row-Major 到 Column-Major 的转换)
  const std::vector<float> input_values(input_data.memptr(), input_data.memptr() + input_data.n_elem);
  input->Fill(input_values); 
  inputs.push_back(input);

  // 3. 创建 Reshape Layer
  // Target Shape: (1, 4, 2, 3, 3) -> 5D
  std::vector<int32_t> target_shapes = {1, 4, 2, 3, 3};
  ReshapeLayer layer(target_shapes);
  
  std::vector<std::shared_ptr<Tensor<float>>> outputs(b);
  
  // 4. 推理
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);

  // 5. 验证
  const auto& output = outputs.at(0);
  
  // 验证 A: 形状是否正确更新为 5D
  const std::vector<uint32_t>& out_shapes = output->raw_shapes();
  ASSERT_EQ(out_shapes.size(), 5);
  ASSERT_EQ(out_shapes[0], 1);
  ASSERT_EQ(out_shapes[1], 4);
  ASSERT_EQ(out_shapes[2], 2);
  ASSERT_EQ(out_shapes[3], 3);
  ASSERT_EQ(out_shapes[4], 3);
  
  // 验证 B: 数据完整性
  // Reshape 不应改变物理内存中的数值顺序，Output 数据应与 Input 严格一致
  // 我们直接对比 Input 和 Output 的物理内存
  ASSERT_EQ(input->size(), output->size());
  
  const float* in_ptr = input->raw_ptr();
  const float* out_ptr = output->raw_ptr();
  
  for(uint32_t i=0; i<input->size(); ++i) {
      ASSERT_EQ(in_ptr[i], out_ptr[i]) << "Data mismatch between Input and Output at index " << i;
  }
}