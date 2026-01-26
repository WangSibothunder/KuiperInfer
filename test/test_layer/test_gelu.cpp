// test/test_layer/test_gelu.cpp
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../../source/layer/details/gelu.hpp"
#include <cmath>

TEST(test_layer, forward_gelu) {
  using namespace kuiper_infer;
  
  // 1. 创建输入: 包括正数、负数、零
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 5);
  input->index(0) = 1.0f;
  input->index(1) = -1.0f;
  input->index(2) = 0.0f;
  input->index(3) = 3.0f;   // 应该接近 3
  input->index(4) = -3.0f;  // 应该接近 0
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  GELULayer layer;
  layer.Forward(inputs, outputs);

  const auto& output = outputs.at(0);
  
  // 2. 验证
  // GELU(1) ≈ 0.8413
  // GELU(-1) ≈ -0.1586
  // GELU(0) = 0
  // GELU(3) ≈ 2.995 (接近 x)
  // GELU(-3) ≈ -0.004 (接近 0)
  
  ASSERT_NEAR(output->index(0), 0.8413f, 0.001f);
  ASSERT_NEAR(output->index(1), -0.1586f, 0.001f);
  ASSERT_NEAR(output->index(2), 0.0f, 0.0001f);
  ASSERT_NEAR(output->index(3), 2.9959f, 0.01f);
  ASSERT_NEAR(output->index(4), -0.0040f, 0.01f);
}