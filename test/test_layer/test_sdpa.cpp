// 2026-王思博
// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Modified for DeiT support

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../../source/layer/details/scaled_dot_product_attention.hpp"
#include "../../source/layer/details/reshape.hpp"
#include "data/load_data.hpp"
#include <vector>
#include <armadillo>

TEST(test_layer, forward_sdpa) {
  using namespace kuiper_infer;
  
  // 1. 加载数据
  arma::fmat q_data = CSVDataLoader::LoadData<float>("test_data/sdpa/q.csv");
  arma::fmat k_data = CSVDataLoader::LoadData<float>("test_data/sdpa/k.csv");
  arma::fmat v_data = CSVDataLoader::LoadData<float>("test_data/sdpa/v.csv");
  arma::fmat gt_data = CSVDataLoader::LoadData<float>("test_data/sdpa/out_gt.csv");
  
  // 2. 构造输入 (B=1, H=2, S=4, D=8)
  std::vector<uint32_t> shape = {1, 2, 4, 8};
  
  auto create_tensor = [&](const arma::fmat& data) {
      auto t = std::make_shared<Tensor<float>>(1, data.n_elem, 1);
      const std::vector<float> vals(data.memptr(), data.memptr() + data.n_elem);
      t->Fill(vals);
      t->Reshape(shape);
      return t;
  };
  
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(create_tensor(q_data));
  inputs.push_back(create_tensor(k_data));
  inputs.push_back(create_tensor(v_data));

  // 3. 运行 SDPA
  SDPALayer layer;
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  
  ASSERT_EQ(layer.Forward(inputs, outputs), StatusCode::kSuccess);
  
  // 4. 验证
  auto output = outputs.at(0);
  const float* out_ptr = output->raw_ptr();
  const float* gt_ptr = gt_data.memptr();
  
  for(uint32_t i=0; i<output->size(); ++i) {
      ASSERT_NEAR(out_ptr[i], gt_ptr[i], 1e-4f) << "Mismatch at " << i;
  }
}