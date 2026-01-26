// 2026-王思博
// MIT License
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../../source/layer/details/unbind.hpp"
#include "data/load_data.hpp"
#include <vector>
#include <armadillo>
#include <string>

TEST(test_layer, forward_unbind_qkv) {
  using namespace kuiper_infer;
  
  // 1. 准备路径
  const std::string input_path = "test_data/unbind/input.csv";
  // 我们知道 Python 生成了 3 个输出 (Q, K, V)
  const int split_count = 3;
  
  // 2. 加载输入数据
  arma::fmat input_data = CSVDataLoader::LoadData<float>(input_path);
  ASSERT_FALSE(input_data.empty());
  
  // 构造 Input Tensor
  // Python Shape: (1, 4, 3, 8) -> (Batch, Seq, 3, Dim)
  // Kuiper 逻辑维度需要 Reshape 匹配
  uint32_t b = 1;
  uint32_t seq = 4;
  uint32_t three = 3;
  uint32_t dim = 8;
  
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  // 初始化物理内存
  auto input = std::make_shared<Tensor<float>>(1, input_data.n_elem, 1);
  std::vector<float> in_vals(input_data.memptr(), input_data.memptr() + input_data.n_elem);
  input->Fill(in_vals);
  
  // 设置 4D 形状 (模拟上一层输出)
  input->Reshape({b, seq, three, dim});
  inputs.push_back(input);

  // 3. 创建 Unbind Layer
  // 沿着 dim=2 (大小为3的那一维) 拆分
  UnbindLayer layer(2);
  
  // 预分配输出容器 (模拟 Graph 的行为，或者传入空让 Layer 自己 resize)
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  
  // 4. 推理
  ASSERT_EQ(layer.Forward(inputs, outputs), StatusCode::kSuccess);
  
  // 5. 验证
  ASSERT_EQ(outputs.size(), split_count);
  
  // 验证每个分片的形状和数据
  for (int i = 0; i < split_count; ++i) {
      auto& output = outputs.at(i);
      
      // 验证形状: 应该是 (1, 4, 8) -> (Batch, Seq, Dim)
      // Unbind 移除了 dim=2
      const auto& shapes = output->raw_shapes();
      ASSERT_EQ(shapes.size(), 3);
      ASSERT_EQ(shapes[0], 1);
      ASSERT_EQ(shapes[1], 4);
      ASSERT_EQ(shapes[2], 8);
      
      // 加载对应的 GT
      std::string gt_path = "test_data/unbind/output_" + std::to_string(i) + ".csv";
      arma::fmat gt_data = CSVDataLoader::LoadData<float>(gt_path);
      ASSERT_FALSE(gt_data.empty());
      
      // 逐元素对比
      const float* out_ptr = output->raw_ptr();
      const float* gt_ptr = gt_data.memptr();
      
      ASSERT_EQ(output->size(), gt_data.n_elem);
      
      for (uint32_t k = 0; k < output->size(); ++k) {
          ASSERT_NEAR(out_ptr[k], gt_ptr[k], 1e-5f) 
            << "Mismatch in Split " << i << " at index " << k;
      }
  }
}