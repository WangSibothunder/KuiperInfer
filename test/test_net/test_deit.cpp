// 2026-王思博
// MIT License
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <fstream>
#include <filesystem>
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"
#include <armadillo>

// 辅助函数: Softmax
std::vector<float> softmax(const arma::fmat& scores) {
    std::vector<float> probs(scores.n_elem);
    float max_val = scores.max();
    float sum = 0.0f;
    for(uint32_t i=0; i<scores.n_elem; ++i) {
        probs[i] = std::exp(scores[i] - max_val);
        sum += probs[i];
    }
    for(uint32_t i=0; i<scores.n_elem; ++i) {
        probs[i] /= sum;
    }
    return probs;
}

// 辅助函数: 获取 TopK
std::vector<std::pair<int, float>> get_topk(const std::vector<float>& probs, int k) {
    std::vector<std::pair<int, float>> pairs;
    for(int i=0; i<probs.size(); ++i) {
        pairs.push_back({i, probs[i]});
    }
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), 
        [](const auto& a, const auto& b){ return a.second > b.second; });
    
    std::vector<std::pair<int, float>> result;
    for(int i=0; i<k; ++i) result.push_back(pairs[i]);
    return result;
}


TEST(test_model, deit_tiny_e2e) {
  using namespace kuiper_infer;
  
  // 1. 路径设置
  const std::string param_path = "deit_tiny.pnnx.param"; 
  const std::string bin_path = "deit_tiny.pnnx.bin";
  const std::string input_csv = "test_data/deit_e2e/input.csv";
  const std::string output_gt_csv = "test_data/deit_e2e/output_gt.csv";
  
  // 检查文件是否存在
  ASSERT_TRUE(std::filesystem::exists(param_path)) << "Param file not found! Copy deit_tiny.pnnx.param to project root.";
  ASSERT_TRUE(std::filesystem::exists(bin_path)) << "Bin file not found! Copy deit_tiny.pnnx.bin to project root.";
  ASSERT_TRUE(std::filesystem::exists(input_csv)) << "Input CSV not found! Run python script first.";

  RuntimeGraph graph(param_path, bin_path);
  
  LOG(INFO) << ">>> [Probe] Building Graph...";
  graph.Build(); 
  LOG(INFO) << ">>> [Probe] Graph Build Complete.";

  const std::string input_name = "pnnx_input_0";
  const std::string output_name = "pnnx_output_0";
  
  arma::fmat input_data = CSVDataLoader::LoadData<float>(input_csv);
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
  std::vector<float> input_vals(input_data.memptr(), input_data.memptr() + input_data.n_elem);
  input->Fill(input_vals); 
  
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  
  LOG(INFO) << ">>> [Probe] Setting Inputs...";
  graph.set_inputs(input_name, inputs);
  
  LOG(INFO) << ">>> [Probe] Starting Forward...";
  // 开启 debug 模式，打印执行到的每个算子名字
  graph.Forward(true); 
  LOG(INFO) << ">>> [Probe] Forward Complete.";
  
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs(output_name);
  ASSERT_FALSE(outputs.empty());
  auto output_tensor = outputs[0]; 
  
  LOG(INFO) << ">>> [Probe] Verifying Output...";
  
  // 加载 GT
  arma::fmat gt_data = CSVDataLoader::LoadData<float>(output_gt_csv);
  const float* out_ptr = output_tensor->raw_ptr();
  const float* gt_ptr = gt_data.memptr();
  
  // A. 余弦相似度检查 (Cosine Similarity)
  double dot_sum = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  
  // Output shape (1, 1000)
  uint32_t size = output_tensor->size();
  CHECK_EQ(size, gt_data.n_elem);

  for(uint32_t i=0; i<size; ++i) {
      float val = out_ptr[i];
      float gt = gt_ptr[i];
      
      dot_sum += val * gt;
      norm_a += val * val;
      norm_b += gt * gt;
  }
  double cos_sim = dot_sum / (std::sqrt(norm_a) * std::sqrt(norm_b));
  
  LOG(INFO) << "Cosine Similarity: " << cos_sim;
  ASSERT_GT(cos_sim, 0.99) << "Inference accuracy too low!";

  // B. Top-K 验证
  auto probs = softmax(output_tensor->data().slice(0)); 
  auto top5 = get_topk(probs, 5);
  
  LOG(INFO) << "C++ Top-5 Prediction:";
  for(const auto& p : top5) {
      LOG(INFO) << "Class ID: " << p.first << ", Prob: " << p.second;
  }
}