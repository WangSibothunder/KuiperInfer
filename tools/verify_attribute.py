import torch
import numpy as np
import os

def save_tensor_to_csv(tensor, filename):
    data = tensor.flatten().detach().numpy()
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')
    print(f"Saved {filename} with shape {tensor.shape}")

def verify_attribute():
    output_dir = "test_data/attribute"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Generating Attribute Layer Data ---")
    
    # 模拟一个常见的 Attribute，比如 Positional Embedding
    # Shape: (1, 197, 192)
    # 这就是 "Ground Truth"，既是权重，也是输出
    attr_value = torch.randn(1, 197, 192)
    
    save_tensor_to_csv(attr_value, f"{output_dir}/output_gt.csv")
    
    print(f"[Info] Attribute Shape: {attr_value.shape}")

if __name__ == "__main__":
    verify_attribute()