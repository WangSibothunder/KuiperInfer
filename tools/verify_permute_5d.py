import torch
import numpy as np
import os

def save_tensor_to_csv(tensor, filename):
    data = tensor.flatten().detach().numpy()
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')
    print(f"Saved {filename} with shape {tensor.shape}")

def verify_permute_5d():
    os.makedirs("test_data/permute_5d", exist_ok=True)
    
    # 模拟 DeiT 的 5D 结构，但缩小尺寸以加速验证
    # 原始: (1, 197, 3, 3, 64)
    # 测试: Batch=1, Seq=4, Head=2, Window=3, Dim=2
    # Input Shape: (1, 4, 2, 3, 2)
    input_tensor = torch.randn(1, 4, 2, 3, 2)
    
    # 模拟 PNNX 的 dims=(2, 0, 3, 1, 4)
    # Output Shape calculation:
    # dim 0 from input dim 2 -> size 2
    # dim 1 from input dim 0 -> size 1
    # dim 2 from input dim 3 -> size 3
    # dim 3 from input dim 1 -> size 4
    # dim 4 from input dim 4 -> size 2
    # Expected Output: (2, 1, 3, 4, 2)
    output_tensor = input_tensor.permute(2, 0, 3, 1, 4)
    
    save_tensor_to_csv(input_tensor, "test_data/permute_5d/input.csv")
    save_tensor_to_csv(output_tensor, "test_data/permute_5d/output_gt.csv")
    
    print(f"\n[Info] Input Shape: {input_tensor.shape}")
    print(f"[Info] Output Shape: {output_tensor.shape}")
    print(f"[Info] PNNX Dims: (2, 0, 3, 1, 4)")

if __name__ == "__main__":
    verify_permute_5d()