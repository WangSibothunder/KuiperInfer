import torch
import numpy as np
import os

def save_tensor_to_csv(tensor, filename):
    data = tensor.flatten().detach().numpy()
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')
    print(f"Saved {filename} with shape {tensor.shape}")

def verify_reshape():
    os.makedirs("test_data/reshape", exist_ok=True)
    
    # 模拟 DeiT: (1, 197, 576) -> (1, 197, 3, 3, 64)
    # Total elements: 113472
    # 为了测试速度，我们缩小尺寸，但保持 5D 结构
    # Input: (1, 4, 18) -> Output: (1, 4, 2, 3, 3)
    # 1*4*18 = 72 elements
    # 1*4*2*3*3 = 72 elements
    
    input_tensor = torch.randn(1, 4, 18)
    # PNNX Reshape shape param: (1, 4, 2, 3, 3)
    output_tensor = input_tensor.reshape(1, 4, 2, 3, 3)
    
    save_tensor_to_csv(input_tensor, "test_data/reshape/input.csv")
    save_tensor_to_csv(output_tensor, "test_data/reshape/output_gt.csv")
    
    print(f"\n[Info] Input Shape: {input_tensor.shape}")
    print(f"[Info] Output Shape: {output_tensor.shape}")

if __name__ == "__main__":
    verify_reshape()