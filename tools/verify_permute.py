import torch
import numpy as np
import os

def save_tensor_to_csv(tensor, filename):
    # 保存为一维 Flatten 数据，方便 C++ 逐元素加载对比
    data = tensor.flatten().detach().numpy()
    # 使用 %.6f 保证精度
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')
    print(f"Saved {filename} with shape {tensor.shape}")

def verify_permute():
    os.makedirs("test_data/permute", exist_ok=True)
    
    # 模拟 DeiT: Batch=1, Seq=4(缩减), Head=2(缩减), Dim=3(缩减)
    # 原始: (1, 197, 3, 64) -> 我们用 (1, 4, 2, 3) 做测试
    # 对应 PNNX/Kuiper维度: Channel=Seq=4, Row=Head=2, Col=Dim=3 (假设 dim=1 映射到 Channel)
    input_tensor = torch.randn(1, 4, 2, 3) 
    
    # PNNX Permute: (0, 2, 1, 3) 
    # 交换 dim 1 (Seq) 和 dim 2 (Head)
    # Output: (1, 2, 4, 3)
    output_tensor = input_tensor.permute(0, 2, 1, 3)
    
    save_tensor_to_csv(input_tensor, "test_data/permute/input.csv")
    save_tensor_to_csv(output_tensor, "test_data/permute/output_gt.csv")
    
    print(f"\n[Info] Input Shape: {input_tensor.shape}")
    print(f"[Info] Output Shape: {output_tensor.shape}")

if __name__ == "__main__":
    verify_permute()