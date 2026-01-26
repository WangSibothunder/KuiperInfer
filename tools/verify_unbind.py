import torch
import numpy as np
import os

def save_tensor_to_csv(tensor, filename):
    # Flatten 并保存，fmt='%.6f' 保证精度
    data = tensor.flatten().detach().numpy()
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')
    print(f"Saved {filename} with shape {tensor.shape}")

def verify_unbind():
    output_dir = "test_data/unbind"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Generating Unbind (QKV Split) Data ---")
    
    # 模拟 DeiT-Tiny 的 Attention 输入
    # Shape: (Batch=1, Seq=4, 3, Dim=8) 
    # (为了测试速度，我们缩小 Seq 和 Dim，但保留 3 这个关键维度)
    B, Seq, Three, Dim = 1, 4, 3, 8
    
    input_tensor = torch.randn(B, Seq, Three, Dim)
    
    # 动作: 沿着第2维 (索引为2，大小为3) 拆分
    # 结果应该是 3 个 (B, Seq, Dim) 的张量
    outputs = torch.unbind(input_tensor, dim=2)
    
    # 保存输入
    save_tensor_to_csv(input_tensor, f"{output_dir}/input.csv")
    
    # 保存输出 (Q, K, V)
    for i, out in enumerate(outputs):
        save_tensor_to_csv(out, f"{output_dir}/output_{i}.csv")
        
    print(f"[Info] Input Shape: {input_tensor.shape}")
    print(f"[Info] Unbind Dim: 2")
    print(f"[Info] Output Count: {len(outputs)}")
    print(f"[Info] Output Shape: {outputs[0].shape}")

if __name__ == "__main__":
    verify_unbind()