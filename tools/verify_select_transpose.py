import torch
import numpy as np
import os

def save_tensor_to_csv(tensor, filename):
    # Flatten 并保存
    data = tensor.flatten().detach().numpy()
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')
    print(f"Saved {filename} with shape {tensor.shape}")

def verify_ops():
    output_dir = "test_data/select_transpose"
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # Case 1: Tensor.select
    # ==========================================
    print("\n--- Generating Select Data ---")
    # 模拟 DeiT 输入: (Batch=1, Seq=4, Dim=8)
    # 我们选择 dim=1 (Seq), index=0 (第一个 token)
    input_sel = torch.randn(1, 4, 8)
    
    # 动作: 取第0个序列元素
    # Output Shape: (1, 8) (PyTorch select 会移除该维度)
    output_sel = input_sel.select(dim=1, index=0)
    
    save_tensor_to_csv(input_sel, f"{output_dir}/select_input.csv")
    save_tensor_to_csv(output_sel, f"{output_dir}/select_output_gt.csv")
    print(f"Select Expect Shape: {output_sel.shape}")

    # ==========================================
    # Case 2: torch.transpose
    # ==========================================
    print("\n--- Generating Transpose Data ---")
    # 模拟: (Batch=1, Dim=8, Seq=4)
    input_trans = torch.randn(1, 8, 4)
    
    # 动作: 交换 dim 1 和 dim 2
    # Output Shape: (1, 4, 8)
    output_trans = torch.transpose(input_trans, 1, 2)
    
    save_tensor_to_csv(input_trans, f"{output_dir}/transpose_input.csv")
    save_tensor_to_csv(output_trans, f"{output_dir}/transpose_output_gt.csv")
    print(f"Transpose Expect Shape: {output_trans.shape}")

if __name__ == "__main__":
    verify_ops()