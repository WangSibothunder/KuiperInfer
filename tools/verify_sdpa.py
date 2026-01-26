import torch
import numpy as np
import os
import torch.nn.functional as F

def save_tensor_to_csv(tensor, filename):
    data = tensor.flatten().detach().numpy()
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')

def verify_sdpa():
    os.makedirs("test_data/sdpa", exist_ok=True)
    
    # 模拟参数: Batch=1, Head=2, Seq=4, Dim=8
    B, H, S, D = 1, 2, 4, 8
    
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    
    # 标准 SDPA
    # scale = 1 / sqrt(D)
    output = F.scaled_dot_product_attention(q, k, v, scale=1.0/np.sqrt(D))
    
    save_tensor_to_csv(q, "test_data/sdpa/q.csv")
    save_tensor_to_csv(k, "test_data/sdpa/k.csv")
    save_tensor_to_csv(v, "test_data/sdpa/v.csv")
    save_tensor_to_csv(output, "test_data/sdpa/out_gt.csv")
    
    print("SDPA data generated.")

if __name__ == "__main__":
    verify_sdpa()