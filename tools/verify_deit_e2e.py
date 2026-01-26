import torch
import timm
import numpy as np
import os
# import cv2
from PIL import Image
from torchvision import transforms

def save_tensor_to_csv(tensor, filename):
    # Flatten 并保存，高精度
    data = tensor.flatten().detach().numpy()
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')
    print(f"Saved {filename} with shape {tensor.shape}")

def run_e2e_verification():
    work_dir = "test_data/deit_e2e"
    os.makedirs(work_dir, exist_ok=True)
    
    # 1. 加载模型
    print("Loading DeiT-Tiny model...")
    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    model.eval()
    
    # 2. 准备图片 (使用 imgs/car.jpg 或随机生成)
    img_path = "../imgs/car.jpg" 
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, using random tensor instead.")
        input_tensor = torch.randn(1, 3, 224, 224)
    else:
        print(f"Processing {img_path}...")
        # 标准 ImageNet 预处理
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0) # Add batch dim -> (1, 3, 224, 224)

    # 3. 运行推理
    with torch.no_grad():
        output_logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output_logits, dim=1)
    
    # 获取 Top-5 结果
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print(f"\nPyTorch Top-1 Class ID: {top5_catid[0][0].item()}, Prob: {top5_prob[0][0].item():.4f}")
    
    # 4. 保存数据供 C++ 验证
    # Input: (1, 3, 224, 224)
    save_tensor_to_csv(input_tensor, f"{work_dir}/input.csv")
    # Output: (1, 1000)
    save_tensor_to_csv(output_logits, f"{work_dir}/output_gt.csv")
    
    # 同时导出最新的 PNNX (防止之前步骤漏掉)
    print("\nExporting PNNX for safety...")
    traced_model = torch.jit.trace(model, input_tensor)
    traced_model.save("deit_tiny.pt")
    # 请确保 pnnx 工具在系统路径或当前目录
    os.system("./pnnx deit_tiny.pt inputshape=[1,3,224,224]")

if __name__ == "__main__":
    run_e2e_verification()
