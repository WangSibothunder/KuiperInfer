import torch
import timm
import os

def export_deit_pnnx():
    # 1. 加载预训练的 DeiT-Tiny 模型
    # pretrained=True 会自动下载权重
    print("Loading DeiT-Tiny model...")
    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    model.eval()

    # 2. 定义输入 Tensor (Batch Size=1, RGB, 224x224)
    x = torch.randn(1, 3, 224, 224)

    # 3. 导出为 TorchScript 模型 (.pt)
    # PNNX 需要 TorchScript 作为输入
    print("Tracing model to TorchScript...")
    traced_script_module = torch.jit.trace(model, x)
    
    output_pt_path = "deit_tiny.pt"
    traced_script_module.save(output_pt_path)
    print(f"Model saved to {output_pt_path}")

    # 4. 调用 PNNX 进行转换 (使用系统命令)
    # pnnx 能够优化计算图，并将 torch算子转换为更底层的 ncnn/pnnx 算子
    # inputshape 是必须的，用于静态图推断
    print("Converting to PNNX format...")
    pnnx_cmd = f"pnnx {output_pt_path} inputshape=[1,3,224,224]"
    os.system(pnnx_cmd)
    
    print("\nExport finished! Please check 'deit_tiny.ncnn.param' and 'deit_tiny.ncnn.bin'")

if __name__ == "__main__":
    export_deit_pnnx()