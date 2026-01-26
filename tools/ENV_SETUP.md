# DeiT-Tiny E2E 验证环境配置

## ✓ 已完成的配置

### 1. Python 脚本配置
- **文件**: `tools/verify_deit_e2e.py`
- **状态**: ✓ 已完成
- **功能**:
  - 加载预训练 DeiT-Tiny 模型
  - 使用 `imgs/car.jpg` 或随机张量作为输入
  - 运行 PyTorch 推理
  - 保存输入输出为 CSV 格式供 C++ 验证
  - 导出 PNNX 模型格式

### 2. Python 依赖
```
 torch
 timm
 numpy
 PIL (Pillow)
 torchvision
 cv2 (OpenCV)
```

### 3. 输入数据准备
- **图片**: `imgs/car.jpg` 已存在 (101KB)
- **预处理**: ImageNet 标准化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **输入尺寸**: (1, 3, 224, 224) - Batch=1, RGB, 224x224

### 4. 输出数据位置
- **工作目录**: `test_data/deit_e2e/`
- **输入 CSV**: `test_data/deit_e2e/input.csv` - 169,632 个浮点数
- **输出真值**: `test_data/deit_e2e/output_gt.csv` - 1000 个类别的 logits

### 5. 模型导出
- **TorchScript**: `deit_tiny.pt`
- **PNNX 命令**: `./pnnx deit_tiny.pt inputshape=[1,3,224,224]`
- **输出格式**: PNNX 参数和二进制文件

## 使用步骤

### Step 1: 运行 Python 验证脚本
```bash
cd /code/KuiperInfer/tools
python3 verify_deit_e2e.py
```

**预期输出**:
```
Loading DeiT-Tiny model...
Processing ../imgs/car.jpg...
Saved test_data/deit_e2e/input.csv with shape torch.Size([1, 3, 224, 224])
Saved test_data/deit_e2e/output_gt.csv with shape torch.Size([1, 1000])
PyTorch Top-1 Class ID: XX, Prob: X.XXXX
Exporting PNNX for safety...
```

### Step 2: 检查生成的文件
```bash
ls -lah test_data/deit_e2e/
ls -lah *.pt *.pnnx* 2>/dev/null
```

### Step 3: 编译并运行 C++ 测试
```bash
cd /code/KuiperInfer
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --target test_kuiper --config Debug
./test/test_kuiper --gtest_filter="*deit*"
```

## 关键文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `tools/verify_deit_e2e.py` | Python 验证脚本 | ✓ |
| `imgs/car.jpg` | 测试图片 | ✓ |
| `test/test_deit.cpp` | C++ 验证代码 | ✓ |
| `include/data/load_data.hpp` | CSV 加载器 | ✓ |
| `deit_tiny.pt` | TorchScript 模型 (需要生成) | - |
| `deit_tiny.pnnx.param` | PNNX 参数文件 (需要生成) | - |
| `deit_tiny.pnnx.bin` | PNNX 二进制文件 (需要生成) | - |

## 故障排除

### 问题1: PNNX 工具不存在
**解决**: 需要在当前目录放置 `pnnx` 可执行文件或安装相应工具链

### 问题2: car.jpg 不存在
**解决**: 脚本会自动使用随机张量替代，生成的测试数据仍然有效

### 问题3: 内存不足
**解决**: DeiT-Tiny 相对轻量，通常需要 < 2GB VRAM，确保系统有足够内存

## 环境检查命令

```bash
# 检查 Python 包
python3 -c "import torch, timm, numpy as np; print('✓ All OK')"

# 检查文件
test -f tools/verify_deit_e2e.py && echo "✓ Script exists"
test -f imgs/car.jpg && echo "✓ Image exists"

# 检查路径
cd /code/KuiperInfer/tools && python3 verify_deit_e2e.py
```

---
 2026-01-26
