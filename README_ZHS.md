# RMBG 2.0 - ComfyUI 自定义节点 v1.0.0

使用 RMBG 2.0 模型（BiRefNet 架构）进行高级背景移除的强大 ComfyUI 自定义节点。

## 简介

RMBG 2.0 是一个高性能的 ComfyUI 背景移除节点，采用最先进的 BiRefNet 架构。模型在首次使用时会自动从 Hugging Face 下载，并提供专业级别的背景分割功能，带有实时进度追踪。

## 功能特点

- **自动下载模型**：首次使用时自动从 Hugging Face 下载模型文件并显示进度
- **本地模型检测**：优先检查 `models/RMBG/RMBG-2.0` 目录中的本地文件
- **高质量结果**：基于 BiRefNet 架构，在 15,000+ 高质量图像上训练
- **灵活输出**：支持 RGBA、带遮罩的 RGB 和仅遮罩输出模式
- **批量处理**：高效处理多张图像
- **GPU 加速**：自动使用 CUDA（如可用）
- **可调灵敏度**：使用灵敏度参数微调前景检测

## 安装方法

### 方法 1：通过 Git 克隆

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/S4MUEL-404/RMBG-2.0
cd RMBG-2.0
pip install -r requirements.txt
```

### 方法 2：手动安装

1. 下载此仓库
2. 解压到 `ComfyUI/custom_nodes/RMBG-2.0`
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 依赖项

- torch>=2.0.0
- torchvision>=0.15.0
- transformers>=4.30.0
- huggingface_hub>=0.16.0
- Pillow>=9.0.0
- numpy>=1.21.0
- safetensors>=0.3.0
- tqdm>=4.65.0

## 使用方法

### 在 ComfyUI 中使用

1. 启动 ComfyUI
2. 在节点菜单的 `image/segmentation` 下找到 "RMBG 2.0 - Remove Background" 节点
3. 连接图像输入
4. 配置参数：
   - **process_res**：处理分辨率（512-2048，默认：1024）
   - **sensitivity**：前景检测灵敏度（0.0-1.0，默认：0.5）
   - **output_type**：选择 RGBA、带遮罩的 RGB 或仅遮罩
5. 首次使用时模型会自动下载（约 844MB）

### 模型文件

节点自动管理以下位置的模型文件：
- **ComfyUI 模式**：`ComfyUI/models/RMBG/RMBG-2.0/`
- **独立模式**：`RMBG-2.0/models/RMBG/RMBG-2.0/`

模型文件包括：
- config.json
- model.safetensors（844MB）
- birefnet.py
- BiRefNet_config.py

### 参数说明

- **image**（必需）：输入图像张量
- **process_res**（默认：1024）：处理分辨率 - 更高的值提供更好的质量但处理更慢
- **sensitivity**（默认：0.5）：调整前景检测灵敏度 - 较低的值用于保守分割，较高用于激进分割
- **output_type**：
  - **RGBA**：输出带透明背景的图像
  - **RGB with Mask**：输出原始 RGB 图像和单独的遮罩
  - **Mask Only**：仅输出分割遮罩

### 输出

- **image**：根据 output_type 处理的图像
- **mask**：二值分割遮罩

## 示例工作流

```
加载图像 → RMBG 2.0 → 保存图像
             ↓
          预览
```

## 技术细节

- **模型**：BiRefNet（双边参考网络）
- **架构**：高分辨率二分图像分割
- **训练**：15,000+ 专业标注图像
- **分辨率**：针对 1024x1024 处理优化
- **设备**：自动 CUDA/CPU 选择

## 许可证

本项目遵循 GPL-3.0 许可证。

模型许可证：RMBG 2.0 模型基于 BiRefNet，并遵循其相应的许可条款。

## 作者

**S4MUEL**

- 主页：https://s4muel.com
- GitHub：https://github.com/S4MUEL-404/RMBG-2.0

## 致谢

- Zheng 等人的 BiRefNet 架构
- RMBG 2.0 训练和优化
- ComfyUI 框架

## 更新日志

### v1.0.0（2024-12-19）
- 初始发布
- 从 Hugging Face 自动下载
- 本地模型检测
- 下载进度追踪
- 多种输出模式
- 批量处理支持
