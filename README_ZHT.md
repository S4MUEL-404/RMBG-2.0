# RMBG 2.0 - ComfyUI 自訂節點 v1.0.0

使用 RMBG 2.0 模型（BiRefNet 架構）進行高級背景移除的強大 ComfyUI 自訂節點。

## 簡介

RMBG 2.0 是一個高性能的 ComfyUI 背景移除節點，採用最先進的 BiRefNet 架構。模型在首次使用時會自動從 Hugging Face 下載，並提供專業級別的背景分割功能，帶有實時進度追蹤。

## 功能特點

- **自動下載模型**：首次使用時自動從 Hugging Face 下載模型文件並顯示進度
- **本地模型檢測**：優先檢查 `models/RMBG/RMBG-2.0` 目錄中的本地文件
- **高質量結果**：基於 BiRefNet 架構，在 15,000+ 高質量圖像上訓練
- **靈活輸出**：支持 RGBA、帶遮罩的 RGB 和僅遮罩輸出模式
- **批量處理**：高效處理多張圖像
- **GPU 加速**：自動使用 CUDA（如可用）
- **可調靈敏度**：使用靈敏度參數微調前景檢測

## 安裝方法

### 方法 1：通過 Git 克隆

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/S4MUEL-404/RMBG-2.0
cd RMBG-2.0
pip install -r requirements.txt
```

### 方法 2：手動安裝

1. 下載此倉庫
2. 解壓到 `ComfyUI/custom_nodes/RMBG-2.0`
3. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

## 依賴項

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

1. 啟動 ComfyUI
2. 在節點菜單的 `image/segmentation` 下找到 "RMBG 2.0 - Remove Background" 節點
3. 連接圖像輸入
4. 配置參數：
   - **process_res**：處理分辨率（512-2048，默認：1024）
   - **sensitivity**：前景檢測靈敏度（0.0-1.0，默認：0.5）
   - **output_type**：選擇 RGBA、帶遮罩的 RGB 或僅遮罩
5. 首次使用時模型會自動下載（約 844MB）

### 模型文件

節點自動管理以下位置的模型文件：
- **ComfyUI 模式**：`ComfyUI/models/RMBG/RMBG-2.0/`
- **獨立模式**：`RMBG-2.0/models/RMBG/RMBG-2.0/`

模型文件包括：
- config.json
- model.safetensors（844MB）
- birefnet.py
- BiRefNet_config.py

### 參數說明

- **image**（必需）：輸入圖像張量
- **process_res**（默認：1024）：處理分辨率 - 更高的值提供更好的質量但處理更慢
- **sensitivity**（默認：0.5）：調整前景檢測靈敏度 - 較低的值用於保守分割，較高用於激進分割
- **output_type**：
  - **RGBA**：輸出帶透明背景的圖像
  - **RGB with Mask**：輸出原始 RGB 圖像和單獨的遮罩
  - **Mask Only**：僅輸出分割遮罩

### 輸出

- **image**：根據 output_type 處理的圖像
- **mask**：二值分割遮罩

## 示例工作流

```
加載圖像 → RMBG 2.0 → 保存圖像
             ↓
          預覽
```

## 技術細節

- **模型**：BiRefNet（雙邊參考網絡）
- **架構**：高分辨率二分圖像分割
- **訓練**：15,000+ 專業標註圖像
- **分辨率**：針對 1024x1024 處理優化
- **設備**：自動 CUDA/CPU 選擇

## 許可證

本項目遵循 GPL-3.0 許可證。

模型許可證：RMBG 2.0 模型基於 BiRefNet，並遵循其相應的許可條款。

## 作者

**S4MUEL**

- 主頁：https://s4muel.com
- GitHub：https://github.com/S4MUEL-404/RMBG-2.0

## 致謝

- Zheng 等人的 BiRefNet 架構
- RMBG 2.0 訓練和優化
- ComfyUI 框架

## 更新日誌

### v1.0.0（2024-12-19）
- 初始發布
- 從 Hugging Face 自動下載
- 本地模型檢測
- 下載進度追蹤
- 多種輸出模式
- 批量處理支持
