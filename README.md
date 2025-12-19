# RMBG 2.0 - ComfyUI Custom Node v1.0.0

A powerful ComfyUI custom node for advanced background removal using the RMBG 2.0 model (BiRefNet architecture).

## Introduction

RMBG 2.0 is a high-performance background removal node for ComfyUI that leverages the state-of-the-art BiRefNet architecture. The model automatically downloads from Hugging Face on first use and provides professional-grade background segmentation with real-time progress tracking.

## Features

- **Automatic Model Download**: First-time use automatically downloads model files from Hugging Face with progress display
- **Local Model Detection**: Checks `models/RMBG/RMBG-2.0` directory first before downloading
- **High-Quality Results**: Based on BiRefNet architecture, trained on 15,000+ high-quality images
- **Flexible Output**: Supports RGBA, RGB with Mask, and Mask Only output modes
- **Batch Processing**: Efficiently processes multiple images
- **GPU Acceleration**: Automatically uses CUDA when available
- **Adjustable Sensitivity**: Fine-tune foreground detection with sensitivity parameter

## Installation

### Method 1: Clone via Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/S4MUEL-404/RMBG-2.0
cd RMBG-2.0
pip install -r requirements.txt
```

### Method 2: Manual Installation

1. Download this repository
2. Extract to `ComfyUI/custom_nodes/RMBG-2.0`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- torch>=2.0.0
- torchvision>=0.15.0
- transformers>=4.30.0
- huggingface_hub>=0.16.0
- Pillow>=9.0.0
- numpy>=1.21.0
- safetensors>=0.3.0
- tqdm>=4.65.0

## Usage

### In ComfyUI

1. Launch ComfyUI
2. Find "RMBG 2.0 - Remove Background" node in the node menu under `image/segmentation`
3. Connect an image input
4. Configure parameters:
   - **process_res**: Processing resolution (512-2048, default: 1024)
   - **sensitivity**: Foreground detection sensitivity (0.0-1.0, default: 0.5)
   - **output_type**: Choose from RGBA, RGB with Mask, or Mask Only
5. The model will auto-download on first use (approximately 844MB)

### Model Files

The node automatically manages model files in:
- **ComfyUI Mode**: `ComfyUI/models/RMBG/RMBG-2.0/`
- **Standalone Mode**: `RMBG-2.0/models/RMBG/RMBG-2.0/`

Model files include:
- config.json
- model.safetensors (844MB)
- birefnet.py
- BiRefNet_config.py

### Parameters

- **image** (required): Input image tensor
- **process_res** (default: 1024): Processing resolution - higher values provide better quality but slower processing
- **sensitivity** (default: 0.5): Adjust foreground detection sensitivity - lower values for conservative segmentation, higher for aggressive
- **output_type**: 
  - **RGBA**: Outputs image with transparent background
  - **RGB with Mask**: Outputs original RGB image and separate mask
  - **Mask Only**: Outputs only the segmentation mask

### Output

- **image**: Processed image according to output_type
- **mask**: Binary segmentation mask

## Example Workflow

```
Load Image → RMBG 2.0 → Save Image
              ↓
           Preview
```

## Technical Details

- **Model**: BiRefNet (Bilateral Reference Network)
- **Architecture**: High-resolution dichotomous image segmentation
- **Training**: 15,000+ professionally labeled images
- **Resolution**: Optimized for 1024x1024 processing
- **Device**: Automatic CUDA/CPU selection

## License

This project follows GPL-3.0 License.

Model License: The RMBG 2.0 model is based on BiRefNet and is available under its respective license terms.

## Author

**S4MUEL**

- Homepage: https://s4muel.com
- GitHub: https://github.com/S4MUEL-404/RMBG-2.0

## Acknowledgments

- BiRefNet architecture by Zheng et al.
- RMBG 2.0 training and optimization
- ComfyUI framework

## Changelog

### v1.0.0 (2024-12-19)
- Initial release
- Auto-download from Hugging Face
- Local model detection
- Progress tracking for downloads
- Multiple output modes
- Batch processing support
