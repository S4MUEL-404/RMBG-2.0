# RMBG-2.0 ComfyUI Node
# Remove Background using RMBG 2.0 model
# Source: https://github.com/S4MUEL-404/RMBG-2.0

import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import sys
import importlib.util
import types
from tqdm import tqdm

# Try to import ComfyUI folder_paths
try:
    import folder_paths
    COMFYUI_MODE = True
except ImportError:
    COMFYUI_MODE = False
    class FolderPaths:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        
        @staticmethod
        def add_model_folder_path(name, path):
            os.makedirs(path, exist_ok=True)
    
    folder_paths = FolderPaths()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Add RMBG model folder
if COMFYUI_MODE:
    folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

# Model configuration for RMBG-2.0
RMBG_MODEL_CONFIG = {
    "repo_id": "S4MUEL/RMBG-2.0",
    "files": {
        "config.json": "config.json",
        "model.safetensors": "model.safetensors",
        "birefnet.py": "birefnet.py",
        "BiRefNet_config.py": "BiRefNet_config.py"
    },
    "cache_dir": "RMBG-2.0"
}

# Utility functions
def tensor2pil(image):
    """Convert torch tensor to PIL Image"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL Image to torch tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ProgressCallback:
    """Custom progress callback for huggingface_hub downloads"""
    def __init__(self, filename):
        self.filename = filename
        self.pbar = None
        
    def __call__(self, downloaded_size, total_size):
        if self.pbar is None and total_size is not None:
            self.pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {self.filename}"
            )
        if self.pbar is not None:
            self.pbar.update(downloaded_size - self.pbar.n)
            
    def close(self):
        if self.pbar is not None:
            self.pbar.close()

class RMBG20Node:
    """RMBG 2.0 Background Removal Node for ComfyUI"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        if COMFYUI_MODE:
            self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")
        else:
            self.base_cache_dir = os.path.join(os.path.dirname(__file__), "models", "RMBG")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "process_res": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 128,
                    "display": "number"
                }),
                "sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "output_type": (["RGBA", "RGB with Mask", "Mask Only"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "image/segmentation"
    
    def get_cache_dir(self):
        """Get model cache directory"""
        cache_path = os.path.join(self.base_cache_dir, RMBG_MODEL_CONFIG["cache_dir"])
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    
    def check_model_files(self):
        """Check if all model files exist locally"""
        cache_dir = self.get_cache_dir()
        
        if not os.path.exists(cache_dir):
            return False, []
        
        missing_files = []
        for filename in RMBG_MODEL_CONFIG["files"].keys():
            file_path = os.path.join(cache_dir, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)
        
        if missing_files:
            return False, missing_files
        
        return True, []
    
    def download_model_files(self):
        """Download model files from Hugging Face with progress"""
        cache_dir = self.get_cache_dir()
        
        print(f"\n[RMBG-2.0] Downloading model files from {RMBG_MODEL_CONFIG['repo_id']}...")
        print(f"[RMBG-2.0] Saving to: {cache_dir}")
        
        try:
            for filename in RMBG_MODEL_CONFIG["files"].keys():
                file_path = os.path.join(cache_dir, filename)
                
                # Skip if file already exists
                if os.path.exists(file_path):
                    print(f"[RMBG-2.0] ✓ {filename} already exists")
                    continue
                
                print(f"\n[RMBG-2.0] Downloading {filename}...")
                
                # Download with progress tracking
                downloaded_path = hf_hub_download(
                    repo_id=RMBG_MODEL_CONFIG["repo_id"],
                    filename=filename,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                
                print(f"[RMBG-2.0] ✓ {filename} downloaded successfully")
            
            print(f"\n[RMBG-2.0] All model files downloaded successfully!\n")
            return True
            
        except Exception as e:
            print(f"\n[RMBG-2.0] ✗ Error downloading model files: {str(e)}\n")
            raise RuntimeError(f"Failed to download model files: {str(e)}")
    
    def load_model(self):
        """Load RMBG 2.0 model"""
        if self.model_loaded:
            return
        
        # Check if model files exist, download if necessary
        files_exist, missing = self.check_model_files()
        if not files_exist:
            print(f"[RMBG-2.0] Model files not found. Missing: {', '.join(missing)}")
            self.download_model_files()
        else:
            print("[RMBG-2.0] Model files found locally")
        
        cache_dir = self.get_cache_dir()
        
        print("[RMBG-2.0] Loading model...")
        
        try:
            # Load model using transformers compatible method
            from transformers import PreTrainedModel
            import json
            
            config_path = os.path.join(cache_dir, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            birefnet_path = os.path.join(cache_dir, "birefnet.py")
            BiRefNetConfig_path = os.path.join(cache_dir, "BiRefNet_config.py")
            
            # Load BiRefNet_config module
            config_spec = importlib.util.spec_from_file_location("BiRefNetConfig", BiRefNetConfig_path)
            config_module = importlib.util.module_from_spec(config_spec)
            sys.modules["BiRefNetConfig"] = config_module
            config_spec.loader.exec_module(config_module)
            
            # Load and fix birefnet module
            with open(birefnet_path, 'r', encoding='utf-8') as f:
                birefnet_content = f.read()
            
            # Fix import statement
            birefnet_content = birefnet_content.replace(
                "from .BiRefNet_config import BiRefNetConfig",
                "from BiRefNetConfig import BiRefNetConfig"
            )
            
            # Create and execute module
            module_name = f"custom_birefnet_rmbg20_{hash(birefnet_path)}"
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
            exec(birefnet_content, module.__dict__)
            
            # Find model class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr != PreTrainedModel:
                    BiRefNetConfig = getattr(config_module, "BiRefNetConfig")
                    model_config = BiRefNetConfig()
                    self.model = attr(model_config)
                    
                    # Load weights
                    weights_path = os.path.join(cache_dir, "model.safetensors")
                    try:
                        import safetensors.torch
                        self.model.load_state_dict(safetensors.torch.load_file(weights_path))
                    except ImportError:
                        from transformers.modeling_utils import load_state_dict
                        state_dict = load_state_dict(weights_path)
                        self.model.load_state_dict(state_dict)
                    
                    break
            
            if self.model is None:
                raise RuntimeError("Could not find suitable model class in birefnet.py")
            
            # Prepare model for inference
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            
            self.model_loaded = True
            print(f"[RMBG-2.0] Model loaded successfully on {device}")
            
        except Exception as e:
            print(f"[RMBG-2.0] Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load RMBG-2.0 model: {str(e)}")
    
    def remove_background(self, image, process_res=1024, sensitivity=0.5, output_type="RGBA"):
        """
        Remove background from image
        
        Args:
            image: Input image tensor (B, H, W, C)
            process_res: Processing resolution
            sensitivity: Sensitivity for foreground detection (0-1)
            output_type: Output format
            
        Returns:
            Tuple of (processed_image, mask)
        """
        # Load model if not loaded
        self.load_model()
        
        # Prepare transform
        transform_image = transforms.Compose([
            transforms.Resize((process_res, process_res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Process batch
        batch_results = []
        batch_masks = []
        
        for img_tensor in image:
            # Convert to PIL
            pil_img = tensor2pil(img_tensor)
            orig_w, orig_h = pil_img.size
            
            # Transform and process
            input_tensor = transform_image(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                # Handle different output formats
                if isinstance(outputs, list) and len(outputs) > 0:
                    result = outputs[-1].sigmoid().cpu()
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    result = outputs['logits'].sigmoid().cpu()
                else:
                    result = outputs.sigmoid().cpu()
                
                result = result.squeeze()
                
                # Apply sensitivity adjustment
                result = result * (1 + (1 - sensitivity))
                result = torch.clamp(result, 0, 1)
                
                # Resize to original size
                result = F.interpolate(
                    result.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            # Convert mask to PIL
            mask_pil = tensor2pil(result)
            
            # Generate output based on type
            if output_type == "RGBA":
                # Create RGBA image
                rgba_img = pil_img.convert("RGB")
                rgba_img.putalpha(mask_pil)
                output_img = pil2tensor(rgba_img)
            elif output_type == "RGB with Mask":
                # Keep RGB, return mask separately
                output_img = img_tensor.unsqueeze(0)
            else:  # Mask Only
                # Return mask as grayscale image
                mask_rgb = mask_pil.convert("RGB")
                output_img = pil2tensor(mask_rgb)
            
            batch_results.append(output_img)
            batch_masks.append(pil2tensor(mask_pil).squeeze(0))
        
        # Stack results
        result_images = torch.cat(batch_results, dim=0)
        result_masks = torch.stack(batch_masks, dim=0)
        
        return (result_images, result_masks)

# Node registration
NODE_CLASS_MAPPINGS = {
    "RMBG20": RMBG20Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RMBG20": "RMBG 2.0 - Remove Background"
}
