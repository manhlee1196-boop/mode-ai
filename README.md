# mode-ai 🎨 - FLUX.1-schnell GGUF Workflow Toolkit

Hệ thống workflow tạo ảnh tối ưu cho **FLUX.1-schnell GGUF Q5_K_S**, chạy mượt trên **Colab T4 16GB free** với tổng model **~12.3GB (<15GB)**.

## ✨ Tính năng

- 🚀 **5 workflows ComfyUI JSON** tối ưu sẵn: simple, optimized (main), realistic, hires, inpaint
- 🛠️ **Python Builder**: Tạo workflow programmatically với `FluxWorkflowBuilder`
- 🎨 **Prompt Enhancer**: Style presets (anime, photorealistic, Vietnamese, cinematic...)
- 🌐 **ComfyUI API Client**: Queue workflow tới ComfyUI server
- 📊 **Validation**: Kiểm tra workflow tự động
- 🖥️ **Gradio App**: UI tạo workflow trực quan
- ⚙️ **GitHub Actions**: CI/CD workflow tạo ảnh tự động

## 📦 Model (tổng ~12.3GB)

| File | Size | ComfyUI path |
|---|---|---|
| `flux1-schnell-Q5_K_S.gguf` | 8.26 GB | `models/unet/` |
| `t5-v1_1-xxl-encoder-Q4_K_M.gguf` | 2.9 GB | `models/clip/` |
| `clip_l.safetensors` | 250 MB | `models/clip/` |
| `ae.safetensors` | 320 MB | `models/vae/` |
| `face_yolov8m.pt` | 52 MB | `models/ultralytics/bbox/` |
| `hand_yolov8s.pt` | 22 MB | `models/ultralytics/bbox/` |
| `foot_anime_yolo11m_v3.pt` | ~40 MB | `models/ultralytics/bbox/` |
| `sam_vit_b_01ec64.pth` | 375 MB | `models/sams/` |

## 🚀 Quick Start

### 1. Cài đặt

```bash
git clone https://github.com/manhlee1196-boop/mode-ai
cd mode-ai
pip install -r requirements.txt
```

### 2. Tạo workflow bằng CLI

```bash
# Simple test nhanh
python scripts/generate.py --prompt "1girl, cherry blossoms" --workflow simple --output workflows/test.json

# Optimized đầy đủ face+hand+foot detailer (main)
python scripts/generate.py --prompt "beautiful girl, school uniform" --style aesthetic_anime --workflow optimized --width 1024 --height 1024 --seed 42

# Realistic Vietnamese
python scripts/generate.py --preset vietnamese_aodai --style realistic_vietnamese --workflow realistic --width 832 --height 1216

# List styles và presets
python scripts/generate.py --list-styles
python scripts/generate.py --list-presets

# Validate tất cả workflows
python scripts/validate.py
```

### 3. Python API

```python
from src.mode_ai import FluxWorkflowBuilder, WorkflowConfig
from src.mode_ai.config import MODEL_PRESETS

# Config
model_config = MODEL_PRESETS["q5_optimal"]  # Q5_K_S 8.26GB
wf_config = WorkflowConfig.square_aesthetic()
wf_config.width = 1024
wf_config.height = 1024

# Builder
builder = FluxWorkflowBuilder(model_config, wf_config)

# Tạo workflow tối ưu
workflow = builder.build_optimized(
    prompt="masterpiece, 1girl, long silver hair, aqua eyes, school uniform, cherry blossoms",
    negative_prompt="extra fingers, blurry",
    seed=42
)

# Validate & save
is_valid, errors = builder.validate(workflow)
builder.save(workflow, "workflows/my_workflow.json")
print(f"Valid: {is_valid}, Nodes: {len(workflow)}")
```

### 4. ComfyUI

**Trong ComfyUI UI:**
1. Mở ComfyUI (Colab link từ notebook)
2. Load → chọn file JSON trong `workflows/`
3. Queue Prompt

**Via API:**
```python
from src.mode_ai.comfy_api import ComfyUIClient
import json

with open("workflows/flux_schnell_gguf_toi_uu.json") as f:
    workflow = json.load(f)

client = ComfyUIClient("127.0.0.1:8188")
result = client.generate_image(workflow, wait=True)
```

### 5. Gradio App

```bash
python app.py
# Mở http://localhost:7860
```

## 🗂️ Workflows

| File | Mô tả | Thời gian T4 |
|---|---|---|
| `flux_schnell_simple.json` | Simple 4 steps, không detailer | ~15s |
| `flux_schnell_gguf_toi_uu.json` ⭐ | **MAIN** - Tối ưu đầy đủ face(SAM)+hand+foot | ~35s |
| `flux_schnell_realistic.json` | Photorealistic 832x1216 | ~25s |
| `flux_schnell_realistic_hires.json` | Realistic + hires 1.5x | ~70s |
| `flux_schnell_inpaint.json` | Inpaint sửa lỗi | ~20s |

Xem chi tiết: [docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md) và [workflows/README.md](workflows/README.md)

## ⚙️ Tham số chuẩn FLUX schnell

- sampler: `euler`
- scheduler: `simple`
- steps: **4** (chuẩn schnell)
- cfg: **1.0** (bắt buộc)
- denoise: 1.0
- size: 1024x1024 native

FaceDetailer:
- Mặt: denoise 0.18, crop 3.0, feather 5, SAM threshold 0.93, tiled
- Tay: denoise 0.25, crop 2.8, feather 24, no SAM
- Chân: denoise 0.30, crop 3.0, feather 24, no SAM

## 📁 Cấu trúc

```
mode-ai/
├── workflows/               # JSON workflows (5 files)
├── src/mode_ai/             # Python package
│   ├── config.py            # ModelConfig, WorkflowConfig
│   ├── workflow_builder.py  # FluxWorkflowBuilder
│   ├── comfy_api.py         # ComfyUIClient
│   └── prompt_enhancer.py   # PromptEnhancer
├── scripts/
│   ├── generate.py          # CLI tạo workflow
│   └── validate.py          # Validate workflows
├── docs/
│   └── WORKFLOW_GUIDE.md    # Hướng dẫn chi tiết
├── .github/workflows/
│   └── image-generation.yml # CI/CD
├── app.py                   # Gradio app
├── ComfyUI_Colab_WAI_fixed.ipynb
└── QUY_TRINH_FLUX.md
```

## 🔄 Quy trình khuyến nghị

1. **Test prompt** với `simple` (15s)
2. **Generate chính** với `toi_uu` (35s, có detailer)
3. **Sửa lỗi** tay/chân bằng `inpaint` hoặc Gradio Cell 6
4. **Upscale** nếu cần in bằng `hires`

## 🛠️ GitHub Actions

Workflow tự động tạo ảnh khi dispatch:

- Vào Actions → Image Generation Workflow → Run workflow
- Nhập prompt, style, size, seed
- Tải artifact workflow JSON

## 📚 Tài liệu

- [QUY_TRINH_FLUX.md](QUY_TRINH_FLUX.md) - Hướng dẫn FLUX gốc
- [docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md) - Chi tiết workflow
- [workflows/README.md](workflows/README.md) - Danh sách workflows

## 📝 License

MIT

## 🤝 Contributing

PR welcome! Đặc biệt:
- Thêm style presets
- Tối ưu thêm workflow
- Hỗ trợ FLUX dev

---

**Made with ❤️ for Colab T4 16GB free**
