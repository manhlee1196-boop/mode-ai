# mode-ai 🎨 - FLUX.1-schnell Workflow Toolkit

Hệ thống workflow tạo ảnh tối ưu cho **FLUX.1-schnell**, chạy được trên **Colab T4 16GB free**.

Có **2 chế độ nodes**:

| Chế độ | Nodes | Ưu / nhược |
|---|---|---|
| **BUILTIN** (khuyến nghị nếu mới bắt đầu) | `UNETLoader`, `DualCLIPLoader` | ✅ **Không cần cài custom nodes** — chạy ngay, fix lỗi *Missing node type*. Model `flux1-schnell-fp8.safetensors`. |
| **GGUF** (nhẹ hơn ~12.3GB) | `UnetLoaderGGUF`, `DualCLIPLoaderGGUF` | ✅ Tiết kiệm VRAM hơn. ❌ Cần cài [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF). |

## 🆘 Gặp lỗi "2 nodes affected - Missing node type"?

Đây là lỗi phổ biến: ComfyUI chưa cài custom node **ComfyUI-GGUF** (cung cấp 2 node `UnetLoaderGGUF` và `DualCLIPLoaderGGUF`).

**Fix nhanh nhất (1 lệnh):**
```bash
bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI --only-gguf
```

**Hoặc không cần cài gì cả** — dùng workflow BUILTIN có sẵn:
```bash
# Mở trực tiếp trong ComfyUI:
workflows/flux_builtin_fp8_simple.json    # 9 nodes, 0 custom nodes

# Hoặc tự tạo:
python scripts/generate.py --prompt "1girl, cherry blossoms" --builtin --workflow simple
```

Chi tiết: **[docs/INSTALL_MISSING_NODES.md](docs/INSTALL_MISSING_NODES.md)**

## ✨ Tính năng

- 🚀 **7 workflows ComfyUI JSON**: 1 BUILTIN simple (0 custom nodes), 1 BUILTIN tối ưu, 5 GGUF
- 🛠️ **Python Builder**: `FluxWorkflowBuilder(use_builtin=True/False)` tạo workflow programmatically
- 🎨 **Prompt Enhancer**: 6 style presets (anime, photorealistic, realistic_vietnamese, cinematic, portrait, full_body) + 5 preset prompts
- 🌐 **ComfyUI API Client**: Queue workflow tới ComfyUI server
- 📊 **Validation**: `scripts/validate.py` kiểm tra workflow + báo cáo custom nodes cần cài
- 🖥️ **Gradio App**: UI tạo workflow trực quan
- ⚙️ **GitHub Actions**: CI/CD validate & build workflow tự động
- 🔧 **Auto-install script**: `scripts/install_comfyui_nodes.sh` cài tất cả custom nodes

## 🚀 Quick Start

### 1. Cài đặt

```bash
git clone https://github.com/manhlee1196-boop/mode-ai
cd mode-ai
pip install -r requirements.txt
```

### 2. Cài custom nodes (chỉ cần nếu dùng workflow GGUF / FaceDetailer)

```bash
# Tự động cài tất cả
bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI

# Chỉ cài ComfyUI-GGUF để fix 2 nodes lỗi
bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI --only-gguf

# Hoặc dùng Python
python scripts/install_comfyui_nodes.py --comfyui-path /path/to/ComfyUI
```

### 3. Tạo workflow bằng CLI

```bash
# BUILTIN - không cần cài gì, chạy ngay
python scripts/generate.py --prompt "1girl, cherry blossoms" --builtin --workflow simple

# BUILTIN tối ưu - cần Impact Pack cho FaceDetailer
python scripts/generate.py --prompt "beautiful girl" --builtin --workflow optimized --width 1024 --height 1024 --seed 42

# GGUF - cần ComfyUI-GGUF, nhẹ hơn (~12.3GB)
python scripts/generate.py --prompt "beautiful girl" --workflow optimized

# Realistic Vietnamese
python scripts/generate.py --preset vietnamese_aodai --style realistic_vietnamese --workflow realistic --width 832 --height 1216 --builtin

# List styles và presets
python scripts/generate.py --list-styles
python scripts/generate.py --list-presets

# Validate tất cả workflows + báo cáo custom nodes cần cài
python scripts/validate.py
```

### 4. Python API

```python
from src.mode_ai import FluxWorkflowBuilder, WorkflowConfig
from src.mode_ai.config import MODEL_PRESETS

# --- Chế độ BUILTIN: không cần cài custom nodes ---
builder = FluxWorkflowBuilder(use_builtin=True)
workflow = builder.build_optimized(
    prompt="masterpiece, 1girl, long silver hair, aqua eyes, school uniform, cherry blossoms",
    negative_prompt="extra fingers, blurry",
    seed=42
)
is_valid, errors = builder.validate(workflow)
stats = builder.get_stats(workflow)     # builtin_only / custom_nodes_required
print(builder.get_install_guide(workflow))  # hướng dẫn cài nodes nếu thiếu
builder.save(workflow, "workflows/my_builtin.json")

# --- Chế độ GGUF: nhẹ hơn, cần ComfyUI-GGUF ---
builder = FluxWorkflowBuilder(
    model_config=MODEL_PRESETS["q5_optimal"],   # Q5_K_S 8.26GB, tổng ~12.3GB
    workflow_config=WorkflowConfig.square_aesthetic(),
    use_builtin=False,
)
workflow = builder.build_optimized("1girl, cherry blossoms", seed=42)
builder.save(workflow, "workflows/my_gguf.json")
```

### 5. ComfyUI

**Trong ComfyUI UI:**
1. Mở ComfyUI
2. Drag & drop file JSON trong `workflows/` vào canvas
3. Queue Prompt

**Via API:**
```python
from src.mode_ai.comfy_api import ComfyUIClient
import json

with open("workflows/flux_builtin_fp8_simple.json") as f:
    workflow = json.load(f)

client = ComfyUIClient("127.0.0.1:8188")
result = client.generate_image(workflow, wait=True)
```

### 6. Gradio App

```bash
python app.py
# Mở http://localhost:7860
```

## 🗂️ Workflows

| File | Nodes | Custom nodes cần | Mô tả | T4 |
|---|---|---|---|---|
| `flux_builtin_fp8_simple.json` | 9 | **0** | BUILTIN, chạy ngay không cài gì | ~15s |
| `flux_builtin_fp8_toi_uu.json` | 17 | Impact x3 | BUILTIN + FaceDetailer face+hand+foot | ~35s |
| `flux_schnell_simple.json` | 9 | GGUF x2 | Simple 4 steps | ~15s |
| `flux_schnell_gguf_toi_uu.json` ⭐ | 17 | GGUF x2 + Impact x3 | **MAIN** face(SAM)+hand+foot | ~35s |
| `flux_schnell_realistic.json` | 12 | GGUF x2 + Impact x3 | Photorealistic 832x1216 | ~25s |
| `flux_schnell_realistic_hires.json` | 15 | GGUF x2 + Impact x4 | + hires 1.5x upscale | ~70s |
| `flux_schnell_inpaint.json` | 12 | GGUF x2 | Inpaint sửa lỗi | ~20s |

**Custom nodes cần cài:**
| Node | Pack |
|---|---|
| `UnetLoaderGGUF`, `DualCLIPLoaderGGUF` | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) |
| `FaceDetailer`, `SAMLoader` | [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) |
| `UltralyticsDetectorProvider` | [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) |
| `UltimateSDUpscale` | [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) |

Xem chi tiết: [docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md), [docs/INSTALL_MISSING_NODES.md](docs/INSTALL_MISSING_NODES.md), [workflows/README.md](workflows/README.md)

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
├── workflows/                    # 7 JSON workflows + README
├── src/mode_ai/                  # Python package
│   ├── config.py                 # ModelConfig, WorkflowConfig, presets Q4/Q5/Q6/Q8
│   ├── workflow_builder.py       # FluxWorkflowBuilder (use_builtin=True/False)
│   ├── comfy_api.py              # ComfyUIClient + MockClient
│   └── prompt_enhancer.py        # PromptEnhancer (6 styles, 5 presets)
├── scripts/
│   ├── generate.py               # CLI tạo workflow (--builtin, --workflow)
│   ├── validate.py               # Validate + báo cáo custom nodes
│   ├── install_comfyui_nodes.sh  # Auto cài custom nodes
│   └── install_comfyui_nodes.py  # Auto cài custom nodes (Python)
├── docs/
│   ├── WORKFLOW_GUIDE.md         # Hướng dẫn chi tiết tham số
│   ├── INSTALL_MISSING_NODES.md  # Fix lỗi Missing node type
│   └── CHECK_REPORT.md           # Báo cáo kiểm tra
├── .github/workflows/
│   └── image-generation.yml      # CI/CD
├── app.py                        # Gradio app
├── ComfyUI_Colab_WAI_fixed.ipynb
└── QUY_TRINH_FLUX.md
```

## 🔄 Quy trình khuyến nghị

1. **Test prompt** với `flux_builtin_fp8_simple.json` (15s, không cần cài gì)
2. **Generate chính** với `flux_schnell_gguf_toi_uu.json` (35s, có detailer)
3. **Sửa lỗi** tay/chân bằng `flux_schnell_inpaint.json` hoặc Gradio Cell 6
4. **Upscale** nếu cần in bằng `flux_schnell_realistic_hires.json`

## 🛠️ GitHub Actions

Workflow tự động validate & build khi push/dispatch:

- Vào Actions → Image Generation Workflow → Run workflow
- Nhập prompt, style, size, seed
- Tải artifact workflow JSON

## 📚 Tài liệu

- [QUY_TRINH_FLUX.md](QUY_TRINH_FLUX.md) - Hướng dẫn FLUX gốc
- [docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md) - Chi tiết workflow
- [docs/INSTALL_MISSING_NODES.md](docs/INSTALL_MISSING_NODES.md) - Fix lỗi Missing node type
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
