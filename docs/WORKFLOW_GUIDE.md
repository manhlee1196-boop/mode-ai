# 🎨 Hướng Dẫn Workflow Tạo Ảnh - FLUX.1-schnell GGUF

Tài liệu chi tiết về hệ thống workflow tạo ảnh cho mode-ai.

## 📦 Tổng Quan Model

| File | Size | Đường dẫn ComfyUI |
|---|---|---|
| `flux1-schnell-Q5_K_S.gguf` (UNET Q5_K_S) | 8.26 GB | `models/unet/` |
| `t5-v1_1-xxl-encoder-Q4_K_M.gguf` (T5-XXL) | 2.9 GB | `models/clip/` |
| `clip_l.safetensors` | 250 MB | `models/clip/` |
| `ae.safetensors` (FLUX VAE) | 320 MB | `models/vae/` |
| `face_yolov8m.pt` (YOLO mặt) | 52 MB | `models/ultralytics/bbox/` |
| `hand_yolov8s.pt` (YOLO tay) | 22 MB | `models/ultralytics/bbox/` |
| `foot_anime_yolo11m_v3.pt` (YOLO chân) | ~40 MB | `models/ultralytics/bbox/` |
| `sam_vit_b_01ec64.pth` (SAM mặt) | 375 MB | `models/sams/` |
| **TỔNG** | **~12.3 GB** | **<15GB ✅** |

## 🗂️ Danh Sách Workflows

### 1. `flux_schnell_simple.json` - Đơn giản nhất
- **Mục đích**: Test nhanh, 4 steps, không detailer
- **Nodes**: 9 nodes (UnetLoaderGGUF, DualCLIPLoaderGGUF, VAELoader, EmptyLatent, 2x CLIPTextEncode, KSampler, VAEDecode, SaveImage)
- **Thời gian**: ~15-20s trên T4
- **Khi dùng**: Test prompt, preview nhanh

### 2. `flux_schnell_gguf_toi_uu.json` - Tối ưu đầy đủ (MAIN) ⭐
- **Mục đích**: Workflow chính, chất lượng cao nhất
- **Nodes**: 18 nodes + 3 FaceDetailer (mặt có SAM, tay/chân không SAM)
- **Cấu trúc**:
  ```
  UnetLoaderGGUF (Q5_K_S) → MODEL
  DualCLIPLoaderGGUF (clip_l + t5 Q4_K_M) → CLIP
  VAELoader (ae.safetensors) → VAE
  EmptyLatent 1024x1024
  CLIPTextEncode positive/negative
  KSampler euler/simple/4 steps/cfg 1.0 → latent
  VAEDecode → base image
  ├── YOLO face + SAM → FaceDetailer (denoise 0.18, crop 3.0, tiled)
  ├── YOLO hand → FaceDetailer (denoise 0.25, crop 2.8, feather 24)
  └── YOLO foot → FaceDetailer (denoise 0.30, crop 3.0, feather 24)
  SaveImage → FLUX_Q5_schnell_toi_uu.png
  ```
- **Thời gian**: ~30-40s trên T4 16GB lowvram
- **Tham số FaceDetailer**:
  - Mặt: denoise 0.18, steps 6, cfg 1.0, guide_size 512, SAM threshold 0.93, tiled encode/decode
  - Tay: denoise 0.25, steps 6, cfg 1.0, crop 2.8, feather 24, no SAM
  - Chân: denoise 0.30, steps 6, cfg 1.0, crop 3.0, feather 24, no SAM

### 3. `flux_schnell_realistic.json` - Chân thực
- **Mục đích**: Ảnh photorealistic, chân dung người Việt
- **Size**: 832x1216 portrait
- **Prompt**: photorealistic, natural skin texture, pores visible
- **Detailer**: 1x FaceDetailer mặt với denoise 0.22 cao hơn
- **Khi dùng**: Ảnh chân dung, áo dài, đời thường

### 4. `flux_schnell_realistic_hires.json` - Hires fix
- **Mục đích**: Realistic + upscale 1.5x với UltimateSDUpscale
- **Thêm**: UpscaleModelLoader (4x-UltraSharp.pth) + UltimateSDUpscale
- **Thời gian**: ~60-80s
- **Khi dùng**: Cần ảnh độ phân giải cao, in ấn

### 5. `flux_schnell_inpaint.json` - Sửa lỗi tay/chân/mặt
- **Mục đích**: Inpaint vùng lỗi sau khi generate
- **Cách dùng**:
  1. Load ảnh gốc + mask (vùng trắng cần sửa)
  2. VAEEncodeForInpaint với grow_mask 12px
  3. KSampler denoise 0.5, steps 8
- **Dùng trong**: Cell 6 Gradio của notebook

## ⚙️ Tham Số Khuyến Nghị

### KSampler chính (FLUX schnell)
| Tham số | Giá trị | Ghi chú |
|---|---|---|
| sampler | `euler` | Schnell distill |
| scheduler | `simple` | Tốt nhất cho schnell |
| steps | **4** | Chuẩn schnell |
| CFG | **1.0** | Bắt buộc cho FLUX schnell |
| denoise | 1.0 | Generate gốc |

### FaceDetailer
| Loại | denoise | steps | cfg | crop | feather | SAM |
|---|---|---|---|---|---|---|
| Mặt | 0.18 | 6 | 1.0 | 3.0 | 5 | Có, threshold 0.93 |
| Tay | 0.25 | 6 | 1.0 | 2.8 | 24 | Không |
| Chân | 0.30 | 6 | 1.0 | 3.0 | 24 | Không |

### Inpaint
| Tham số | Giá trị |
|---|---|
| denoise | 0.5 |
| steps | 8 |
| cfg | 1.0 |
| grow_mask | 12px |

## 🚀 Cách Sử Dụng

### Cách 1: Python Builder (Khuyến nghị)
```python
from src.mode_ai import FluxWorkflowBuilder, WorkflowConfig
from src.mode_ai.config import MODEL_PRESETS

# Config
model_config = MODEL_PRESETS["q5_optimal"]
wf_config = WorkflowConfig.square_aesthetic()
wf_config.width = 1024
wf_config.height = 1024

# Builder
builder = FluxWorkflowBuilder(model_config, wf_config)
workflow = builder.build_optimized(
    prompt="1girl, cherry blossoms, masterpiece",
    negative_prompt="extra fingers, blurry",
    seed=42
)

# Save
builder.save(workflow, "workflows/my_workflow.json")

# Validate
is_valid, errors = builder.validate(workflow)
print(is_valid, errors)
```

### Cách 2: CLI Script
```bash
# Tạo workflow đơn giản
python scripts/generate.py --prompt "1girl, school uniform" --workflow simple --output workflows/test.json

# Tạo workflow tối ưu với style realistic
python scripts/generate.py --prompt "Vietnamese woman in ao dai" --style realistic_vietnamese --workflow optimized --width 832 --height 1216 --seed 123

# Dùng preset
python scripts/generate.py --preset vietnamese_aodai --style photorealistic --workflow realistic

# Chỉ tạo JSON, không queue
python scripts/generate.py --prompt "beautiful landscape" --no-queue --output workflows/landscape.json

# List styles
python scripts/generate.py --list-styles
python scripts/generate.py --list-presets
```

### Cách 3: ComfyUI UI
1. Mở ComfyUI (link cloudflared từ Cell 3 notebook)
2. Load → chọn file JSON trong `workflows/`
3. Sửa prompt nếu cần
4. Queue Prompt

### Cách 4: ComfyUI API
```python
from src.mode_ai.comfy_api import ComfyUIClient
import json

# Load workflow
with open("workflows/flux_schnell_gguf_toi_uu.json") as f:
    workflow = json.load(f)

# Client
client = ComfyUIClient("127.0.0.1:8188")
if client.is_server_running():
    result = client.generate_image(workflow, wait=True)
    print(result)
```

## 💡 Mẹo Prompt FLUX

FLUX hiểu tiếng Anh tự nhiên tốt hơn tag soup của SDXL.

### Prompt tốt (tự nhiên):
```
masterpiece, best quality, very aesthetic, 1girl, long silver hair, aqua eyes, school uniform, standing under cherry blossoms, soft afternoon sunlight, detailed face, detailed hands, five fingers, full body
```

### Prompt photorealistic:
```
photorealistic, ultra realistic, 8k, masterpiece, best quality, 1girl, 20 years old, beautiful Vietnamese woman, natural skin texture, pores visible, long black hair, brown eyes, wearing white ao dai, standing in Hanoi old quarter, morning sunlight, bokeh background, detailed face, natural makeup, candid photo, shot on Sony A7R IV, 85mm f/1.4
```

### Negative prompt:
FLUX schnell không cần negative phức tạp, có thể để trống. Nếu muốn an toàn:
```
extra fingers, mutated hands, bad anatomy, blurry, low quality, watermark
```

### Tỷ lệ khung hình:
- 1024x1024 vuông - tốt nhất cho FLUX
- 832x1216 dọc - chân dung
- 1216x832 ngang - phong cảnh
- Giữ bội số của 32, tổng pixel gần 1M là tốt nhất

## 🛠 Cấu Trúc Code

```
mode-ai/
├── workflows/                          # JSON workflows
│   ├── flux_schnell_simple.json
│   ├── flux_schnell_gguf_toi_uu.json   # MAIN ⭐
│   ├── flux_schnell_realistic.json
│   ├── flux_schnell_realistic_hires.json
│   └── flux_schnell_inpaint.json
├── src/mode_ai/                        # Python package
│   ├── __init__.py
│   ├── config.py                       # ModelConfig, WorkflowConfig
│   ├── workflow_builder.py             # FluxWorkflowBuilder
│   ├── comfy_api.py                    # ComfyUIClient
│   └── prompt_enhancer.py              # PromptEnhancer
├── scripts/
│   ├── generate.py                     # CLI tạo workflow
│   └── validate.py                     # Validate workflows
├── docs/
│   └── WORKFLOW_GUIDE.md               # Tài liệu này
├── .github/workflows/
│   └── image-generation.yml            # CI/CD workflow
└── app.py                              # Gradio app (optional)
```

## ⏱ Tốc Độ Trên T4 16GB (lowvram)

| Công đoạn | Thời gian |
|---|---|
| Load model lần đầu | 30-60s |
| Generate 4 bước 1024x1024 | ~15-20s |
| FaceDetailer mặt (SAM) | ~5-8s |
| FaceDetailer tay | ~4-6s |
| FaceDetailer chân | ~4-6s |
| **Tổng / ảnh tối ưu** | **~30-40s** |
| Hires 1.5x | +30s |

## ❌ Xử Lý Lỗi

- **OOM / CUDA out of memory**:
  - Chắc chắn `VRAM=lowvram` trong Cell 3
  - Đổi `VAE_PREC = cpu-vae`
  - `PREVIEW = none`
  - Đóng tab Gradio inpaint nếu không dùng
  - Giảm `max_size` trong FaceDetailer xuống 768

- **Ảnh bị đen / nhiễu**: chưa load xong model, chờ vài giây rồi Queue lại

- **Mắt/tay/chân còn lỗi**: tăng denoise của FaceDetailer tương ứng (mặt 0.22, tay 0.30, chân 0.35)

- **T5 hoặc GGUF không hiện**: kiểm tra symlink trong `/content/ComfyUI/models/clip` và `models/unet`, restart Cell 3

- **HuggingFace bị chặn**: thêm `%env HF_ENDPOINT=https://hf-mirror.com`

## 🔄 Quy Trình Khuyến Nghị

1. **Generate base** với `flux_schnell_simple.json` để test prompt nhanh (15s)
2. **Chọn ảnh ưng** → chạy lại với `flux_schnell_gguf_toi_uu.json` để có detailer đầy đủ (35s)
3. **Nếu còn lỗi tay/chân** → dùng `flux_schnell_inpaint.json` hoặc Gradio Cell 6 để tô và sửa
4. **Nếu cần in** → dùng `flux_schnell_realistic_hires.json` để upscale 1.5x

## 📚 Tham Khảo

- FLUX.1-schnell: https://huggingface.co/black-forest-labs/FLUX.1-schnell
- ComfyUI-GGUF: https://github.com/city96/ComfyUI-GGUF
- Impact Pack: https://github.com/ltdrdata/ComfyUI-Impact-Pack
- QUY_TRINH_FLUX.md trong repo
