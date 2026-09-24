# Workflows - FLUX.1-schnell GGUF

Thư mục chứa các ComfyUI workflow JSON tối ưu cho FLUX.1-schnell.

> 🆘 **Gặp lỗi "2 nodes affected - Missing node type / Missing Node Packs"?**
> Nguyên nhân: chưa cài **ComfyUI-GGUF** (cung cấp 2 node `UnetLoaderGGUF` và `DualCLIPLoaderGGUF`).
> Fix: xem [docs/INSTALL_MISSING_NODES.md](../docs/INSTALL_MISSING_NODES.md) hoặc chạy
> `bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI --only-gguf`
> Hoặc dùng ngay `flux_builtin_fp8_simple.json` — **0 custom nodes**.

## Danh sách

| File | Nodes | Custom nodes cần | Mô tả | Thời gian T4 |
|---|---|---|---|---|
| `flux_builtin_fp8_simple.json` | 9 | **0 (chạy ngay)** | FLUX schnell FP8 built-in, test nhanh | ~15s |
| `flux_builtin_fp8_toi_uu.json` | 17 | Impact Pack x3 | FP8 + FaceDetailer face+hand+foot | ~35s |
| `flux_schnell_simple.json` | 9 | GGUF x2 | Simple 4 steps, test nhanh | ~15s |
| `flux_schnell_gguf_toi_uu.json` ⭐ | 17 | GGUF x2 + Impact x3 | **MAIN** - Tối ưu đầy đủ face(SAM)+hand+foot | ~35s |
| `flux_schnell_realistic.json` | 12 | GGUF x2 + Impact x3 | Photorealistic portrait 832x1216 | ~25s |
| `flux_schnell_realistic_hires.json` | 15 | GGUF x2 + Impact x4 | Realistic + hires 1.5x upscale | ~70s |
| `flux_schnell_inpaint.json` | 12 | GGUF x2 | Inpaint sửa lỗi vùng tô | ~20s |

## Custom nodes theo từng loại

| Custom node | Pack cần cài | Dùng trong |
|---|---|---|
| `UnetLoaderGGUF`, `DualCLIPLoaderGGUF` | **ComfyUI-GGUF** | 5 workflow GGUF |
| `FaceDetailer`, `SAMLoader` | **ComfyUI-Impact-Pack** | optimized / realistic / hires |
| `UltralyticsDetectorProvider` | **ComfyUI-Impact-Subpack** | optimized / realistic / hires |
| `UltimateSDUpscale` | **ComfyUI_UltimateSDUpscale** | hires |

Cài nhanh tất cả:
```bash
bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI
# hoặc
python scripts/install_comfyui_nodes.py --comfyui-path /path/to/ComfyUI
```

## Models cần

### Cho workflow BUILTIN (khỏi cài custom nodes)
```
ComfyUI/models/unet/flux1-schnell-fp8.safetensors
ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors
ComfyUI/models/clip/clip_l.safetensors
ComfyUI/models/vae/ae.safetensors
```

### Cho workflow GGUF
```
ComfyUI/models/unet/flux1-schnell-Q5_K_S.gguf (8.26GB)
ComfyUI/models/clip/t5-v1_1-xxl-encoder-Q4_K_M.gguf (2.9GB)
ComfyUI/models/clip/clip_l.safetensors
ComfyUI/models/vae/ae.safetensors
```

### Cho FaceDetailer (Impact Pack)
```
ComfyUI/models/ultralytics/bbox/face_yolov8m.pt
ComfyUI/models/ultralytics/bbox/hand_yolov8s.pt
ComfyUI/models/ultralytics/bbox/foot_anime_yolo11m_v3.pt
ComfyUI/models/sams/sam_vit_b_01ec64.pth
```

## Sử dụng nhanh

### ComfyUI UI
1. Mở ComfyUI
2. Drag & drop file JSON vào canvas
3. Sửa prompt
4. Queue Prompt

### Python
```python
from src.mode_ai import FluxWorkflowBuilder

# BUILTIN - không cần custom nodes (fix lỗi Missing node type)
builder = FluxWorkflowBuilder(use_builtin=True)
wf = builder.build_simple("1girl, cherry blossoms", seed=42)
builder.save(wf, "workflows/my_builtin.json")

# GGUF - cần cài ComfyUI-GGUF
builder = FluxWorkflowBuilder(use_builtin=False)
wf = builder.build_optimized("1girl, cherry blossoms", seed=42)
builder.save(wf, "workflows/my_gguf.json")
```

### CLI
```bash
# BUILTIN - chạy ngay không cần cài gì
python scripts/generate.py --prompt "1girl, cherry blossoms" --builtin --workflow simple

# GGUF - cần ComfyUI-GGUF
python scripts/generate.py --prompt "beautiful girl" --workflow optimized

# Validate tất cả workflows + báo cáo custom nodes cần cài
python scripts/validate.py
```

## Thông số chuẩn FLUX schnell

- sampler: `euler`
- scheduler: `simple`
- steps: `4`
- cfg: `1.0`
- denoise: `1.0`
- size: 1024x1024 (hoặc 832x1216 portrait, 1216x832 landscape)

## FaceDetailer

- Mặt: denoise 0.18, crop 3.0, feather 5, có SAM threshold 0.93, tiled
- Tay: denoise 0.25, crop 2.8, feather 24, không SAM
- Chân: denoise 0.30, crop 3.0, feather 24, không SAM
