# Workflows - FLUX.1-schnell GGUF

Thư mục chứa các ComfyUI workflow JSON đã tối ưu cho FLUX.1-schnell GGUF Q5_K_S.

## Danh sách

| File | Mô tả | Nodes | Thời gian T4 |
|---|---|---|---|
| `flux_schnell_simple.json` | Simple 4 steps, không detailer, test nhanh | 9 | ~15s |
| `flux_schnell_gguf_toi_uu.json` | **MAIN** - Tối ưu đầy đủ face+hand+foot với SAM | 18 | ~35s |
| `flux_schnell_realistic.json` | Photorealistic portrait 832x1216 | 11 | ~25s |
| `flux_schnell_realistic_hires.json` | Realistic + hires 1.5x upscale | 13 | ~70s |
| `flux_schnell_inpaint.json` | Inpaint sửa lỗi vùng tô | 12 | ~20s |

## Sử dụng nhanh

### ComfyUI UI
1. Mở ComfyUI
2. Drag & drop file JSON vào canvas
3. Sửa prompt
4. Queue Prompt

### Python
```python
from src.mode_ai import FluxWorkflowBuilder
builder = FluxWorkflowBuilder()
wf = builder.build_optimized("1girl, cherry blossoms", seed=42)
builder.save(wf, "workflows/my.json")
```

### CLI
```bash
python scripts/generate.py --prompt "beautiful girl" --workflow optimized --output workflows/custom.json
python scripts/validate.py  # validate tất cả
```

## Thông số chuẩn FLUX schnell

- sampler: euler
- scheduler: simple
- steps: 4
- cfg: 1.0
- denoise: 1.0
- size: 1024x1024 (hoặc 832x1216 portrait, 1216x832 landscape)

## FaceDetailer

- Mặt: denoise 0.18, crop 3.0, feather 5, có SAM threshold 0.93, tiled
- Tay: denoise 0.25, crop 2.8, feather 24, không SAM
- Chân: denoise 0.30, crop 3.0, feather 24, không SAM
