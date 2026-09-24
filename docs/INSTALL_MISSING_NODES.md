# 🛠️ Fix Lỗi Missing Node Type - Cài Đặt Custom Nodes

Bạn gặp lỗi:
```
2 nodes affected
Missing node type
Missing Node Packs
```

Đây là do ComfyUI chưa cài custom nodes cần thiết cho FLUX GGUF workflow.

## 🔍 Phân tích lỗi

### Workflow `flux_schnell_simple.json` cần 2 custom nodes:
| Node | Pack | Repo |
|---|---|---|
| `UnetLoaderGGUF` | ComfyUI-GGUF | https://github.com/city96/ComfyUI-GGUF |
| `DualCLIPLoaderGGUF` | ComfyUI-GGUF | https://github.com/city96/ComfyUI-GGUF |

### Workflow `flux_schnell_gguf_toi_uu.json` cần 5+ custom nodes:
| Node | Pack | Repo |
|---|---|---|
| `UnetLoaderGGUF` | ComfyUI-GGUF | city96/ComfyUI-GGUF |
| `DualCLIPLoaderGGUF` | ComfyUI-GGUF | city96/ComfyUI-GGUF |
| `UltralyticsDetectorProvider` | Impact Subpack | ltdrdata/ComfyUI-Impact-Subpack |
| `SAMLoader` | Impact Pack | ltdrdata/ComfyUI-Impact-Pack |
| `FaceDetailer` | Impact Pack | ltdrdata/ComfyUI-Impact-Pack |
| `UltimateSDUpscale` (hires) | Ultimate SD Upscale | ssitu/ComfyUI_UltimateSDUpscale |

## ✅ Cách Fix 1: Cài qua ComfyUI-Manager (Khuyến nghị)

### Bước 1: Cài ComfyUI-Manager
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
# Restart ComfyUI với flag --enable-manager
python main.py --enable-manager
```

Hoặc pip:
```bash
pip install -U --pre comfyui-manager
```

### Bước 2: Trong ComfyUI UI
1. Mở ComfyUI
2. Click **Manager** button (bên phải)
3. **Install Custom Nodes** → Search:
   - `ComfyUI-GGUF` → Install (cho UnetLoaderGGUF, DualCLIPLoaderGGUF)
   - `ComfyUI-Impact-Pack` → Install (cho FaceDetailer, SAMLoader)
   - `ComfyUI-Impact-Subpack` → Install (cho UltralyticsDetectorProvider)
   - `ComfyUI_UltimateSDUpscale` → Install (cho hires workflow)
4. **Restart** ComfyUI
5. Load lại workflow JSON → Queue Prompt

## ✅ Cách Fix 2: Cài thủ công (Manual)

```bash
cd /path/to/ComfyUI/custom_nodes

# 1. ComfyUI-GGUF (bắt buộc cho GGUF workflows)
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ComfyUI-GGUF
pip install -r requirements.txt
cd ..

# 2. Impact Pack (cho FaceDetailer)
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
cd ComfyUI-Impact-Pack
pip install -r requirements.txt
python install.py  # nếu có
cd ..

# 3. Impact Subpack (cho YOLO detectors)
git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git
cd ComfyUI-Impact-Subpack
pip install -r requirements.txt
cd ..

# 4. Ultimate SD Upscale (cho hires, optional)
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git

# Restart ComfyUI
cd ../..
python main.py
```

## ✅ Cách Fix 3: Dùng workflow BUILTIN không cần custom nodes (Nhanh nhất)

Nếu bạn không muốn cài custom nodes, dùng workflows **builtin** đã tạo sẵn:

| File | Custom nodes cần | Mô tả |
|---|---|---|
| `flux_builtin_fp8_simple.json` | **0** - chỉ built-in | FLUX schnell FP8, 4 steps, chạy ngay |
| `flux_builtin_fp8_toi_uu.json` | 3 (Impact Pack) | FP8 + FaceDetailer, không cần GGUF |
| `flux_schnell_simple.json` | 2 (GGUF) | GGUF Q5_K_S, cần cài ComfyUI-GGUF |
| `flux_schnell_gguf_toi_uu.json` | 5 (GGUF + Impact) | Tối ưu đầy đủ |

**Khuyến nghị thử trước:**
```bash
# Trong ComfyUI, Load:
workflows/flux_builtin_fp8_simple.json
# File này chỉ dùng UNETLoader, DualCLIPLoader, VAELoader (built-in)
# Model cần: flux1-schnell-fp8.safetensors, clip_l.safetensors, t5xxl_fp8.safetensors, ae.safetensors
```

## 📦 Models cần cho từng workflow

### GGUF workflows (cần ComfyUI-GGUF):
```
ComfyUI/models/unet/flux1-schnell-Q5_K_S.gguf (8.26GB)
ComfyUI/models/clip/t5-v1_1-xxl-encoder-Q4_K_M.gguf (2.9GB)
ComfyUI/models/clip/clip_l.safetensors (250MB)
ComfyUI/models/vae/ae.safetensors (320MB)
```

### FP8 builtin workflows (không cần GGUF, chỉ built-in):
```
ComfyUI/models/unet/flux1-schnell-fp8.safetensors (hoặc flux1-schnell.safetensors)
ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors
ComfyUI/models/clip/clip_l.safetensors
ComfyUI/models/vae/ae.safetensors
```

Tải FP8 models:
- https://huggingface.co/Comfy-Org/flux1-schnell (flux1-schnell-fp8.safetensors)
- https://huggingface.co/comfyanonymous/flux_text_encoders (t5xxl_fp8.safetensors, clip_l.safetensors)
- https://huggingface.co/black-forest-labs/FLUX.1-schnell (ae.safetensors)

## 🔧 Script tự động cài

Chạy script tự động:

```bash
# Linux/Mac/Colab
bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI

# Hoặc Python
python scripts/install_comfyui_nodes.py --comfyui-path /path/to/ComfyUI
```

Script sẽ tự động clone 4 custom nodes cần thiết.

## ❓ Vẫn lỗi?

1. **Kiểm tra ComfyUI version**: Cần ComfyUI >= 2024-08 cho DualCLIPLoader built-in
   ```bash
   cd ComfyUI
   git pull
   ```

2. **Kiểm tra Python env**: Đảm bảo pip install trong đúng venv của ComfyUI

3. **Xem log**: Khi khởi động ComfyUI, log sẽ báo node nào load fail
   ```bash
   python main.py --verbose
   ```

4. **Dùng workflow builtin**: Nếu vẫn lỗi, dùng `flux_builtin_fp8_simple.json` để test ComfyUI hoạt động, sau đó mới cài thêm custom nodes.

## 📚 Tham khảo

- ComfyUI-GGUF: https://github.com/city96/ComfyUI-GGUF
- Impact Pack: https://github.com/ltdrdata/ComfyUI-Impact-Pack
- Impact Subpack: https://github.com/ltdrdata/ComfyUI-Impact-Subpack
- Manager: https://github.com/ltdrdata/ComfyUI-Manager
- UltimateSDUpscale: https://github.com/ssitu/ComfyUI_UltimateSDUpscale
