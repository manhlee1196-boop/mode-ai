# 🎨 Hướng Dẫn Sử Dụng FLUX.1-schnell GGUF (phiên bản tốt nhất dưới 15GB)

## 📦 Tổng dung lượng

| File | Size | Nơi chứa |
|---|---|---|
| `flux1-schnell-Q5_K_S.gguf` (UNET Q5_K_S) | **8.26 GB** | `AI_Models/gguf/` → `models/unet/` |
| `t5-v1_1-xxl-encoder-Q4_K_M.gguf` (T5-XXL) | **2.9 GB** | `AI_Models/clip/` → `models/clip/` |
| `clip_l.safetensors` | 250 MB | `AI_Models/clip/` → `models/clip/` |
| `ae.safetensors` (FLUX VAE) | 320 MB | `AI_Models/vae/` → `models/vae/` |
| `face_yolov8m.pt` (YOLO mặt) | 52 MB | `ultralytics/bbox/` |
| `hand_yolov8s.pt` (YOLO tay) | 22 MB | `ultralytics/bbox/` |
| `foot_anime_yolo11m_v3.pt` (YOLO chân anime) | ~40 MB | `ultralytics/bbox/` |
| `sam_vit_b_01ec64.pth` (SAM mặt) | 375 MB | `sams/` |
| **TỔNG MODEL** | **~12.3 GB** | **Dưới 15GB ✅** |

Chọn Q5_K_S cho UNET (cao nhất có thể mà tổng dưới 15GB) + Q4_K_M cho T5-XXL (text encoder ít nhạy hơn, Q4 đủ dùng). Kết quả: chất lượng gần bằng FP8 nhưng dung lượng chỉ ~12GB, chạy mượt trên **T4 16GB Colab free**.

## 🚀 Cách chạy trên Colab

1. **Cell 1**: Mount Drive, cài ComfyUI.
2. **Cell 1B**: Cài Impact Pack, Impact Subpack, **ComfyUI-GGUF** (node load GGUF), YOLO/SAM. Symlink `models/vae`, `models/clip`, `models/unet` (gguf).
3. **Cell 2**: Tải FLUX Q5_K_S + T5 Q4_K_M + VAE + CLIP-L + YOLO + SAM. (Tick `BO_QUA_MODEL=True` nếu đã tải từ lần chạy trước).
4. **Cell 3**: Khởi chạy ComfyUI.
   - **VRAM = `lowvram`** (bắt buộc trên T4 16GB — đã đặt sẵn)
   - VAE_PREC = `fp16-vae` (nếu OOM đổi sang `cpu-vae`)
   - ATTENTION = `pytorch`, DISABLE_XFORMERS = True
5. **Cell 3C**: Tải `workflow_flux_schnell_gguf_toi_uu.json` + copy vào `ComfyUI/input/`.
6. Mở ComfyUI bằng link cloudflared hiện ra → **Load → workflow_flux_schnell_gguf_toi_uu.json** → **Queue Prompt**.

## ⚙️ Tham số khuyến nghị

### Generate ảnh gốc (KSampler chính)
| Tham số | Giá trị | Ghi chú |
|---|---|---|
| sampler | `euler` | Schnell distill |
| scheduler | `simple` | Tốt nhất cho schnell |
| steps | **4** | Đúng chuẩn schnell |
| CFG | **1.0** | FLUX schnell yêu cầu CFG=1 |
| denoise | 1.0 | Generate gốc |
| size | 1024×1024 | Độ phân giải native của FLUX |
| seed | random | — |

### FaceDetailer (mặt)
- denoise 0.18, steps 6, cfg 1.0, guide_size 512, crop 3.0
- **Có SAM** (sam_vit_b) cho mask mặt siêu nét, `sam_threshold=0.93`, `sam_dilation=0`
- `tiled_encode=True, tiled_decode=True` (tránh OOM)

### FaceDetailer (tay)
- denoise 0.25, steps 6, cfg 1.0, crop 2.8
- **Không nối SAM** (feather=24 để blend mượt), tiết kiệm VRAM
- `force_inpaint=True, noise_mask=True`

### FaceDetailer (chân)
- denoise 0.30, steps 6, cfg 1.0, crop 3.0
- **Không nối SAM** (feather=24)

### Inpaint (Cell 6 Gradio)
- Upload ảnh + tô đen vùng lỗi
- denoise **0.5**, steps **8**, cfg **1.0**, grow_mask **12px**
- Tự động dùng `UnetLoaderGGUF + DualCLIPLoaderGGUF + VAELoader + VAEEncodeForInpaint`

## 🛠 Các node chính trong workflow

```
UnetLoaderGGUF → flux1-schnell-Q5_K_S.gguf              → MODEL
DualCLIPLoaderGGUF → clip_l.safetensors + t5-xxl-Q4_K_M.gguf (type=flux) → CLIP
VAELoader → ae.safetensors                               → VAE
EmptyLatentImage 1024×1024                               → LATENT
CLIPTextEncode (positive/negative) → conditioning
KSampler (euler/simple/4/1.0) → latent
VAEDecode → image
FaceDetailer (mặt, có SAM)
FaceDetailer (tay, không SAM, feather=24)
FaceDetailer (chân, không SAM, feather=24)
SaveImage → FLUX_Q5_schnell_*.png
```

## ⏱ Tốc độ ước tính trên T4 16GB (lowvram)

| Công đoạn | Thời gian |
|---|---|
| Load model lần đầu | 30-60s |
| Generate 4 bước 1024×1024 | ~15-20s |
| FaceDetailer mặt (SAM) | ~5-8s |
| FaceDetailer tay | ~4-6s |
| FaceDetailer chân | ~4-6s |
| **Tổng / ảnh** | **~30-40s** |

## ❌ Xử lý lỗi

- **OOM / CUDA out of memory**:
  - Chắc chắn `VRAM=lowvram`
  - Đổi `VAE_PREC = cpu-vae`
  - `PREVIEW = none`
  - Đóng tab Gradio inpaint (Cell 6) nếu không dùng — nó cũng chiếm VRAM
- **Ảnh bị đen / nhiễu**: chưa load xong model, chờ vài giây rồi Queue lại
- **Mắt/tay/chân còn lỗi**: tăng denoise của FaceDetailer tương ứng (mặt 0.22, tay 0.30, chân 0.35)
- **T5 hoặc GGUF không hiện trong list**: kiểm tra symlink trong `/content/ComfyUI/models/clip` và `models/unet`, restart Cell 3
- **HuggingFace bị chặn**: cell sẽ tự thử qua `hf-mirror.com`; nếu vẫn lỗi, vào Colab thêm: `%env HF_ENDPOINT=https://hf-mirror.com` rồi chạy lại Cell 2.

## 💡 Mẹo prompt FLUX

FLUX hiểu tiếng Anh rất tốt, prompt tự nhiên như mô tả sẽ cho kết quả tốt hơn "tag soup" của SDXL.

- **Prompt mẫu**: `masterpiece, best quality, very aesthetic, 1girl, long silver hair, aqua eyes, school uniform, standing under cherry blossoms, soft afternoon sunlight, detailed face, detailed hands, five fingers, full body`
- **Negative prompt**: FLUX không cần negative phức tạp, có thể để trống. Nếu muốn an toàn thêm: `extra fingers, mutated hands, bad anatomy, blurry, low quality, watermark`.
- Tỷ lệ: 1024×1024 (vuông), 832×1216 (dọc), 1216×832 (ngang) — giữ bội số của 32, tổng pixel gần 1M là tốt nhất.
