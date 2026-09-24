# 📁 Workflows FLUX.1-schnell (GGUF Q5_K_S)

Tất cả file trong thư mục này được **sinh ra**, không viết tay. Nguồn duy nhất là
[`scripts/build_workflows.py`](../scripts/build_workflows.py) — cũng chính là đoạn code được
nhúng vào Cell 4 của notebook.

```bash
python3 scripts/build_workflows.py     # sinh lại toàn bộ
```

## Hai định dạng

| Thư mục | Định dạng | Dùng cho |
|---|---|---|
| `workflows/*.json` | **API format** | `POST /prompt` (Cell 6 của notebook, script, tự động hoá) |
| `workflows/ui/*.json` | **UI format** | kéo/thả hoặc `Workflow → Open` trong giao diện ComfyUI (có toạ độ node) |

Chuyển đổi: `python3 scripts/api_to_ui.py workflows/flux_q5_standard.json`

UI format được sinh từ API format bằng `api_to_ui.py`, rồi **đọc ngược lại** để kiểm tra
(`validate_workflows.py` làm việc này tự động) — đảm bảo hai bản luôn mô tả cùng một graph.

## Chọn pipeline nào

| File | Node | Làm gì | T4 16 GB |
|---|---|---|---|
| `flux_q5_fast.json` | 9 | 4 bước `euler/simple`, `cfg 1.0`, không detailer | ~15-20 s |
| `flux_q5_standard.json` | 13 | + FaceDetailer mặt (YOLO + SAM) | ~25-35 s |
| `flux_q5_quality.json` | 15 | + FaceDetailer tay | ~40-55 s |
| `flux_q5_hires.json` | 17 | 832×1216 → sửa mặt → ×1.5 → 4 bước denoise 0.35 | ~70-90 s |
| `flux_q5_inpaint.json` | 12 | sửa vùng tô trên ảnh có sẵn | ~15-25 s |

Thời gian đo trên Colab free T4, `PREVIEW=taesd`, model trên Drive.

## Sơ đồ node

### `flux_q5_standard.json`

```
1  UnetLoaderGGUF        flux1-schnell-Q5_K_S.gguf          → MODEL
2  DualCLIPLoaderGGUF    clip_l + t5-xxl-Q4_K_M  (flux)     → CLIP
3  VAELoader             ae.safetensors                     → VAE
4  EmptyLatentImage      1024×1024                          → LATENT
5  CLIPTextEncode        prompt                             → +COND
6  CLIPTextEncode        ""  (cfg=1.0 nên không dùng)       → -COND
7  KSampler              euler/simple, 4 bước, cfg 1.0      → LATENT
8  VAEDecode                                                → IMAGE
10 UltralyticsDetectorProvider  bbox/face_yolov8m.pt        → BBOX_DETECTOR
13 SAMLoader             sam_vit_b_01ec64.pth (AUTO)        → SAM_MODEL
20 FaceDetailer          denoise 0.22, crop 3.0, có SAM     → IMAGE
30 SaveImage             flux/base      (ảnh chưa sửa)
9  SaveImage             flux/standard  (ảnh đã sửa)
```

`flux_q5_quality` nối thêm `11 UltralyticsDetectorProvider (bbox/hand_yolov8s.pt)` và
`21 FaceDetailer (denoise 0.28, feather 16, crop 2.5, không SAM)`.

`flux_q5_hires` nối thêm `40 ImageScaleBy (lanczos ×1.5) → 41 VAEEncode → 42 KSampler
(denoise 0.35, 4 bước) → 43 VAEDecode`. Chỉ dùng node có sẵn trong ComfyUI — không cần
`UpscaleModelLoader` / ESRGAN / `UltimateSDUpscale`.

## Các file khác

| File | Là gì |
|---|---|
| `node_spec.json` | `INPUT_TYPES` / `RETURN_TYPES` trích tự động từ mã nguồn ComfyUI + GGUF + Impact Pack/Subpack (891 node). Sinh bằng `scripts/node_spec.py`. Cần thiết để `validate_workflows.py` chạy mà không phải clone lại. |

## Kiểm tra

```bash
python3 scripts/validate_workflows.py
```

Trình kiểm tra đối chiếu từng node với spec thật:

* `class_type` có tồn tại
* thiếu/thừa input so với `required` + `optional`
* kiểu dữ liệu của từng đường nối (MODEL/CLIP/VAE/LATENT/…)
* enum hợp lệ (`sampler_name`, `scheduler`, `type`, `channel`, `device_mode`)
* giá trị nằm trong `min`/`max`
* graph không có chu trình, có node xuất ảnh
* quy ước riêng: `bbox/` prefix, `type=flux` khi có T5, `cfg=1.0` cho schnell
* file UI đọc ngược lại phải ra đúng graph API gốc
* notebook chỉ được nhắc tới workflow **thật sự có** trong repo

Muốn có spec mới nhất (sau khi ComfyUI/Impact Pack cập nhật):

```bash
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /tmp/ComfyUI
git clone --depth 1 https://github.com/city96/ComfyUI-GGUF /tmp/ComfyUI-GGUF
git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack /tmp/Impact-Pack
git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack /tmp/Impact-Subpack

python3 scripts/node_spec.py --comfy /tmp/ComfyUI --gguf /tmp/ComfyUI-GGUF \
        --impact /tmp/Impact-Pack --subpack /tmp/Impact-Subpack -o workflows/node_spec.json
```
