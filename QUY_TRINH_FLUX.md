# 🎨 Quy trình FLUX.1-schnell GGUF Q5 trên Colab (bản tối ưu 2026-09)

Notebook: [`ComfyUI_Colab_WAI_fixed.ipynb`](ComfyUI_Colab_WAI_fixed.ipynb)
Chạy trên **Colab free T4 16 GB**, model để trên Drive (~12 GB).

---

## 🚀 Chạy lần đầu (đúng 6 bước)

| Bước | Cell | Việc |
|---|---|---|
| 1 | ⚙️ Cell 1 | Nhận GPU → tự chọn PROFILE → mount Drive → clone ComfyUI → cài requirements |
| 2 | 🧩 Cell 2 | Clone 3 custom node → cài dependency → **kiểm tra import** |
| 3 | ⬇️ Cell 3 | Tải 7 model (~12 GB, tải song song 3 luồng, có mirror) |
| 4 | 🧠 Cell 4 | Ghi 5 workflow tối ưu vào `/content/workflows` |
| 5 | 🚀 Cell 5 | Khởi chạy ComfyUI + tunnel cloudflared |
| 6 | 🖼 Cell 6 | **Tạo ảnh ngay trong Colab** — không cần mở giao diện |

Từ lần thứ hai: tick `BO_QUA_MODEL = True` ở Cell 3 (model đã có trên Drive) → chỉ mất ~2 phút.

Cell 7 = inpaint vẽ tay · Cell 8 = chẩn đoán · Cell 9 = tunnel dự phòng.

---

## ⚙️ Tham số (đã đặt sẵn, không cần chỉnh)

### Model

| Vai trò | File | Dung lượng |
|---|---|---|
| UNET | `flux1-schnell-Q5_K_S.gguf` | 8.26 GB |
| T5-XXL | `t5-v1_1-xxl-encoder-Q4_K_M.gguf` | 2.9 GB |
| CLIP-L | `clip_l.safetensors` | 246 MB |
| VAE | `ae.safetensors` | 335 MB |
| YOLO mặt | `bbox/face_yolov8m.pt` | 52 MB |
| YOLO tay | `bbox/hand_yolov8s.pt` | 22 MB |
| SAM | `sam_vit_b_01ec64.pth` | 375 MB |

### KSampler chính

| Tham số | Giá trị | Vì sao |
|---|---|---|
| steps | **4** | schnell là model chưng cất 4 bước; thêm bước chỉ tốn thời gian |
| cfg | **1.0** | schnell không dùng CFG → ComfyUI bỏ luôn nhánh negative |
| sampler | `euler` | — |
| scheduler | `simple` | — |
| size | 1024×1024 / 832×1216 / 1216×832 | giữ ~1 MP, bội số của 16 |

### FaceDetailer

| Vùng | denoise | steps | crop | feather | SAM |
|---|---|---|---|---|---|
| Mặt | 0.22 | 4 | 3.0 | 8 | ✅ `sam_vit_b`, threshold 0.93 |
| Tay | 0.28 | 4 | 2.5 | 16 | ❌ (tiết kiệm VRAM) |

`guide_size 384`, `max_size 768`, `tiled_encode = tiled_decode = True`.

### Khởi chạy ComfyUI (Cell 5)

```
--reserve-vram 1.0   --force-fp16   --fp16-vae
--use-pytorch-cross-attention (SDPA)   --disable-xformers   --preview-method taesd
```

**Không truyền `--lowvram`** — ComfyUI ≥ 0.3x đã bật Dynamic VRAM mặc định cho GPU NVIDIA,
`--lowvram` khi đó hầu như vô tác dụng và còn làm chậm. Chỉ thêm khi bạn chọn tay.

### Inpaint (Cell 7)

`denoise 0.5` · `steps 6` · `grow_mask 12px` · `cfg 1.0`

---

## 💡 Prompt

FLUX hiểu câu mô tả tự nhiên, không cần "tag soup". Viết theo thứ tự:
**chủ thể → trang phục/bối cảnh → ánh sáng → ống kính → chi tiết cần giữ**.

```
photorealistic portrait of a young Vietnamese woman, natural skin texture with visible pores,
soft window light, 85mm lens, shallow depth of field, detailed eyes and hands, five fingers,
casual linen shirt, warm neutral background, film grain, high detail
```

**Negative để trống.** Với `cfg = 1.0` thì negative không được dùng; để trống còn giúp T5
encode nhanh hơn.

Mẹo nhanh:
- Muốn đổi góc máy: thêm `low angle` / `close-up` / `full body shot`.
- Muốn cố định nhân vật: giữ nguyên `seed`, chỉ đổi một cụm mô tả mỗi lần.
- Tay vẫn lỗi: tăng denoise tay `0.28 → 0.35`, hoặc dùng Cell 7 tô lên bàn tay.

---

## 🛠 Xử lý lỗi

| Hiện tượng | Nguyên nhân | Cách sửa |
|---|---|---|
| **Không thấy node `UnetLoaderGGUF`** | thiếu package `gguf` | `!pip install -q "gguf>=0.13.0" sentencepiece protobuf`, khởi động lại Cell 5 |
| **Không thấy `FaceDetailer` / `SAMLoader`** | thiếu `scikit-image` | `!pip install -q scikit-image piexif dill segment-anything matplotlib`, khởi động lại Cell 5 |
| **`POST /prompt` trả 400** | workflow thiếu input required | Chạy Cell 4 để ghi lại workflow mới; xem lỗi ở `/content/comfyui.log` |
| **`[Impact Subpack] model file ... is not found`** | `model_name` thiếu tiền tố `bbox/` | Chạy lại Cell 4 (workflow mới đã có tiền tố) |
| **`CUDA out of memory`** | VRAM không đủ | Giảm size còn 832×832 · `VAE_PREC = cpu-vae` · `PREVIEW = none` · tắt Cell 7 |
| **`TypeError: resume_download`** | `huggingface_hub` ≥ 1.x đã xoá tham số | Chạy lại Cell 3 của bản notebook này (đã sửa) |
| **HF bị chặn** | mạng | Tick `USE_HF_MIRROR = True` ở Cell 1 rồi chạy lại |
| **Tải model đứng im** | rớt mạng giữa chừng | Chạy lại Cell 3 — `curl -C -` tiếp tục từ phần đã tải |
| **`TAESD previews enabled, but could not find models/vae_approx/taef1_decoder`** | tên file preview sai | Cell 3 của bản này tải đúng `taef1_decoder.pth`; nếu vẫn lỗi, đặt `PREVIEW = auto` |
| **cloudflared không ra link** | block UDP/QUIC | Chuyển `TUNNEL` sang `cloudflared http2`, hoặc chạy Cell 9 (localtunnel) |

Khi nghi ngờ: **chạy Cell 8** — nó liệt kê đúng các node còn thiếu, model ComfyUI nhìn thấy,
và VRAM đang dùng.

---

## 📚 Tài liệu kèm theo

| File | Nội dung |
|---|---|
| [`docs/AUDIT_FLUX_2026-09.md`](docs/AUDIT_FLUX_2026-09.md) | Quét quy trình cũ: 12 lỗi + bằng chứng trích từ mã nguồn |
| [`workflows/README.md`](workflows/README.md) | Sơ đồ node từng pipeline, cách sinh & kiểm tra |
| [`QUY_TRINH_FLUX.md`](QUY_TRINH_FLUX.md) | File này — hướng dẫn sử dụng |

## 🔧 Nếu muốn chỉnh workflow

Workflow **không sửa tay** — chúng được sinh từ `scripts/build_workflows.py`.
Sửa file đó, rồi chạy:

```bash
python3 scripts/check_sync.py --fix
```

Lệnh này sinh lại `workflows/`, `workflows/ui/` và notebook, so từng byte, rồi kiểm tra tĩnh
đối chiếu với 891 node class trích từ mã nguồn thật. Nếu bạn sửa file JSON trực tiếp, CI trên
GitHub sẽ báo lệch.
