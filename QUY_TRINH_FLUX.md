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
| cfg | **1.0** (2.0 khi bật negative) | `cfg=1.0` → ComfyUI bỏ luôn nhánh negative (`samplers.py:610`); muốn negative có tác dụng phải `cfg ≥ 2.0` |
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

`denoise 0.5` · `steps 6` · `grow_mask 12px` · `cfg` theo ô CFG của Cell 6
(để negative có tác dụng cả ở bước sửa mặt/sửa tay)

---

## 💡 Prompt — cách viết để KHÔNG bị lỗi

### Quy tắc 1: Negative prompt CHỈ ăn khi cfg > 1.0

`comfy/samplers.py:610` — `if math.isclose(cond_scale, 1.0): uncond_ = None`.
Với `cfg = 1.0` (mặc định của schnell), ComfyUI **bỏ hẳn nhánh negative**: viết
"no extra fingers" vào cũng không được đọc.

→ Có hai đường, và nên dùng cả hai:

| | Cách | Chi phí |
|---|---|---|
| **(a)** | Tránh lỗi ngay trong prompt **dương** (quy tắc 2) | miễn phí, hiệu quả nhất |
| **(b)** | Bật negative trong Cell 6 → Cell tự nâng `cfg 1.0 → 2.0`, `steps 4 → 8` | chậm hơn rõ rệt |

Cell 6 in rõ trạng thái trước khi chạy: `✅ Negative đang BẬT (cfg=2.0 > 1.0, 27 từ)` hoặc
`⚠️ Có negative mà cfg=1.0 → negative KHÔNG được đọc.` — nhìn vào đó là biết negative có
được tính hay không, không phải đoán.

### Quy tắc 2: Tránh lỗi tay bằng cách giấu/cấp việc cho tay

FLUX.1-schnell là model chưng cất 4 bước, `cfg=1.0` → rất ít lực lái để sửa giải phẫu.
Cách hiệu quả nhất là **đừng bắt model phải tự bịa ngón tay**:

| Viết cái này | Đừng viết |
|---|---|
| `hands tucked into pockets` | `five fingers` |
| `both hands wrapped around a ceramic cup` | `perfect hands` |
| `hands clasped together on her lap` | `detailed fingers` |
| `carrying a canvas tote bag` | `(tay trôi nổi, không tả gì)` |

Nghịch lý: càng nhấn mạnh **số ngón**, model chưng cất càng hay sinh **thêm** ngón.

### Quy tắc 3: FLUX hiểu câu tự nhiên, không phải tag soup

Viết theo thứ tự: **chủ thể → tư thế/tay → trang phục/bối cảnh → ánh sáng → ống kính → khung hình**.

```
Close-up portrait of a young Vietnamese woman, natural skin with visible pores,
soft window light from the left, 85mm lens, shallow depth of field,
head and shoulders framing, plain warm backdrop, subtle film grain
```

### Quy tắc 4: Giữ ~1 megapixel

832×1216 (dọc) · 1216×832 (ngang) · 1024×1024 (vuông).
Xa khỏi ~1MP (VD 512² hay 2048²) thì schnell bắt đầu sinh lỗi cấu trúc: thừa chi, méo mặt.

### Dùng preset có sẵn (đỡ phải gõ)

Cell 6 có ô **PRESET** với 9 prompt đã được thiết kế sẵn, kèm nhãn rủi ro:

| Preset | Rủi ro lỗi |
|---|---|
| `chan_dung_can` — cận cảnh, không có tay | **Thấp** |
| `toan_than_tui_quan` — toàn thân, tay trong túi | **Thấp** |
| `toan_than_ngoi` — ngồi, tay đan trên đùi | **Thấp** |
| `phong_canh`, `san_pham` — không có người | **Thấp** |
| `ban_than_cam_coc`, `thoi_trang`, `duong_pho` — tay có việc làm | Trung bình |
| `anh_minh_hoa` — anime (YOLO mặt không nhận diện được mặt anime) | **Cao** |

Danh sách nằm ở `workflows/prompts.json`, nguồn là `scripts/prompt_presets.py`.

Mẹo nhanh:
- Đổi góc máy: thêm `low angle` / `close-up` / `full body in frame`.
- Cố định nhân vật: giữ nguyên `seed`, chỉ đổi một cụm mô tả mỗi lần.
- **Tay vẫn lỗi**: chuyển sang `flux_q5_quality`, tăng denoise tay `0.28 → 0.35`,
  hoặc dùng Cell 7 tô lên bàn tay rồi inpaint.
- **Mặt vẫn lỗi**: đừng dùng `flux_q5_fast` (không có FaceDetailer); dùng `standard`/`quality`.

### Giao diện Cell 6 — từng ô là gì

| Ô | Chức năng |
|---|---|
| `PRESET` | 9 cảnh dựng sẵn. Chọn preset → prompt, pipeline, khung hình **và negative** đều theo preset. Chọn `(tự viết prompt ở dưới)` để tự gõ |
| `PIPELINE` | 5 pipeline (bị preset ghi đè nếu bạn chọn preset) |
| `PROMPT` | Prompt dương |
| `THEM_VAO_PROMPT` | Nối thêm vào cuối prompt (kể cả khi dùng preset) — tiện đổi một chi tiết mà không sửa preset |
| `NEG_MODE` | 9 chế độ negative, xem bảng dưới |
| `NEGATIVE_PROMPT` | Negative tự gõ (dùng ở chế độ `tu_viet`, hoặc nối thêm ở mọi chế độ khác) |
| `CFG` | 1.0 = nhanh, negative **không** được đọc · ≥ 2.0 = negative bắt đầu có tác dụng |
| `TU_DONG_BAT_CFG` | Có negative mà `cfg = 1.0` → tự nâng `cfg = 2.0`, `steps = 8` và in ra lý do |
| `STEPS` / `BC_SUA_CHI_TIET` | Số bước của KSampler chính / của bước sửa mặt-tay (FaceDetailer) |
| `SAMPLER` / `SCHEDULER` | Đổi thuật toán lấy mẫu |
| `SEED` | `-1` = ngẫu nhiên mỗi ảnh; số ≥ 0 = cố định (để so sánh) |
| `SO_ANH` | Số ảnh mỗi prompt (> 2 trên T4 16GB sẽ bị cảnh báo OOM) |
| `SIZE` | 6 khung hình ~1MP, hoặc `theo preset` |
| `TEN_FILE` | Tiền tố tên file đầu ra |
| `NHIEU_PROMPT` | Mỗi dòng một prompt → chạy lần lượt, bỏ qua ô `PROMPT` |
| `LUU_VAO_DRIVE` | Copy ảnh vào `MyDrive/FLUX_output` |

**9 chế độ negative** (đều đọc từ `workflows/prompts.json`, nguồn `scripts/prompt_presets.py`):

| Chế độ | Dùng khi |
|---|---|
| `theo preset` | Mặc định — negative gắn sẵn, khớp từng cảnh |
| `chung` | Chống lỗi tổng quát (mờ, méo, watermark, thừa chi…) |
| `tay` | Đang ra ảnh lỗi ngón tay |
| `mat` | Đang ra ảnh lỗi mắt/mặt |
| `chu` | Ảnh hay dính chữ/ký tự rác (phổ biến ở ảnh sản phẩm) |
| `co_the` | Thừa tay chân, dính chi |
| `tat_ca` | Ghép mọi khối (dài nhất, encode T5 lâu nhất) |
| `tu_viet` | Chỉ dùng chuỗi ở ô `NEGATIVE_PROMPT` |
| `khong` | Tắt negative — về `cfg = 1.0`, 4 bước, nhanh nhất |

Gọi bằng code cũng được: `nhanh("prompt")` · `dep("prompt", n=2)` ·
`generate(prompt="...", neg_mode="tay - lỗi bàn tay", cfg=3.0, steps=12)`.
Cấu hình của lượt chạy cuối được ghi ở `/content/lan_chay_cuoi.json`.

---

## 🔁 Quy trình toàn diện (Cell 8c) — tạo → đo → sửa → chốt

Thay vì "tạo rồi nhắm mắt chọn", Cell 8c chạy một vòng kín:

```
   TẠO n ứng viên (mỗi cái một seed)
        │
        ▼
   ĐO từng ảnh ────── đạt? ──── CÓ ──► CHỐT ảnh điểm cao nhất
        │                                     │
        └── KHÔNG ──► SỬA                     ▼
                      lần 1: làm nét 0.35   báo cáo JSON + hiện ảnh
                      lần 2: nâng pipeline
                      (vẫn không đạt → trả ảnh tốt nhất + nói rõ lỗi còn lại)
```

| Mức | Ứng viên | Pipeline | Số lần sửa | Dùng khi |
|---|---|---|---|---|
| `nhanh` | 2 | theo preset | 0 | thử prompt, cần kết quả ngay |
| `chuan` | 3 | theo preset | 1 (làm nét) | mặc định |
| `ky` | 4 | **ép `quality`** | 2 (làm nét + nâng `hires`) | ảnh để dùng thật |

**Tiêu chí chấm điểm** (chỉ dùng thứ đo được bằng thống kê ảnh):

| Lỗi | Cách đo | Trừ |
|---|---|---|
| mờ | độ nét < `NGUONG_NET` × **mốc tự hiệu chuẩn** (chính ảnh đó bị mờ radius=2) | −30 |
| cháy sáng | > 2% pixel ≥ 250 | −25 |
| quá tối | > 2% pixel ≤ 5 | −15 |
| loãng | độ lệch chuẩn mức xám < 15 | −15 |

Ngưỡng "mờ" **không phải số bịa ra**: nó so với chính bức ảnh đó khi bị làm mờ nhân tạo,
nên áp dụng được cho mọi nội dung ảnh. Mặc định `NGUONG_NET = 3.0` (gấp 3 lần mốc mờ).

⚠️ **Điểm số KHÔNG đo được giải phẫu.** Thừa ngón, méo mặt, dính chi — máy tính hiện tại
không tự đánh giá được mấy thứ đó, và em sẽ không giả vờ là có. Khoản đó vẫn nhờ
**preset + prompt** (Cell 6) hoặc **sửa tay bằng Cell 7** (tô vùng rồi inpaint).

Gọi bằng code:

```python
tao_anh_tot(preset="chan_dung_can", muc="ky", so_ung_vien=4)
tao_anh_tot(prompt="a lighthouse at dawn", size="1216x832 (ngang, ~1MP)", muc="nhanh")
```

Kết quả: ảnh được chọn + bảng điểm mọi ứng viên + `/content/bao_cao_chat_luong.json`.

## 🧭 Kiểm tra tài nguyên (Cell 8)

Cell 8 giờ in thêm: GPU/VRAM/RAM/ổ đĩa (đọc từ `/system_stats` + `shutil.disk_usage`) và
**cấu hình khuyên dùng theo VRAM thực tế của máy**:

| VRAM | Khung hình | Pipeline | VAE_PREC |
|---|---|---|---|
| ≥ 20 GB | 1024×1024 | `quality` | Mặc định |
| ≥ 12 GB | 832×1216 | `quality` | Mặc định |
| ≥ 8 GB | 832×1216 | `standard` | Mặc định |
| < 8 GB | 768×1024 | `fast` | `fp16-vae` (ảnh sẽ mờ hơn — xem mục ảnh mờ) |

`/system_stats` chỉ trả `devices[].vram_total/vram_free` và `system.ram_total/ram_free`
(**không có** thông tin ổ đĩa), nên ổ đĩa lấy bằng `shutil.disk_usage('/content')`.

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
| **Ảnh mờ / loãng** | VAE đang chạy fp16 · cfg>1 · UNET lượng tử mạnh | Xem mục [Ảnh bị mờ](#-ảnh-bị-mờ) bên dưới |

Khi nghi ngờ: **chạy Cell 8** — nó liệt kê đúng các node còn thiếu, model ComfyUI nhìn thấy,
và VRAM đang dùng.

### 🫧 Ảnh bị mờ

"Mờ" là cảm giác — phải **đo** mới sửa được. Chạy **Cell 8b**: nó đo độ nét bằng phương sai
Laplacian trên 4 cấu hình (cùng prompt, cùng seed) rồi xếp hạng, kèm **mốc tham chiếu tự hiệu
chuẩn trên chính ảnh của bạn** (so với bản ảnh đó bị làm mờ radius=2). Không có ngưỡng cố định
nào được bịa ra.

Ba nguyên nhân, theo thứ tự nên kiểm tra:

**1. VAE đang chạy fp16 — nguyên nhân số 1.** `comfy/sd.py:1101`:

```python
if dtype is None:
    dtype = model_management.vae_dtype(self.device, self.working_dtypes)
self.vae_dtype = dtype
self.first_stage_model.to(self.vae_dtype)   # cast TOÀN BỘ VAE
```

VAE chuẩn có `working_dtypes = [bf16, fp32]` (`sd.py:515`) — **không có fp16**. Cờ
`--fp16-vae` ép thẳng fp16, bỏ qua danh sách được phép. Mặc định của ComfyUI trên T4
(không có bf16) là **fp32**. Decode bằng fp16 mất mantissa → ảnh mờ, loãng màu.
→ Cell 5: `VAE_PREC = "Mặc định"` (hoặc `fp32-vae`), rồi **chạy lại Cell 5** (phải khởi
động lại ComfyUI mới có tác dụng). fp16 chỉ nên dùng khi thật sự thiếu VRAM (~1.5 GB).

**2. `cfg > 1` trên model chưng cất.** schnell được chưng cất cho **4 bước, cfg = 1.0**.
Nâng cfg (để negative có tác dụng) hoặc chạy 8 bước đều có thể làm ảnh mềm/cháy.
→ Cell 8b sẽ nói rõ ② có mờ hơn ① không. Nếu có: `NEG_MODE = khong`, hoặc hạ `CFG` 1.5.

**3. UNET lượng tử quá mạnh.** `flux1-schnell-Q5_K_S` là bản Q5 nhỏ nhất. Nếu VRAM còn,
thử `Q5_K_M` / `Q6_K` (Cell 3 đổi tên file).

**Làm nét (`SAC_NET`)** là vá triệu chứng, không phải chữa nguyên nhân — hãy đo bằng Cell 8b
trước. `alpha = 1.0` của node `ImageSharpen` đã rất mạnh (kernel = gaussian × `-(alpha*10)`,
`comfy_extras/nodes_post_processing.py`); mức vừa dùng là **0.2–0.4**.

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
