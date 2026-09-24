# 🔍 Quét & đánh giá quy trình FLUX hiện có — 2026-09-24

Tài liệu này ghi lại **bằng chứng** cho từng lỗi được sửa trong bản viết lại của
`ComfyUI_Colab_WAI_fixed.ipynb`. Mọi câu "đối chiếu với mã nguồn" bên dưới đều lấy từ bản clone
tại thời điểm viết:

| Thành phần | Commit / phiên bản |
|---|---|
| ComfyUI | `1568e6cfd04586a4b3c4e1817ea7dde09b1bf9e7` (2026-09-23), `comfyui_version.__version__ = 0.37.0` |
| ComfyUI-GGUF | `main`, `pyproject.toml` version `2.0.0` |
| ComfyUI-Impact-Pack | `main` |
| ComfyUI-Impact-Subpack | `main`, version `1.3.5` |
| `huggingface_hub` trên PyPI | `2.0.0` |
| `ultralytics` trên PyPI | `8.4.161` |

---

## 1. Thiếu package → node GGUF không tồn tại

**Triệu chứng trên Colab:** mở giao diện không thấy `UnetLoaderGGUF` / `DualCLIPLoaderGGUF`,
hoặc có node nhưng khi Queue thì báo `ERROR: Could not detect model type`.

**Bằng chứng.** `ComfyUI-GGUF/nodes.py` ngay dòng đầu đã `from .loader import ...`, và
`loader.py` có `import gguf`:

```python
# ComfyUI-GGUF/loader.py
import gguf            # ← package "gguf", KHÔNG phải thư viện có sẵn
from .ops import GGMLTensor
```

`ComfyUI-GGUF/requirements.txt`:
```
gguf>=0.13.0
sentencepiece
protobuf
```

**Notebook cũ cài gì (Cell 1B):**
```python
!pip install -q piexif dill segment-anything
!pip install -q ultralytics --no-deps
```
→ **không có `gguf`, `sentencepiece`, `protobuf`**. Cả 3 thiếu.

**Sửa:** Cell 2 cài đủ 3 gói này và kiểm tra `gguf.__version__ >= 0.13` ngay sau đó.

---

## 2. Thiếu `scikit-image` → toàn bộ Impact Pack không load

**Bằng chứng.** `ComfyUI-Impact-Pack/__init__.py` dòng 20–36:

```python
try:
    from skimage.measure import label, regionprops   # noqa: F401
    ...
except Exception as e:
    logging.error("[Impact Pack] Failed to import due to several dependencies are missing!!!!")
    raise e                                          # ← raise, không phải warning
```

`skimage` đến từ package **`scikit-image`**, mà `ComfyUI-Impact-Pack/requirements.txt` là:
```
segment-anything
scikit-image          # ← notebook cũ KHÔNG cài
scikit-image
piexif
transformers
opencv-python-headless
scipy
numpy
dill
matplotlib
git+https://github.com/facebookresearch/sam2
```

**Notebook cũ chỉ cài** `piexif dill segment-anything` (thiếu `scikit-image`, `matplotlib`,
và `transformers` thì may mắn có sẵn trên Colab).

**Hậu quả:** `FaceDetailer`, `SAMLoader`, `UltralyticsDetectorProvider` **đều biến mất**,
nhưng ComfyUI vẫn khởi động bình thường → người dùng chỉ phát hiện khi load workflow và
thấy node đỏ.

**Sửa:** Cell 2 cài `scikit-image piexif dill segment-anything matplotlib`, rồi **in bảng
kiểm tra import** (✅/❌ từng package + lý do cần) và `raise` nếu thiếu — thay vì để lỗi
âm thầm như bản cũ.

---

## 3. Workflow cũ thiếu input bắt buộc của `FaceDetailer`

**Bằng chứng.** Trích `INPUT_TYPES` thật từ `ComfyUI-Impact-Pack/modules/impact/impact_pack.py`
(lớp `FaceDetailer`). Danh sách `required` kết thúc bằng:

```
... "drop_size", "bbox_detector", "wildcard", "cycle"
```

`execution.py` của ComfyUI (`validate_inputs`) sinh lỗi `required_input_missing` cho mọi
input trong `required` mà prompt không có.

**Workflow cũ** (`workflows/flux_schnell_gguf_toi_uu.json` trên branch cũ) thiếu cả
`wildcard` lẫn `cycle` ở cả 3 node FaceDetailer → **`POST /prompt` trả HTTP 400**.

Đây là kết quả chạy trình kiểm tra của repo này lên workflow cũ:

```
❌ flux_schnell_gguf_toi_uu.json              17 node  12 lỗi
   ✗ [20:FaceDetailer]: THIẾU input required `wildcard`
   ✗ [20:FaceDetailer]: THIẾU input required `cycle`
   ✗ [21:FaceDetailer]: THIẾU input required `wildcard`
   ✗ [21:FaceDetailer]: THIẾU input required `cycle`
   ✗ [22:FaceDetailer]: THIẾU input required `wildcard`
   ✗ [22:FaceDetailer]: THIẾU input required `cycle`
```

**Sửa:** `scripts/build_workflows.py` luôn ghi đủ bộ required (kể cả `wildcard: ""`,
`cycle: 1`), và `scripts/validate_workflows.py` bắt lỗi này trước khi lên Colab.

---

## 4. `UltralyticsDetectorProvider` cần tiền tố `bbox/`

**Bằng chứng.** `ComfyUI-Impact-Subpack/modules/subpack_nodes.py`:

```python
bboxs = ["bbox/"+x for x in folder_paths.get_filename_list("ultralytics_bbox")]
...
def doit(self, model_name):
    model_path = folder_paths.get_full_path("ultralytics", model_name)
    if model_path is None:
        if model_name.startswith('bbox/'):
            model_path = folder_paths.get_full_path("ultralytics_bbox", model_name[5:])
```

Tên hiện ra trong combo **đã có sẵn** `bbox/`, nên giá trị trong JSON cũng phải là
`bbox/face_yolov8m.pt`.

**Workflow cũ** ghi `"model_name": "face_yolov8m.pt"` → `folder_paths.get_full_path("ultralytics", ...)`
trả `None` → `raise ValueError("[Impact Subpack] model file 'face_yolov8m.pt' is not found.")`.

**Sửa:** builder dùng hằng `bbox/face_yolov8m.pt`; trình kiểm tra báo lỗi nếu thiếu tiền tố.

---

## 5. `hf_hub_download(resume_download=True)` không còn tồn tại

**Bằng chứng.** Kiểm tra trực tiếp `huggingface_hub` 2.0.0 (bản mới nhất trên PyPI):

```
version 2.0.0
params: ['repo_id', 'filename', 'subfolder', 'repo_type', 'revision', 'library_name',
         'library_version', 'cache_dir', 'local_dir', 'user_agent', 'force_download',
         'etag_timeout', 'token', 'local_files_only', 'headers', 'endpoint',
         'tqdm_class', 'dry_run']
has resume_download: False     ← ĐÃ BỊ XOÁ
has local_dir: True
```

**Notebook cũ** (Cell 2, hàm `hf_get`):
```python
kw = dict(repo_id=repo, filename=fn, resume_download=True)
sf = hf_hub_download(**kw)     # → TypeError: unexpected keyword argument
```

Vì `hf_get` bọc trong `try/except` và chỉ `log('HF lỗi')`, người dùng sẽ thấy
"thử mirror lỗi" rồi… **hết mirror, model không tải được**, trong khi thông báo không nói rõ
nguyên nhân.

**Sửa:** bỏ `resume_download`, thay bằng `local_dir=` (vẫn có tính năng tiếp tục tải qua cache
của hub) và ưu tiên `curl -C -` (tiếp tục tải thật sự khi rớt mạng) trước khi dùng hub.

---

## 6. `--lowvram` không còn tác dụng trên ComfyUI mới

**Bằng chứng.** `comfy/cli_args.py` cuối file:

```python
def enables_dynamic_vram():
    if args.enable_dynamic_vram:
        return True
    return not args.disable_dynamic_vram and not args.highvram \
        and not args.gpu_only and not args.novram and not args.cpu
```

và `main.py`:

```python
if args.enable_dynamic_vram or (enables_dynamic_vram() and dynamic_vram_supported()):
    ...
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
```

`dynamic_vram_supported()` trả `True` cho mọi GPU NVIDIA. Nghĩa là **mặc định ComfyUI đã bật
Dynamic VRAM**, còn `--lowvram` chỉ được xử lý ở `comfy/model_management.py`:

```python
if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
...
if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to
```

→ `--lowvram` **không tắt** Dynamic VRAM, chỉ đổi `vram_state`, và trên thực tế còn làm
chậm vì ép text encoder sang CPU dù VRAM còn chỗ.

**Notebook cũ** đặt sẵn `VRAM = "lowvram"` và ghi chú *"bắt buộc trên T4 16GB"* — không còn đúng.

**Sửa:** Cell 5 mặc định là **không truyền cờ VRAM nào** (để Dynamic VRAM hoạt động), dùng
`--reserve-vram 1.0` để chừa chỗ trống, và chỉ thêm `--lowvram` khi người dùng chọn tay.

---

## 7. Tải workflow từ repo ngoài — link đã chết

**Bằng chứng.** Cell 3C của notebook cũ:

```python
BRANCHES_TO_TRY = ["arena/01a0d36a-t-i-li-u", "main", "arena/01a0cc76-t-i-li-u"]
url_test = f"https://raw.githubusercontent.com/caone1196-sketch/t-i-li-u/{br}/workflow_flux_schnell_Q5_realistic.json"
```

Kiểm tra thực tế (2026-09-24) tất cả 6 tổ hợp branch × file:

```
000  arena/01a0d36a-t-i-li-u/workflow_flux_schnell_gguf_toi_uu.json
000  main/workflow_flux_schnell_gguf_toi_uu.json
000  arena/01a0cc76-t-i-li-u/workflow_flux_schnell_gguf_toi_uu.json
000  arena/01a0d36a-t-i-li-u/workflow_flux_schnell_Q5_realistic.json
000  main/workflow_flux_schnell_Q5_realistic.json
000  arena/01a0cc76-t-i-li-u/workflow_flux_schnell_Q5_realistic.json
```

(`000` = không kết nối được / repo không tồn tại.)

**Sửa:** Cell 4 **nhúng thẳng** bộ sinh workflow vào notebook (lấy nguyên văn từ
`scripts/build_workflows.py`). Không phụ thuộc bất kỳ repo ngoài nào. `scripts/make_notebook.py`
có bước tự kiểm tra: đoạn code trong notebook phải **giống hệt từng ký tự** file gốc, và khi
chạy phải sinh ra đúng 5 workflow đã qua validate.

---

## 8. Phần tìm/tải WAI-illustrious (SDXL) — bỏ

Cell 1 của notebook cũ dành ~90 dòng để tìm `WAI-illustrious.safetensors` (6.9 GB SDXL) qua
Drive share, shortcut, `gdown`… nhưng **không một workflow FLUX nào dùng đến file này**, và
phần log của nó (`ls -lhR` toàn bộ Drive, `find` toàn bộ MyDrive) làm chậm Cell 1 thêm hàng
chục giây.

**Sửa:** bỏ hoàn toàn. Cell 1 chỉ: nhận diện GPU → tự chọn PROFILE → mount Drive → tìm/ tạo
thư mục model → clone ComfyUI → cài requirements.

---

## 9. Tối ưu hoá detailer (đo được bằng số node/thông lượng)

| Thông số | Cũ | Mới | Lý do |
|---|---|---|---|
| FaceDetailer `guide_size` | 512 | **384** | crop 384² vẫn đủ cho khuôn mặt 1024², diện tích latent giảm 44 % |
| FaceDetailer `max_size` | 1024 | **768** | trần crop thấp hơn → không bao giờ render lại toàn ảnh |
| FaceDetailer `steps` | 6 | **4** | cùng số bước với model chính (schnell distill) |
| Số FaceDetailer | 3 (mặt + tay + chân) | **1–2** | `foot_anime_yolo11m_v3.pt` là model **anime**, nhận diện chân người thật rất kém → hay tạo crop rác |
| Negative prompt | chuỗi dài | **trống** | với `cfg = 1.0` ComfyUI bỏ qua negative → encode T5 nhanh hơn |

---

## 10. TAESD preview: tên file sai

ComfyUI tìm file trong `models/vae_approx/` **bắt đầu bằng** `latent_format.taesd_decoder_name`
(`comfy/latent_formats.py`, lớp `Flux` → `self.taesd_decoder_name = "taef1_decoder"`;
kiểm tra tại `latent_preview.py` dòng 93–99).

Repo HF `madebyollin/taesd` **không có** file nào tên `taef1_*` (chỉ có `taesd_decoder`,
`taesdxl_decoder`, …). File đúng nằm ở:

* `https://github.com/madebyollin/taesd/raw/main/taef1_decoder.pth` (tên được sửa đúng
  trong commit *"Fix TAEF1 filenames"*, 2025-12-25), hoặc
* `UmeAiRT/ComfyUI-Auto-Installer-Assets` → `models/vae_approx/taef1_decoder.safetensors`.

---

## 11. Nguồn model thực tế (kiểm tra 2026-09-24)

| Model | Repo | File | Dung lượng |
|---|---|---|---|
| UNET Q5_K_S | `city96/FLUX.1-schnell-gguf` | `flux1-schnell-Q5_K_S.gguf` | 8.26 GB ✅ |
| T5-XXL Q4_K_M | `city96/t5-v1_1-xxl-encoder-gguf` | `t5-v1_1-xxl-encoder-Q4_K_M.gguf` | 2.9 GB ✅ |
| CLIP-L | `comfyanonymous/flux_text_encoders` | `clip_l.safetensors` | 246 MB ✅ |
| VAE | `Comfy-Org/Lumina_Image_2.0_Repackaged` | `split_files/vae/ae.safetensors` | 335 MB ✅ |
| VAE (dự phòng) | `camenduru/FLUX.1-dev` | `ae.safetensors` | 335 MB ✅ |
| YOLO mặt/tay | `Bingsu/adetailer` | `face_yolov8m.pt` / `hand_yolov8s.pt` | 52 / 22 MB ✅ |
| SAM ViT-B | `dl.fbaipublicfiles.com` | `sam_vit_b_01ec64.pth` | 375 MB ✅ |

Tổng: **~12.0 GB** — vừa Drive free 15 GB.

> ⚠️ `black-forest-labs/FLUX.1-schnell` là repo **gated** (phải đồng ý điều khoản mới tải được)
> nên không dùng làm nguồn.

---

## 12. Những thứ bản cũ làm ĐÚNG (được giữ lại)

Để công bằng, đây là các quyết định đúng của bản cũ và vẫn được giữ:

* Chọn **Q5_K_S** cho UNET: 8.26 GB — điểm cân bằng tốt nhất giữa chất lượng và 15 GB Drive
  (Q8_0 = 12.7 GB vượt tổng; Q4_K_S = 6.78 GB giảm chất lượng thấy rõ).
* **T5-XXL ở Q4_K_M** (2.9 GB): text encoder ít nhạy với lượng tử hoá hơn UNET.
* `sampler=euler` + `scheduler=simple` + `steps=4` + `cfg=1.0`: đúng chuẩn schnell.
* `tiled_encode=True` / `tiled_decode=True` trong FaceDetailer để tránh OOM.
* Dùng `curl -C -` để tiếp tục tải khi rớt mạng.
* Symlink `models/unet` + `models/unet_gguf` cùng trỏ về một thư mục GGUF (ComfyUI-GGUF đăng
  ký key `unet_gguf` với fallback `unet`).
