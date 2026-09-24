# mode-ai

Pipeline **FLUX.1-schnell GGUF Q5_K_S** chạy trên Google Colab (T4 16 GB, Drive free 15 GB)
qua ComfyUI.

```
notebook  ComfyUI_Colab_WAI_fixed.ipynb   ← chạy trên Colab
workflows  flux_q5_{fast,standard,quality,hires,inpaint}.json
scripts    sinh + kiểm tra workflow (chạy được không cần GPU)
docs       AUDIT_FLUX_2026-09.md: quét quy trình cũ, 12 lỗi kèm bằng chứng
```

## Dùng nhanh

Mở `ComfyUI_Colab_WAI_fixed.ipynb` trên Colab → chạy Cell 1 → 2 → 3 → 4 → 5 → **Cell 6**
(tạo ảnh ngay trong Colab, không cần mở giao diện).

Chi tiết tham số, prompt và xử lý lỗi: [`QUY_TRINH_FLUX.md`](QUY_TRINH_FLUX.md).

| Pipeline | Node | Ảnh 1024² trên T4 |
|---|---|---|
| `flux_q5_fast` | 9 | ~15-20 s |
| `flux_q5_standard` | 13 | ~25-35 s |
| `flux_q5_quality` | 15 | ~40-55 s |
| `flux_q5_hires` | 17 | ~70-90 s |
| `flux_q5_inpaint` | 12 | ~15-25 s |

**9 prompt preset có sẵn** (`workflows/prompts.json`, nguồn `scripts/prompt_presets.py`) —
chọn trong ô PRESET ở Cell 6. Chúng được viết để tránh lỗi giải phẫu: tả rõ tay đang cầm/giấu/
đan thay vì đòi "five fingers" (càng nhấn số ngón, model chưng cất càng hay sinh thêm ngón).

**Negative prompt có trong Cell 6**, nhưng kèm một sự thật cần nhớ: `comfy/samplers.py:610`
bỏ hẳn nhánh negative ở `cfg = 1.0`, nên Cell 6 tự nâng `cfg → 2.0`, `steps → 8` mỗi khi bạn
bật negative — và in rõ `✅ Negative đang BẬT` / `⚠️ negative KHÔNG được đọc` để bạn không phải
đoán. Muốn nhanh nhất: `NEG_MODE = khong - không dùng negative`. Chi tiết:
[`QUY_TRINH_FLUX.md`](QUY_TRINH_FLUX.md#-prompt--cách-viết-để-không-bị-lỗi).

**Quy trình khép kín (Cell 8c)**: tạo nhiều ứng viên → chấm điểm từng ảnh (mờ / cháy sáng /
tối / loãng) → tự sửa (làm nét, nâng pipeline) → chốt ảnh tốt nhất + báo cáo JSON.
Ba mức `nhanh` / `chuan` / `ky`. Ngưỡng "mờ" tự hiệu chuẩn trên chính bức ảnh, không phải
số cố định. **Điểm số không đo được giải phẫu** (thừa ngón, méo mặt) — khoản đó vẫn nhờ
preset/prompt và Cell 7. Chi tiết: [`QUY_TRINH_FLUX.md`](QUY_TRINH_FLUX.md).

**Cell 8** giờ cũng kiểm tra tài nguyên (VRAM/RAM/ổ đĩa) và đề xuất cấu hình phù hợp máy.

Model ~12 GB: UNET `flux1-schnell-Q5_K_S.gguf` (8.26) + T5-XXL `Q4_K_M` (2.9) +
CLIP-L (0.25) + VAE `ae.safetensors` (0.34) + YOLO mặt/tay + SAM.

## Vì sao viết lại

Bản notebook cũ không chạy được trên ComfyUI hiện tại. [`docs/AUDIT_FLUX_2026-09.md`](docs/AUDIT_FLUX_2026-09.md)
ghi đầy đủ bằng chứng; tóm tắt 6 lỗi làm gãy hoàn toàn pipeline:

1. Không cài `gguf` / `sentencepiece` / `protobuf` → `UnetLoaderGGUF` không tồn tại.
2. Không cài `scikit-image` → Impact Pack `raise` khi import → mất `FaceDetailer`, `SAMLoader`.
3. Workflow thiếu input required `wildcard`, `cycle` của `FaceDetailer` → `/prompt` trả 400.
4. `UltralyticsDetectorProvider` thiếu tiền tố `bbox/` → không tìm thấy file YOLO.
5. `hf_hub_download(resume_download=True)` đã bị xoá từ `huggingface_hub` 1.x.
6. Workflow được tải từ repo ngoài `caone1196-sketch/t-i-li-u` — link đã chết.

## Phát triển

Workflow **không viết tay** — `scripts/build_workflows.py` là nguồn duy nhất, và chính đoạn
code đó được nhúng vào Cell 4 của notebook. `scripts/make_notebook.py` từ chối ghi notebook
nếu hai bản lệch nhau.

```bash
python3 scripts/check_sync.py          # một lệnh làm hết (đồng bộ + kiểm tra tĩnh)
python3 scripts/check_sync.py --fix    # tự ghi lại artifact cho khớp nguồn
```

Nó sinh lại toàn bộ artifact vào thư mục tạm, so **từng byte** với bản đã commit, rồi chạy
kiểm tra tĩnh. Dùng khi bạn sửa `build_workflows.py` hoặc khi nghi ngờ ai đó sửa tay file JSON.

| Script | Việc |
|---|---|
| `scripts/check_sync.py` | **Một lệnh làm hết:** sinh lại → so khớp → kiểm tra tĩnh |
| `scripts/build_workflows.py` | Sinh 5 pipeline (cũng là đoạn nhúng trong Cell 4) |
| `scripts/api_to_ui.py` | Đổi API format ↔ UI format (có layout) |
| `scripts/make_notebook.py` | Sinh notebook, tự kiểm tra trước khi ghi |
| `scripts/validate_workflows.py` | Kiểm tra tĩnh mọi workflow + tham chiếu trong notebook |
| `scripts/node_spec.py` | Trích `INPUT_TYPES`/`RETURN_TYPES` từ mã nguồn ComfyUI + GGUF + Impact Pack/Subpack → `workflows/node_spec.json` |

Kiểm tra không cần GPU, không cần ComfyUI chạy, không cần mạng — chỉ cần
`workflows/node_spec.json` (đã commit sẵn). Cách cập nhật spec: `workflows/README.md`.

**CI** (`.github/workflows/validate-flux.yml`) chạy `check_sync.py` trên mọi push/PR.
Nó fail nếu artifact lệch khỏi nguồn hoặc workflow có lỗi — chạy xong trong ~8 giây.
