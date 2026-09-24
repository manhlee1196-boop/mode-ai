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
# sinh workflows/ (API) + workflows/ui/ (có layout, kéo-thả vào giao diện)
python3 scripts/build_workflows.py
for f in workflows/flux_q5_*.json; do python3 scripts/api_to_ui.py "$f"; done

# kiểm tra tĩnh: đối chiếu 891 node class trích từ mã nguồn thật
python3 scripts/validate_workflows.py
```

| Script | Việc |
|---|---|
| `scripts/node_spec.py` | Trích `INPUT_TYPES`/`RETURN_TYPES` từ mã nguồn ComfyUI + GGUF + Impact Pack/Subpack → `workflows/node_spec.json` |
| `scripts/build_workflows.py` | Sinh 5 pipeline (cũng là đoạn nhúng trong Cell 4) |
| `scripts/api_to_ui.py` | Đổi API format ↔ UI format (có layout) |
| `scripts/validate_workflows.py` | Kiểm tra tĩnh mọi workflow + tham chiếu trong notebook |
| `scripts/make_notebook.py` | Sinh notebook, tự kiểm tra trước khi ghi |

Kiểm tra không cần GPU, không cần ComfyUI chạy — chỉ cần `workflows/node_spec.json`
(đã có sẵn trong repo). Cách cập nhật spec: `workflows/README.md`.
