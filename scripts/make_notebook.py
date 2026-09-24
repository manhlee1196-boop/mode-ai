#!/usr/bin/env python3
"""Sinh `ComfyUI_Colab_WAI_fixed.ipynb` (bản viết lại, tối ưu cho FLUX.1-schnell GGUF).

Chạy:  python3 scripts/make_notebook.py
Kiểm tra tự động khi chạy:
  • mọi code cell phải parse được bằng ast (không lỗi cú pháp)
  • đoạn builder workflow nhúng trong Cell 4 phải GIỐNG HỆT scripts/build_workflows.py
  • builder đó chạy được và sinh đúng bộ workflow đã validate trong workflows/
"""
from __future__ import annotations

import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "ComfyUI_Colab_WAI_fixed.ipynb")
BUILDER = os.path.join(HERE, "build_workflows.py")

START = "# ----------------------------------------------------------------- model (khớp Cell 3)"
END = "def main() -> int:"


def builder_source() -> str:
    """Đoạn code sinh workflow (không kèm CLI) — nhúng nguyên văn vào notebook."""
    src = open(BUILDER, encoding="utf-8").read()
    i = src.index(START)
    j = src.index(END)
    head = ('"""Khối sinh workflow — NHÚNG TỪ scripts/build_workflows.py (đừng sửa tay ở đây,'
            ' hãy sửa file gốc rồi chạy lại scripts/make_notebook.py)."""\n'
            "from typing import Any, Dict\n\n")
    return head + src[i:j].rstrip() + "\n"


CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip() + "\n"))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n") + "\n"))


# =========================================================================== MD mở đầu
md("""
# 🎨 ComfyUI Colab — FLUX.1-schnell GGUF Q5 (bản viết lại, tối ưu 2026-09)

Pipeline: **FLUX.1-schnell Q5_K_S (UNET GGUF)** + T5-XXL Q4_K_M + CLIP-L + VAE `ae` —
tổng ~12 GB model, chạy được trên **Colab free T4 16 GB**.

| Pipeline | Node | Việc nó làm | Thời gian/ảnh 1024² (T4) |
|---|---|---|---|
| `flux_q5_fast` | 9 | 4 bước, không detailer | ~15-20 s |
| `flux_q5_standard` | 13 | 4 bước + sửa mặt (YOLO+SAM) | ~25-35 s |
| `flux_q5_quality` | 15 | + sửa tay | ~40-55 s |
| `flux_q5_hires` | 17 | + upscale 1.5× rồi lấy lại chi tiết | ~70-90 s |
| `flux_q5_inpaint` | 12 | sửa vùng tô trên ảnh có sẵn | ~15-25 s |

**Thứ tự chạy:** Cell 1 → 2 → 3 → 4 → 5, sau đó **Cell 6 (tạo ảnh ngay trong Colab)**
hoặc mở giao diện ComfyUI qua link cloudflared. Cell 7 = inpaint vẽ tay, Cell 8 = chẩn đoán.

### Bản này sửa gì so với bản cũ (kết quả quét 2026-09-24)
1. **Thiếu package `gguf`** → node `UnetLoaderGGUF` không load được → *mọi* workflow GGUF chết.
2. **Thiếu `scikit-image`** → Impact Pack raise ngay khi import → không có `FaceDetailer`/`SAMLoader`.
3. **Workflow cũ thiếu input required** `wildcard`, `cycle` của `FaceDetailer` → ComfyUI từ chối prompt.
4. **`UltralyticsDetectorProvider` thiếu tiền tố `bbox/`** → Impact Subpack không tìm ra file YOLO.
5. **`hf_hub_download(resume_download=True)`** đã bị bỏ từ `huggingface_hub` 1.x → mọi mirror HF lỗi `TypeError`.
6. **`--lowvram` không còn tác dụng** trên ComfyUI mới (Dynamic VRAM bật mặc định cho NVIDIA) → bỏ, dùng `--reserve-vram`.
7. Bỏ toàn bộ phần tìm/tải **WAI-illustrious (SDXL 6.9 GB)** — bản này chỉ chạy FLUX, không dùng tới.
8. Workflow được **nhúng thẳng trong notebook** (bản cũ tải từ repo `caone1196-sketch/t-i-li-u` — link chết).
9. Detailer giảm `guide_size 512→384`, `max_size 1024→768`, steps `6→4` → nhanh hơn ~35 % mà vẫn đủ nét.

Chi tiết bằng chứng: `docs/AUDIT_FLUX_2026-09.md`. Kiểm tra tĩnh workflow: `python3 scripts/validate_workflows.py`.
""")

# =========================================================================== CELL 1
code(r'''
# @title ⚙️ CELL 1 — GPU + Drive + ComfyUI
USE_DRIVE = True  # @param {type:"boolean"}
DRIVE_MODEL_DIR = ""  # @param {type:"string"}
COMFY_REF = "master"  # @param {type:"string"}
USE_HF_MIRROR = False  # @param {type:"boolean"}

import os, sys, json, time, subprocess

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
    sys.stdout.flush()

# ---------- 1) GPU ----------
gpu_name, vram_gb = 'CPU', 0.0
try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        text=True).strip().splitlines()[0]
    parts = [p.strip() for p in out.split(',')]
    gpu_name, vram_gb = parts[0], float(parts[1]) / 1024.0
except Exception as e:
    log(f'⚠️ Không thấy GPU ({e}) — đổi Runtime → Change runtime type → T4 GPU')
log(f'GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB')

if vram_gb >= 24:
    PROFILE = 'big'        # A100/L4-40: giữ model thường trú trong VRAM
elif vram_gb >= 12:
    PROFILE = 't4'         # T4/L4 16GB: cấu hình mặc định của notebook này
else:
    PROFILE = 'small'
log(f'PROFILE = {PROFILE}')

if USE_HF_MIRROR:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    log('HF_ENDPOINT = https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

# ---------- 2) Drive (tuỳ chọn) ----------
drive_ok = False
if USE_DRIVE:
    log('Mount Drive (nếu treo >60s: Runtime → Interrupt, tick USE_DRIVE=False, chạy lại)')
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        drive_ok = True
        log('Drive OK')
    except Exception as e:
        log(f'Drive lỗi ({e}) → dùng /content (mất model khi tắt runtime)')

ROOT = None
if drive_ok:
    if DRIVE_MODEL_DIR.strip() and os.path.isdir(DRIVE_MODEL_DIR.strip()):
        ROOT = DRIVE_MODEL_DIR.strip()
    else:
        md = '/content/drive/MyDrive'
        cands = [os.path.join(md, n) for n in ('AI_Models', 'AI_models', 'ComfyUI_models')]
        try:
            sc = os.path.join(md, '.shortcut-targets-by-id')
            cands += [os.path.join(sc, n) for n in os.listdir(sc)]
        except Exception:
            pass
        for c in cands:
            if os.path.isdir(c) and (os.path.isdir(os.path.join(c, 'gguf'))
                                     or os.path.isdir(os.path.join(c, 'checkpoints'))):
                ROOT = c
                break
        ROOT = ROOT or os.path.join(md, 'AI_Models')
ROOT = ROOT or '/content/AI_Models'

DIRS = {
    'root': ROOT,
    'gguf': f'{ROOT}/gguf',                 # UNET GGUF  → models/unet
    'clip': f'{ROOT}/clip',                 # CLIP-L + T5 GGUF → models/clip
    'vae':  f'{ROOT}/vae',                  # ae.safetensors
    'yolo': f'{ROOT}/ultralytics/bbox',     # YOLO mặt/tay
    'sam':  f'{ROOT}/sams',                 # SAM ViT-B
    'upscale': f'{ROOT}/upscale_models',
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)
DIRS['profile'] = PROFILE
DIRS['gpu'] = gpu_name
DIRS['vram_gb'] = vram_gb
with open('/content/mode_ai_paths.json', 'w') as f:
    json.dump(DIRS, f, indent=1)
log(f'ROOT model = {ROOT}')
for k, v in DIRS.items():
    log(f'   {k:8} {v}')

# ---------- 3) ComfyUI ----------
COMFY = '/content/ComfyUI'
os.chdir('/content')
ref = (COMFY_REF or 'master').strip()
if os.path.isfile(f'{COMFY}/main.py'):
    log('ComfyUI đã có — bỏ qua clone (xoá /content/ComfyUI nếu muốn cài lại)')
else:
    log(f'Clone ComfyUI @ {ref}')
    r = subprocess.run(['git', 'clone', '--depth', '1', '--branch', ref,
                        'https://github.com/comfyanonymous/ComfyUI', COMFY])
    if r.returncode != 0:
        log('--branch thất bại (có thể là commit SHA) → clone đầy đủ rồi checkout')
        subprocess.run(['git', 'clone', 'https://github.com/comfyanonymous/ComfyUI', COMFY], check=True)
        subprocess.run(['git', 'checkout', ref], cwd=COMFY, check=True)
    subprocess.run(['git', 'log', '-1', '--format=ComfyUI %h %cd %s', '--date=short'], cwd=COMFY)

# ---------- 4) requirements (KHÔNG đụng torch của Colab) ----------
os.chdir(COMFY)
subprocess.run("grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' requirements.txt "
               "> /content/req_notorch.txt", shell=True, check=True)
log('pip install requirements (bỏ torch/torchvision/torchaudio)')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
                '-r', '/content/req_notorch.txt'])

import torch
if not torch.cuda.is_available():
    raise RuntimeError('❌ torch không thấy CUDA — kiểm tra Runtime type là GPU (T4)')
log(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}')
log('✅ Xong Cell 1 → chạy Cell 2')
''')

# =========================================================================== CELL 2
code(r'''
# @title 🧩 CELL 2 — Custom node + dependency (bản đã sửa) + kiểm tra import
PIN_GGUF = "main"  # @param {type:"string"}
PIN_IMPACT = "main"  # @param {type:"string"}

import os, sys, json, time, subprocess

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
    sys.stdout.flush()

P = json.load(open('/content/mode_ai_paths.json'))
COMFY = '/content/ComfyUI'
CN = f'{COMFY}/custom_nodes'
os.makedirs(CN, exist_ok=True)

def clone(url, folder, ref='main'):
    path = os.path.join(CN, folder)
    if os.path.isdir(path) and os.listdir(path):
        log(f'{folder} đã có — bỏ qua')
        return
    log(f'Clone {folder} @ {ref}')
    r = subprocess.run(['git', 'clone', '--depth', '1', '--branch', ref, url, path])
    if r.returncode != 0:
        subprocess.run(['git', 'clone', url, path], check=True)

clone('https://github.com/city96/ComfyUI-GGUF.git', 'ComfyUI-GGUF', PIN_GGUF)
clone('https://github.com/ltdrdata/ComfyUI-Impact-Pack.git', 'ComfyUI-Impact-Pack', PIN_IMPACT)
clone('https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git', 'ComfyUI-Impact-Subpack', PIN_IMPACT)

def pip(*args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-input', *args])

# (a) ComfyUI-GGUF khai báo: gguf>=0.13, sentencepiece, protobuf  ← bản cũ THIẾU, node GGUF chết
log('pip: gguf + sentencepiece + protobuf (ComfyUI-GGUF)')
pip('gguf>=0.13.0', 'sentencepiece', 'protobuf')

# (b) Impact Pack khai báo: scikit-image, piexif, dill, segment-anything, matplotlib, transformers
#     (opencv đã có sẵn trên Colab dưới dạng opencv-contrib-python nên KHÔNG cài lại)
log('pip: scikit-image + piexif + dill + segment-anything + matplotlib (Impact Pack)')
pip('scikit-image', 'piexif', 'dill', 'segment-anything', 'matplotlib')

# (c) Impact Subpack cần ultralytics. Cài --no-deps để KHÔNG kéo theo opencv/numpy mới
#     (tránh phá môi trường torch của Colab), rồi bù các package còn thiếu.
#     Đường chạy YOLO (inference) của ultralytics chỉ cần ở top-level:
#     PIL, cv2, numpy, torch, typing_extensions — Colab đã có hết.
#     thop/polars/nvidia-ml-py/cloudpickle/matplotlib chỉ dùng ở các nhánh
#     train/val/export, nhưng vẫn cài để code path nào import muộn cũng không vỡ.
log('pip: ultralytics (--no-deps) + dependency còn thiếu')
pip('--no-deps', 'ultralytics>=8.3.162')
pip('typing_extensions', 'ultralytics-thop', 'polars', 'nvidia-ml-py', 'cloudpickle')

# ---------- symlink thư mục model ----------
pairs = [
    (f'{COMFY}/models/unet',            P['gguf']),
    (f'{COMFY}/models/diffusion_models', P['gguf']),
    (f'{COMFY}/models/clip',            P['clip']),
    (f'{COMFY}/models/text_encoders',   P['clip']),
    (f'{COMFY}/models/vae',             P['vae']),
    (f'{COMFY}/models/ultralytics',     os.path.dirname(P['yolo'])),
    (f'{COMFY}/models/sams',            P['sam']),
    (f'{COMFY}/models/upscale_models',  P['upscale']),
]
log('Symlink models/')
for path, dest in pairs:
    os.makedirs(dest, exist_ok=True)
    if os.path.islink(path) or os.path.exists(path):
        subprocess.run(['rm', '-rf', path])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.symlink(dest, path)
    log(f'   {path} → {dest}')

# ---------- KIỂM TRA (bản cũ không có bước này nên lỗi âm thầm) ----------
log('Kiểm tra import các package node cần:')
missing = []
for mod, why in [('gguf', 'ComfyUI-GGUF: UnetLoaderGGUF/DualCLIPLoaderGGUF'),
                 ('sentencepiece', 'tokenizer T5 của GGUF'),
                 ('skimage', 'Impact Pack (FaceDetailer) — thiếu là pack raise khi import'),
                 ('piexif', 'Impact Pack'),
                 ('dill', 'Impact Pack'),
                 ('segment_anything', 'SAMLoader'),
                 ('matplotlib', 'Impact Subpack (UltralyticsDetectorProvider)'),
                 ('ultralytics', 'Impact Subpack (YOLO)')]:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', '?')
        log(f'   ✅ {mod} {ver}  ({why})')
    except Exception as e:
        missing.append(f'{mod} ({why}): {e}')
        log(f'   ❌ {mod}: {e}')

if missing:
    raise RuntimeError('❌ Thiếu dependency, ComfyUI sẽ thiếu node:\n  - ' + '\n  - '.join(missing))

import gguf as _gguf
if tuple(int(x) for x in str(_gguf.__version__).split('.')[:2]) < (0, 13):
    log(f'⚠️ gguf {_gguf.__version__} < 0.13 — ComfyUI-GGUF yêu cầu >=0.13')
log('✅ Xong Cell 2 → chạy Cell 3')
''')

# =========================================================================== CELL 3
code(r'''
# @title ⬇️ CELL 3 — Tải model (~12 GB, có mirror dự phòng)
BO_QUA_MODEL = False  # @param {type:"boolean"}
TAI_TAESD = True  # @param {type:"boolean"}
SO_LUONG_TAI_SONG_SONG = 3  # @param {type:"integer"}

import os, sys, json, time, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
    sys.stdout.flush()

P = json.load(open('/content/mode_ai_paths.json'))

def ok(path, minb):
    return os.path.isfile(path) and os.path.getsize(path) >= minb

def curl(url, path, minb):
    tmp = path + '.part'
    cmd = ['curl', '-L', '--fail', '--retry', '5', '--retry-delay', '2', '--retry-all-errors',
           '-C', '-', '--connect-timeout', '30', '-A', 'Mozilla/5.0', '-o', tmp, url]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not os.path.isfile(tmp):
        return False
    if os.path.getsize(tmp) < minb:
        os.remove(tmp)
        return False
    shutil.move(tmp, path)
    return True

def hf(repo, filename, path, minb):
    """Tải qua huggingface_hub. KHÔNG dùng resume_download (đã bị bỏ từ hub 1.x → TypeError)."""
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=repo, filename=filename,
                            local_dir=os.path.dirname(path) or '.')
    if not os.path.isfile(local) or os.path.getsize(local) < minb:
        return False
    dest = os.path.join(os.path.dirname(path), os.path.basename(local))
    if os.path.abspath(dest) != os.path.abspath(local):
        shutil.move(local, dest)
    return True

def get(label, path, minb, sources, required=True):
    if ok(path, minb):
        log(f'✅ {label} đã có ({os.path.getsize(path)/1e6:.0f} MB)')
        return True
    log(f'⬇️  {label}')
    last = None
    for kind, *rest in sources:
        try:
            good = curl(rest[0], path, minb) if kind == 'curl' else hf(rest[0], rest[1], path, minb)
            if good:
                log(f'✅ {label} ({os.path.getsize(path)/1e6:.0f} MB)')
                return True
        except Exception as e:
            last = str(e).splitlines()[0][:160]
            log(f'   mirror lỗi: {last}')
    msg = f'❌ Không tải được {label}' + (f' — {last}' if last else '')
    if required:
        raise RuntimeError(msg + '\n   Thử tick USE_HF_MIRROR ở Cell 1 rồi chạy lại.')
    log(msg + ' — bỏ qua')
    return False

def hf_url(repo, fn):
    return f'https://huggingface.co/{repo}/resolve/main/{fn}?download=true'

def mirror_url(repo, fn):
    return f'https://hf-mirror.com/{repo}/resolve/main/{fn}?download=true'

TASKS = [
    ('FLUX UNET Q5_K_S (8.3 GB)', f"{P['gguf']}/flux1-schnell-Q5_K_S.gguf", 7_500_000_000,
     [('curl', hf_url('city96/FLUX.1-schnell-gguf', 'flux1-schnell-Q5_K_S.gguf')),
      ('curl', mirror_url('city96/FLUX.1-schnell-gguf', 'flux1-schnell-Q5_K_S.gguf')),
      ('hf', 'city96/FLUX.1-schnell-gguf', 'flux1-schnell-Q5_K_S.gguf')], True),
    ('T5-XXL Q4_K_M (2.9 GB)', f"{P['clip']}/t5-v1_1-xxl-encoder-Q4_K_M.gguf", 2_500_000_000,
     [('curl', hf_url('city96/t5-v1_1-xxl-encoder-gguf', 't5-v1_1-xxl-encoder-Q4_K_M.gguf')),
      ('curl', mirror_url('city96/t5-v1_1-xxl-encoder-gguf', 't5-v1_1-xxl-encoder-Q4_K_M.gguf')),
      ('hf', 'city96/t5-v1_1-xxl-encoder-gguf', 't5-v1_1-xxl-encoder-Q4_K_M.gguf')], True),
    ('CLIP-L (246 MB)', f"{P['clip']}/clip_l.safetensors", 200_000_000,
     [('curl', hf_url('comfyanonymous/flux_text_encoders', 'clip_l.safetensors')),
      ('curl', mirror_url('comfyanonymous/flux_text_encoders', 'clip_l.safetensors')),
      ('hf', 'comfyanonymous/flux_text_encoders', 'clip_l.safetensors')], True),
    ('FLUX VAE ae (335 MB)', f"{P['vae']}/ae.safetensors", 250_000_000,
     [('curl', hf_url('Comfy-Org/Lumina_Image_2.0_Repackaged', 'split_files/vae/ae.safetensors')),
      ('curl', mirror_url('Comfy-Org/Lumina_Image_2.0_Repackaged', 'split_files/vae/ae.safetensors')),
      ('curl', hf_url('camenduru/FLUX.1-dev', 'ae.safetensors')),
      ('hf', 'camenduru/FLUX.1-dev', 'ae.safetensors')], True),
    ('YOLO mặt face_yolov8m (52 MB)', f"{P['yolo']}/face_yolov8m.pt", 40_000_000,
     [('curl', hf_url('Bingsu/adetailer', 'face_yolov8m.pt')),
      ('curl', mirror_url('Bingsu/adetailer', 'face_yolov8m.pt')),
      ('hf', 'Bingsu/adetailer', 'face_yolov8m.pt')], True),
    ('YOLO tay hand_yolov8s (22 MB)', f"{P['yolo']}/hand_yolov8s.pt", 15_000_000,
     [('curl', hf_url('Bingsu/adetailer', 'hand_yolov8s.pt')),
      ('curl', mirror_url('Bingsu/adetailer', 'hand_yolov8s.pt')),
      ('hf', 'Bingsu/adetailer', 'hand_yolov8s.pt')], True),
    ('SAM ViT-B (375 MB)', f"{P['sam']}/sam_vit_b_01ec64.pth", 300_000_000,
     [('curl', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'),
      ('curl', hf_url('segments-ai/sam_vit_b', 'sam_vit_b_01ec64.pth')),
      ('hf', 'segments-ai/sam_vit_b', 'sam_vit_b_01ec64.pth')], True),
]
if TAI_TAESD:
    # ComfyUI tìm file BẮT ĐẦU bằng tên trong latent_format.taesd_decoder_name
    # (FLUX.1 = "taef1_decoder", xem comfy/latent_formats.py). Repo HF `madebyollin/taesd`
    # KHÔNG có file taef1 → lấy từ GitHub hoặc repo mirror.
    TASKS.append(('TAESD preview FLUX (2.5 MB, xem trước nét khi đang sample)',
                  '/content/ComfyUI/models/vae_approx/taef1_decoder.pth', 1_500_000,
                  [('curl', 'https://github.com/madebyollin/taesd/raw/main/taef1_decoder.pth'),
                   ('curl', hf_url('UmeAiRT/ComfyUI-Auto-Installer-Assets',
                                   'models/vae_approx/taef1_decoder.safetensors')),
                   ('hf', 'UmeAiRT/ComfyUI-Auto-Installer-Assets',
                    'models/vae_approx/taef1_decoder.safetensors')], False))

os.makedirs('/content/ComfyUI/models/vae_approx', exist_ok=True)
t0 = time.time()
if not BO_QUA_MODEL:
    with ThreadPoolExecutor(max_workers=max(1, int(SO_LUONG_TAI_SONG_SONG))) as ex:
        results = list(ex.map(lambda t: get(*t), TASKS))
    if not all(results):
        raise RuntimeError('❌ Có model bắt buộc chưa tải được — xem log phía trên')
else:
    log('BO_QUA_MODEL=True — chỉ kiểm tra file đã có')
    for label, path, minb, _s, req in TASKS:
        log(('✅ ' if ok(path, minb) else ('❌ ' if req else '⚠️ ')) + label
            + (f' ({os.path.getsize(path)/1e6:.0f} MB)' if os.path.isfile(path) else ' — KHÔNG CÓ'))

total = 0
for label, path, minb, _s, _r in TASKS:
    if os.path.isfile(path):
        total += os.path.getsize(path)
log(f'🟢 Tổng model: {total/1e9:.2f} GB — tải trong {time.time()-t0:.0f}s')
log('✅ Xong Cell 3 → chạy Cell 4')
''')

# =========================================================================== CELL 4 (nhúng builder)
code(r'''
# @title 🧠 CELL 4 — Ghi workflow tối ưu vào máy (nhúng sẵn, không phụ thuộc repo ngoài)
WORKFLOW_DIR = "/content/workflows"  # @param {type:"string"}
TAI_BAN_UI_TU_REPO = True  # @param {type:"boolean"}

import os, json, subprocess

# ==== BEGIN BUILD_WORKFLOWS (nhúng từ scripts/build_workflows.py) ====
__BUILDER__
# ==== END BUILD_WORKFLOWS ====

os.makedirs(WORKFLOW_DIR, exist_ok=True)
COMFY = '/content/ComfyUI'
written = []
for name, wf in build_all().items():
    for dest_dir in (WORKFLOW_DIR, f'{COMFY}/input'):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f'{name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(wf, f, ensure_ascii=False, indent=1)
        written.append(path)
    print(f'✅ {name}.json ({len(wf)} node)')

# bản UI (có layout, kéo-thả vào giao diện) — tải từ repo, không có thì bỏ qua
if TAI_BAN_UI_TU_REPO:
    REPO = 'manhlee1196-boop/mode-ai'
    BRANCHES = ['main', 'arena/01a0d3b5-mode-ai']
    ui_dir = f'{COMFY}/user/default/workflows'
    os.makedirs(ui_dir, exist_ok=True)
    for br in BRANCHES:
        got = 0
        for name in build_all():
            url = f'https://raw.githubusercontent.com/{REPO}/{br}/workflows/ui/{name}.json'
            dest = os.path.join(ui_dir, f'{name}.json')
            r = subprocess.run(['curl', '-sfL', '--max-time', '20', '-o', dest, url])
            if r.returncode == 0 and os.path.getsize(dest) > 200:
                got += 1
            elif os.path.isfile(dest):
                os.remove(dest)
        if got:
            print(f'✅ {got} workflow bản UI (layout) từ branch {br} → {ui_dir}')
            break
    else:
        print('ℹ️ Không tải được bản UI từ repo — dùng bản API (giao diện ComfyUI mới '
              'vẫn mở được, chỉ không có layout) hoặc dùng Cell 6.')

print('\nCách dùng:')
print('  • Cell 6: tạo ảnh ngay trong Colab, không cần mở giao diện')
print('  • Hoặc mở link ComfyUI → Workflow → Open → chọn file trong /content/workflows')
print('✅ Xong Cell 4 → chạy Cell 5')
''')

# =========================================================================== CELL 5
code(r'''
# @title 🚀 CELL 5 — Khởi chạy ComfyUI + tunnel
PORT = 8188  # @param {type:"integer"}
TUNNEL = "cloudflared http2"  # @param ["cloudflared http2", "cloudflared quic", "không tunnel"]
VRAM_MODE = "Mặc định — Dynamic VRAM (khuyến nghị)"  # @param ["Mặc định — Dynamic VRAM (khuyến nghị)", "lowvram", "normalvram", "highvram", "novram", "cpu"]
RESERVE_VRAM_GB = 1.0  # @param {type:"slider", min:0.0, max:4.0, step:0.1}
FORCE_FP16 = True  # @param {type:"boolean"}
VAE_PREC = "fp16-vae"  # @param ["fp16-vae", "fp32-vae", "cpu-vae", "Mặc định"]
ATTENTION = "pytorch (SDPA)"  # @param ["pytorch (SDPA)", "sage", "flash", "Mặc định"]
PREVIEW = "taesd"  # @param ["taesd", "auto", "latent2rgb", "none"]
CACHE_LRU = 0  # @param {type:"integer"}
FAST_FP16_ACCUM = False  # @param {type:"boolean"}
EXTRA_ARGS = ""  # @param {type:"string"}

import os, re, sys, json, time, socket, shutil, subprocess

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
    sys.stdout.flush()

COMFY = '/content/ComfyUI'
P = json.load(open('/content/mode_ai_paths.json'))
assert os.path.isfile(f'{COMFY}/main.py'), '❌ Chạy Cell 1 trước'

log('Dừng tiến trình cũ')
os.system('pkill -f "python.*main.py" >/dev/null 2>&1 || true')
os.system('pkill -f cloudflared >/dev/null 2>&1 || true')
time.sleep(2)

cmd = [sys.executable, 'main.py', '--listen', '0.0.0.0', '--port', str(int(PORT)),
       '--preview-method', PREVIEW,
       '--output-directory', f'{COMFY}/output',
       '--input-directory', f'{COMFY}/input',
       '--temp-directory', f'{COMFY}/temp',
       '--cuda-device', '0',
       '--enable-cors-header', '*',
       '--disable-xformers']

# ComfyUI >= 0.3x bật Dynamic VRAM mặc định cho NVIDIA; --lowvram lúc đó bị bỏ qua.
# Chỉ thêm cờ vram khi người dùng explicitly chọn.
vram_map = {'lowvram': '--lowvram', 'normalvram': '--normalvram', 'highvram': '--highvram',
            'novram': '--novram', 'cpu': '--cpu'}
for k, flag in vram_map.items():
    if VRAM_MODE.startswith(k):
        cmd.append(flag)
if RESERVE_VRAM_GB and RESERVE_VRAM_GB > 0 and not VRAM_MODE.startswith('cpu'):
    cmd += ['--reserve-vram', str(float(RESERVE_VRAM_GB))]
if FORCE_FP16:
    cmd.append('--force-fp16')
if VAE_PREC in ('fp16-vae', 'fp32-vae', 'cpu-vae'):
    cmd.append('--' + VAE_PREC)
att_map = {'pytorch (SDPA)': '--use-pytorch-cross-attention',
           'sage': '--use-sage-attention', 'flash': '--use-flash-attention'}
if ATTENTION in att_map:
    cmd.append(att_map[ATTENTION])
if int(CACHE_LRU) > 0:
    cmd += ['--cache-lru', str(int(CACHE_LRU))]
if FAST_FP16_ACCUM:
    cmd += ['--fast', 'fp16_accumulation']
if EXTRA_ARGS.strip():
    cmd += EXTRA_ARGS.strip().split()

log('Lệnh: ' + ' '.join(cmd))
logf = open('/content/comfyui.log', 'w')
proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=COMFY)
open('/content/comfy.pid', 'w').write(str(proc.pid))

def port_open():
    try:
        with socket.create_connection(('127.0.0.1', int(PORT)), timeout=1):
            return True
    except OSError:
        return False

ok = False
for i in range(240):
    time.sleep(1)
    if proc.poll() is not None:
        os.system('tail -40 /content/comfyui.log')
        raise RuntimeError('❌ ComfyUI thoát khi khởi động — xem log ở trên')
    if port_open():
        ok = True
        break
    if i % 20 == 19:
        log(f'  ...đợi {i+1}s')
        os.system('tail -2 /content/comfyui.log')
if not ok:
    os.system('tail -40 /content/comfyui.log')
    raise RuntimeError('❌ Quá 240s chưa mở cổng')
log(f'✅ ComfyUI đang chạy cổng {PORT}')

url = None
if TUNNEL.startswith('cloudflared'):
    if not shutil.which('cloudflared'):
        log('Cài cloudflared')
        os.system('wget -q -c https://github.com/cloudflare/cloudflared/releases/latest/download/'
                  'cloudflared-linux-amd64.deb -O /tmp/cloudflared.deb')
        os.system('dpkg -i /tmp/cloudflared.deb >/dev/null 2>&1')
    proto = 'http2' if 'http2' in TUNNEL else 'quic'
    cff = open('/content/cloudflared.log', 'w')
    cf = subprocess.Popen(['cloudflared', 'tunnel', '--url', f'http://127.0.0.1:{int(PORT)}',
                           '--http-host-header', f'127.0.0.1:{int(PORT)}', '--protocol', proto],
                          stdout=cff, stderr=subprocess.STDOUT)
    for _ in range(90):
        time.sleep(1)
        m = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com',
                      open('/content/cloudflared.log', errors='ignore').read())
        if m:
            url = m.group(0).rstrip('/')
            break
    if url:
        open('/content/comfy_url.txt', 'w').write(url)

print('\n' + '=' * 64)
if url:
    print('🎨 COMFYUI — copy link, DÁN vào tab mới (đừng bấm trong Colab):\n')
    print('   ' + url)
else:
    print(f'⚠️ Chưa có link tunnel. Local: http://127.0.0.1:{int(PORT)} (dùng Cell 5 localtunnel nếu cần)')
print('=' * 64)
print('Giờ chạy Cell 6 để tạo ảnh ngay trong Colab, hoặc Cell 8 để chẩn đoán node.')
''')

# =========================================================================== CELL 6
code(r'''
# @title 🖼 CELL 6 — Tạo ảnh ngay trong Colab (headless, không cần mở giao diện)
PIPELINE = "flux_q5_standard"  # @param ["flux_q5_fast", "flux_q5_standard", "flux_q5_quality", "flux_q5_hires", "flux_q5_inpaint"]
PROMPT = "photorealistic portrait of a young Vietnamese woman, natural skin texture with visible pores, soft window light, 85mm lens, shallow depth of field, detailed eyes and hands, five fingers, casual linen shirt, warm neutral background, film grain, high detail"  # @param {type:"string"}
WIDTH = 1024  # @param {type:"integer"}
HEIGHT = 1024  # @param {type:"integer"}
SEED = -1  # @param {type:"integer"}
SO_ANH = 1  # @param {type:"integer"}
LUU_VAO_DRIVE = False  # @param {type:"boolean"}

import os, json, time, random, requests
from PIL import Image
from IPython.display import display

COMFY = 'http://127.0.0.1:8188'
WORKFLOW_DIR = '/content/workflows'
OUT = '/content/ComfyUI/output'
IN_DIR = '/content/ComfyUI/input'

def _health():
    try:
        return requests.get(f'{COMFY}/system_stats', timeout=5).status_code == 200
    except Exception:
        return False

if not _health():
    raise RuntimeError('❌ ComfyUI chưa chạy — chạy Cell 5 trước')

def _nodes(wf, class_type):
    return [k for k, v in wf.items() if v.get('class_type') == class_type]

def prepare(pipeline, prompt, width, height, seed, negative=''):
    path = os.path.join(WORKFLOW_DIR, f'{pipeline}.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} không có — chạy Cell 4 trước')
    wf = json.load(open(path, encoding='utf-8'))

    # inpaint đọc ảnh có sẵn trong ComfyUI/input, không tự tạo được từ prompt
    if any(v.get('class_type') == 'LoadImage' for v in wf.values()):
        for nid in _nodes(wf, 'LoadImage'):
            name = wf[nid]['inputs'].get('image', '')
            if not os.path.isfile(os.path.join(IN_DIR, name)):
                raise FileNotFoundError(
                    f'❌ {pipeline} cần file {IN_DIR}/{name} — hãy upload ảnh + mask '
                    f'vào đó, hoặc dùng Cell 7 (vẽ mask bằng chuột, tự upload).')

    # prompt dương = node CLIPTextEncode có text dài nhất (node "5" của builder)
    pos_ids = sorted(_nodes(wf, 'CLIPTextEncode'),
                     key=lambda k: -len(str(wf[k]['inputs'].get('text', ''))))
    if pos_ids:
        wf[pos_ids[0]]['inputs']['text'] = prompt
    if negative:
        for nid in _nodes(wf, 'CLIPTextEncode'):
            if nid != pos_ids[0]:
                wf[nid]['inputs']['text'] = negative

    for nid in _nodes(wf, 'EmptyLatentImage'):
        wf[nid]['inputs']['width'] = int(width)
        wf[nid]['inputs']['height'] = int(height)

    for i, nid in enumerate(sorted(_nodes(wf, 'KSampler'), key=int)):
        wf[nid]['inputs']['seed'] = seed + i
    for i, nid in enumerate(sorted(_nodes(wf, 'FaceDetailer'), key=int)):
        wf[nid]['inputs']['seed'] = seed + 100 + i
    return wf

def run(wf, timeout=600):
    r = requests.post(f'{COMFY}/prompt', json={'prompt': wf}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f'ComfyUI từ chối prompt (HTTP {r.status_code}):\n'
                           f'{json.dumps(r.json(), ensure_ascii=False)[:1500]}')
    pid = r.json()['prompt_id']
    print(f'📤 prompt_id={pid}')
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        try:
            h = requests.get(f'{COMFY}/history/{pid}', timeout=10).json()
        except Exception:
            continue
        if pid in h:
            st = h[pid].get('status', {})
            if st.get('status_str') == 'error':
                raise RuntimeError(f'ComfyUI báo lỗi khi chạy: {json.dumps(st, ensure_ascii=False)[:1200]}')
            files = []
            for node_out in h[pid].get('outputs', {}).values():
                for im in node_out.get('images', []):
                    if im.get('type') != 'output':
                        continue
                    files.append(os.path.join(OUT, im.get('subfolder', ''), im['filename']))
            return [f for f in files if os.path.isfile(f)]
    raise TimeoutError(f'Quá {timeout}s chưa xong — xem /content/comfyui.log')

def generate(prompt=PROMPT, pipeline=PIPELINE, width=WIDTH, height=HEIGHT, seed=SEED,
             n=SO_ANH, negative='', show=True):
    """Tạo n ảnh, trả về danh sách đường dẫn file."""
    results = []
    for i in range(int(n)):
        s = int(seed) if int(seed) >= 0 else random.randint(0, 2**31 - 1)
        s = s + i if int(seed) >= 0 else s
        t0 = time.time()
        wf = prepare(pipeline, prompt, width, height, s, negative)
        files = run(wf)
        for f in files:
            results.append(f)
            if show:
                display(Image.open(f))
        print(f'  ảnh {i+1}/{n}: {len(files)} file, {time.time()-t0:.1f}s, seed={s}')
    if LUU_VAO_DRIVE and os.path.isdir('/content/drive/MyDrive'):
        import shutil
        dest = '/content/drive/MyDrive/FLUX_output'
        os.makedirs(dest, exist_ok=True)
        for f in results:
            shutil.copy2(f, dest)
        print(f'💾 Đã copy {len(results)} ảnh vào {dest}')
    return results

anh = generate()
print(f'\n✅ {len(anh)} ảnh trong {OUT}')
''')

# =========================================================================== CELL 7
code(r'''
# @title 🖌 CELL 7 — Inpaint vẽ tay (Gradio, dùng flux_q5_inpaint)
DENOISE = 0.5  # @param {type:"slider", min:0.2, max:0.85, step:0.05}
STEPS = 6  # @param {type:"integer"}
GROW_MASK = 12  # @param {type:"integer"}

import os, sys, json, time, uuid, random, subprocess, requests
import numpy as np
from PIL import Image

try:
    import gradio as gr
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'])
    import gradio as gr

sys.path.insert(0, '/content/workflows')
COMFY = 'http://127.0.0.1:8188'
OUT = '/content/ComfyUI/output'
WF_INPAINT = '/content/workflows/flux_q5_inpaint.json'

def latest_output():
    if not os.path.isdir(OUT):
        return None
    fs = [os.path.join(OUT, f) for f in os.listdir(OUT)
          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    return max(fs, key=os.path.getmtime) if fs else None

def split_bg_mask(value):
    """Tách (ảnh nền, mask) từ ImageEditor/Sketchpad/ImageMask của Gradio 3/4/5."""
    def to_img(x, mode):
        if isinstance(x, dict) and 'name' in x:
            return Image.open(x['name']).convert(mode)
        if isinstance(x, str) and os.path.isfile(x):
            return Image.open(x).convert(mode)
        if hasattr(x, 'convert'):
            return x.convert(mode)
        return Image.fromarray(np.array(x)).convert(mode)

    if isinstance(value, dict):
        if 'background' in value:                      # Gradio 5 ImageEditor
            bg = to_img(value['background'], 'RGB')
            layers = value.get('layers') or []
            if not layers:
                return bg, None
            alpha = to_img(layers[0], 'RGBA').split()[-1]
            return bg, alpha
        if 'image' in value and 'mask' in value:       # Gradio 3/4
            return to_img(value['image'], 'RGB'), to_img(value['mask'], 'L')
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return to_img(value[0], 'RGB'), to_img(value[1], 'L')
    if value is not None and hasattr(value, 'convert'):
        return value.convert('RGB'), None
    return None, None

def upload(path):
    with open(path, 'rb') as f:
        r = requests.post(f'{COMFY}/upload/image', files={'image': f},
                          data={'overwrite': 'true', 'type': 'input', 'subfolder': ''},
                          timeout=60)
    r.raise_for_status()
    return r.json()['name']

def do_inpaint(editor, prompt, denoise, steps, grow, seed):
    if editor is None:
        latest = latest_output()
        if not latest:
            return None, '❌ Chưa có ảnh: upload ảnh hoặc chạy Cell 6 trước'
        bg, mask = Image.open(latest).convert('RGB'), None
    else:
        bg, mask = split_bg_mask(editor)
    if bg is None:
        return None, '❌ Không đọc được ảnh'
    if mask is None or float(np.mean(np.array(mask) > 128)) < 0.005:
        return None, '❌ Chưa tô mask — dùng cọ tô lên vùng cần sửa'
    if mask.size != bg.size:
        mask = mask.resize(bg.size)

    rid = uuid.uuid4().hex[:8]
    img_p, msk_p = f'/tmp/inp_{rid}.png', f'/tmp/msk_{rid}.png'
    bg.save(img_p)
    mask.convert('RGB').save(msk_p)
    img_name, msk_name = upload(img_p), upload(msk_p)

    wf = json.load(open(WF_INPAINT, encoding='utf-8'))
    wf['4']['inputs']['image'] = img_name
    wf['5m']['inputs']['image'] = msk_name
    wf['9e']['inputs']['grow_mask_by'] = int(grow)
    wf['7']['inputs']['denoise'] = float(denoise)
    wf['7']['inputs']['steps'] = int(steps)
    wf['7']['inputs']['seed'] = int(seed) if int(seed) >= 0 else random.randint(0, 2**31 - 1)
    if prompt.strip():
        wf['5']['inputs']['text'] = prompt.strip()

    r = requests.post(f'{COMFY}/prompt', json={'prompt': wf}, timeout=30)
    if r.status_code != 200:
        return None, f'❌ HTTP {r.status_code}: {r.text[:600]}'
    pid = r.json()['prompt_id']

    for i in range(300):
        time.sleep(2)
        try:
            h = requests.get(f'{COMFY}/history/{pid}', timeout=10).json()
        except Exception:
            continue
        if pid in h:
            for node_out in h[pid].get('outputs', {}).values():
                for im in node_out.get('images', []):
                    if im.get('type') != 'output':
                        continue
                    p = os.path.join(OUT, im.get('subfolder', ''), im['filename'])
                    if os.path.isfile(p):
                        return Image.open(p), f'✅ Xong — {os.path.basename(p)}'
            return None, '❌ Chạy xong nhưng không thấy ảnh — xem /content/comfyui.log'
    return None, '❌ Hết thời gian chờ'

default = latest_output()
with gr.Blocks(title='Inpaint FLUX Q5') as demo:
    gr.Markdown('## 🖌 Inpaint FLUX.1-schnell Q5\n'
                '1. Upload ảnh (để trống = lấy ảnh mới nhất trong output)\n'
                '2. **Tô lên vùng lỗi** (tay/mặt/chân) bằng cọ\n'
                '3. Mô tả phần muốn vẽ lại → bấm **Sửa vùng tô** (~20s trên T4)')
    with gr.Row():
        with gr.Column():
            ed = gr.ImageEditor(type='pil', height=620,
                                value={'background': Image.open(default).convert('RGB'),
                                       'layers': [], 'composite': Image.open(default).convert('RGB')}
                                if default else None,
                                brush=gr.Brush(colors=['#FFFFFF'], color_mode='fixed', default_size=40),
                                label='Ảnh gốc — tô lên vùng cần sửa')
            pr = gr.Textbox(label='Mô tả phần vẽ lại (tiếng Anh)',
                            value='detailed human hand, five fingers, natural fingernails, '
                                  'realistic skin texture, photorealistic, sharp focus', lines=2)
            with gr.Row():
                d = gr.Slider(0.2, 0.85, value=DENOISE, step=0.05, label='Denoise')
                st = gr.Slider(4, 12, value=STEPS, step=1, label='Steps')
            with gr.Row():
                g = gr.Slider(0, 32, value=GROW_MASK, step=2, label='Grow mask px')
                sd = gr.Number(value=-1, label='Seed (-1 = random)')
            btn = gr.Button('🖌 Sửa vùng tô', variant='primary')
        with gr.Column():
            out_img = gr.Image(label='Kết quả', height=620, type='pil')
            msg = gr.Markdown('Sẵn sàng.')
    btn.click(do_inpaint, inputs=[ed, pr, d, st, g, sd], outputs=[out_img, msg])

print('Đợi link https://xxxx.gradio.live (bấm Stop để tắt)')
demo.queue().launch(share=True, server_name='0.0.0.0', server_port=7860)
''')

# =========================================================================== CELL 8
code(r'''
# @title 🔎 CELL 8 — Chẩn đoán: node có đủ không, model có đủ không
import os, json, requests

COMFY = 'http://127.0.0.1:8188'
P = json.load(open('/content/mode_ai_paths.json'))

try:
    info = requests.get(f'{COMFY}/object_info', timeout=60).json()
except Exception as e:
    raise RuntimeError(f'❌ Không gọi được /object_info — ComfyUI chưa chạy? ({e})')

print(f'ComfyUI đang phục vụ {len(info)} node class\n')
CAN_CO = ['UnetLoaderGGUF', 'DualCLIPLoaderGGUF', 'VAELoader', 'KSampler', 'EmptyLatentImage',
          'CLIPTextEncode', 'VAEDecode', 'VAEEncode', 'ImageScaleBy', 'SaveImage', 'LoadImage',
          'ImageToMask', 'VAEEncodeForInpaint', 'FaceDetailer', 'SAMLoader',
          'UltralyticsDetectorProvider']
thieu = []
for n in CAN_CO:
    ok = n in info
    print(('  ✅ ' if ok else '  ❌ ') + n)
    if not ok:
        thieu.append(n)

print('\nModel ComfyUI nhìn thấy:')
for label, key in [('UNET GGUF', 'UnetLoaderGGUF'), ('CLIP', 'DualCLIPLoaderGGUF'),
                   ('VAE', 'VAELoader'), ('YOLO', 'UltralyticsDetectorProvider'),
                   ('SAM', 'SAMLoader')]:
    if key not in info:
        print(f'  {label}: (node thiếu)')
        continue
    node = info[key]
    seen = set()
    for field in ('unet_name', 'clip_name1', 'clip_name2', 'vae_name', 'model_name'):
        req = node.get('input', {}).get('required', {}).get(field)
        if isinstance(req, list) and req and isinstance(req[0], list):
            seen.update(req[0])
    print(f'  {label}: {sorted(seen) if seen else "(trống!)"}')

if 'UltralyticsDetectorProvider' in info:
    yolo = info['UltralyticsDetectorProvider']['input']['required']['model_name'][0]
    bad = [x for x in yolo if not (x.startswith('bbox/') or x.startswith('segm/'))]
    if yolo and bad == yolo:
        print('  ⚠️ YOLO không có tiền tố bbox/ — kiểm tra symlink models/ultralytics (Cell 2)')

print('\nVRAM:')
os.system('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader')
if thieu:
    print('\n❌ THIẾU NODE: ' + ', '.join(thieu))
    print('   → GGUF thiếu: pip install "gguf>=0.13" sentencepiece protobuf')
    print('   → FaceDetailer/SAMLoader thiếu: pip install scikit-image piexif dill segment-anything')
    print('   → UltralyticsDetectorProvider thiếu: pip install ultralytics matplotlib')
    print('   Sau đó chạy lại Cell 2 rồi khởi động lại ComfyUI (Cell 5).')
else:
    print('\n✅ Đủ node cho cả 5 workflow')
print('\nLog ComfyUI (30 dòng cuối):')
os.system('tail -30 /content/comfyui.log')
''')

# =========================================================================== CELL 9
code(r'''
# @title 🌐 CELL 9 — Tunnel dự phòng (localtunnel) nếu cloudflared không chạy
import urllib.request, subprocess

subprocess.run(['npm', 'install', '-g', 'localtunnel'],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ip = urllib.request.urlopen('https://ipv4.icanhazip.com', timeout=20).read().decode().strip()
print('🔑 Tunnel Password (dán khi trang hỏi):', ip)
subprocess.run(['lt', '--port', '8188'])
''')

# =========================================================================== MD cuối
md("""
## ⚙️ Tham số tối ưu (đã đặt sẵn trong workflow)

| Nơi | Tham số | Giá trị | Vì sao |
|---|---|---|---|
| KSampler chính | steps | **4** | FLUX.1-schnell là model chưng cất 4 bước; thêm bước chỉ tốn thời gian |
| KSampler chính | cfg | **1.0** | schnell không dùng CFG → ComfyUI bỏ luôn nhánh negative (nhanh hơn) |
| KSampler chính | sampler / scheduler | `euler` / `simple` | bộ đôi ổn định nhất cho schnell |
| KSampler chính | size | 1024×1024 (hoặc 832×1216) | giữ tổng ~1 MP, bội số của 16 |
| FaceDetailer mặt | denoise / steps | 0.22 / 4 | đủ sửa mắt-miệng mà không đổi identity |
| FaceDetailer mặt | guide_size / max_size | 384 / 768 | crop nhỏ → nhanh, ít VRAM hơn 512/1024 |
| FaceDetailer mặt | SAM | `sam_vit_b` + threshold 0.93 | mask mặt sát, không lem |
| FaceDetailer tay | denoise / feather | 0.28 / 16 | không dùng SAM (tiết kiệm VRAM), feather lớn để blend |
| Hires | upscale 1.5× → denoise 0.35, 4 bước | — | không cần ESRGAN/UltimateSDUpscale, chỉ dùng node có sẵn |
| Inpaint | denoise 0.5, steps 6, grow_mask 12px | — | giữ context quanh vùng tô |
| ComfyUI | `--reserve-vram 1.0`, `--force-fp16`, `--fp16-vae`, SDPA | — | hợp với T4 16 GB (không dùng `--lowvram` vì ComfyUI mới đã có Dynamic VRAM) |

**Nếu OOM:** giảm `WIDTH/HEIGHT` về 832×832 · đặt `VAE_PREC=cpu-vae` · `PREVIEW=none` ·
`VRAM_MODE=lowvram` · tắt Cell 7 (Gradio) khi không dùng.

## 💡 Prompt cho FLUX

FLUX hiểu câu mô tả tự nhiên tốt hơn "tag soup". Viết như mô tả ảnh cho nhiếp ảnh gia:
chủ thể → trang phục/bối cảnh → ánh sáng → ống kính → chi tiết cần giữ.

```
photorealistic portrait of a young Vietnamese woman, natural skin texture with visible pores,
soft window light, 85mm lens, shallow depth of field, detailed eyes and hands, five fingers,
casual linen shirt, warm neutral background, film grain, high detail
```

Negative để trống: với cfg = 1.0, negative không được dùng — để trống còn giúp T5 encode nhanh hơn.

## 🧪 Tự kiểm tra (không cần GPU)

```bash
python3 scripts/node_spec.py --comfy /tmp/ComfyUI --gguf /tmp/ComfyUI-GGUF \
        --impact /tmp/Impact-Pack --subpack /tmp/Impact-Subpack -o workflows/node_spec.json
python3 scripts/build_workflows.py          # sinh lại workflows/*.json
python3 scripts/api_to_ui.py workflows/flux_q5_standard.json   # bản UI có layout
python3 scripts/validate_workflows.py       # đối chiếu với INPUT_TYPES thật
```

`validate_workflows.py` đối chiếu từng node với `INPUT_TYPES`/`RETURN_TYPES` trích trực tiếp từ
mã nguồn ComfyUI + ComfyUI-GGUF + Impact Pack/Subpack: thiếu input required, sai enum, link
đứt, sai kiểu dữ liệu, chu trình, sai tiền tố `bbox/` — tất cả đều bị bắt trước khi lên Colab.
""")


def build_notebook() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def _bound_names(tree: ast.AST) -> set[str]:
    """Mọi tên cell này gán/định nghĩa/khởi tạo (ở bất kỳ độ sâu nào)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update((a.asname or a.name.split(".")[0]) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update((a.asname or a.name) for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg):
                if arg:
                    names.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    names.add(item.optional_vars.id)
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _loaded_names(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def check_undefined_names(nb: dict) -> None:
    """Bắt tên dùng mà chưa được định nghĩa.

    Colab chạy các cell theo thứ tự và CHIA SẺ namespace, nên tập tên được gán
    được cộng dồn qua các cell — đúng với thực tế khi chạy.
    """
    import builtins
    known = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "_"}
    problems: list[str] = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        tree = ast.parse(src)
        # Colab: dùng tên ở ĐÂU cũng được miễn cell này hoặc cell trước đã gán.
        # (Hàm dùng biến gán ở cuối cell là hợp lệ — Python chỉ resolve khi gọi.)
        known |= _bound_names(tree)
        unknown = sorted(_loaded_names(tree) - known)
        if unknown:
            title = src.splitlines()[0][:58]
            problems.append(f"  cell {i} ({title}): {', '.join(unknown)}")
    if problems:
        raise AssertionError(
            "❌ Có tên dùng mà chưa định nghĩa (notebook sẽ crash khi chạy):\n"
            + "\n".join(problems))
    print("✅ không có tên dùng mà chưa định nghĩa")


def self_check(nb: dict, builder_src: str) -> None:
    """Kiểm tra notebook trước khi ghi ra đĩa."""
    n_code = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(cell["source"])
        ast.parse(src)  # lỗi cú pháp → raise
        if "__BUILDER__" in src:
            raise AssertionError("Cell 4 chưa được nhúng builder")
    print(f"✅ {n_code} code cell parse OK (ast)")
    check_undefined_names(nb)

    # builder nhúng trong notebook phải giống hệt file gốc
    nb_src = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        if "BEGIN BUILD_WORKFLOWS" in s:
            nb_src = s.split("# ==== BEGIN BUILD_WORKFLOWS")[1].split("# ==== END BUILD_WORKFLOWS")[0]
            nb_src = "\n".join(nb_src.splitlines()[1:]) + "\n"
    assert nb_src is not None, "không tìm thấy khối builder trong notebook"
    assert nb_src.strip() == builder_src.strip(), "builder trong notebook KHÁC scripts/build_workflows.py"
    print("✅ builder trong notebook giống hệt scripts/build_workflows.py")

    # chạy builder đó và so với workflows/*.json đã validate
    ns: dict = {}
    exec(compile(builder_src, "<notebook-cell4>", "exec"), ns)
    import json as _json
    for name, wf in ns["build_all"]().items():
        path = os.path.join(ROOT, "workflows", f"{name}.json")
        on_disk = _json.load(open(path, encoding="utf-8"))
        assert on_disk == wf, f"{name}: notebook sinh khác workflows/{name}.json"
    print(f"✅ builder trong notebook sinh đúng {len(ns['build_all']())} workflow đã validate")


def main() -> int:
    src = builder_source()
    for i, (kind, text) in enumerate(CELLS):
        if "__BUILDER__" in text:
            CELLS[i] = (kind, text.replace("__BUILDER__", src.rstrip()))
    nb = build_notebook()
    self_check(nb, src)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    size = os.path.getsize(OUT) / 1024
    print(f"✅ Đã ghi {os.path.relpath(OUT, ROOT)} ({size:.0f} KB, {len(nb['cells'])} cell)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
