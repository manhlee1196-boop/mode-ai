#!/usr/bin/env python3
"""Thư viện prompt đã được thiết kế để TRÁNH LỖI — nguồn duy nhất của workflows/prompts.json.

Vì sao cần viết prompt theo cách này (đều có căn cứ, không phải mẹo truyền miệng):

1. **Negative prompt vô dụng trên pipeline này.** `comfy/samplers.py:610`:
   `if math.isclose(cond_scale, 1.0) ... uncond_ = None` — với cfg=1.0 ComfyUI bỏ hẳn
   nhánh negative. Mọi "chống lỗi" phải nằm trong prompt DƯƠNG.

2. **Cách chắc nhất để không lỗi tay: đừng để tay lộ ngón.**
   FLUX.1-schnell là model chưng cất 4 bước, cfg=1.0 — rất ít "lực lái" để sửa giải phẫu.
   Ghi rõ tay đang ở đâu / đang cầm gì thì model không phải tự bịa ngón tay:
     • "hands tucked into pockets"   → không thấy ngón
     • "hands wrapped around a cup"  → ngón có vật bám
     • "hands clasped on her lap"    → khối kín, dễ render
   Tránh: "five fingers", "perfect hands", "detailed hands" — càng nhấn mạnh số ngón,
   model chưng cất càng hay sinh thêm ngón.

3. **FLUX hiểu câu mô tả tự nhiên, không phải tag soup.** Viết theo thứ tự:
   chủ thể → tư thế/tay → trang phục/bối cảnh → ánh sáng → ống kính → khung hình.

4. **Giữ ~1 megapixel.** 832×1216 (dọc) · 1216×832 (ngang) · 1024×1024 (vuông).
   Xa khỏi ~1MP thì schnell bắt đầu sinh lỗi cấu trúc (thừa chi, méo mặt).

    python3 scripts/prompt_presets.py           # ghi workflows/prompts.json
    python3 scripts/prompt_presets.py --print   # chỉ in
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# cfg=1.0 → nhánh negative bị bỏ; để trống cho T5 encode nhanh hơn
NEGATIVE = ""

# Mức rủi ro sinh lỗi giải phẫu, để người dùng chọn biết mà liệu
THAP = "Thấp"
TRUNG_BINH = "Trung bình"
CAO = "Cao"

PRESETS = [
    {
        "id": "chan_dung_can",
        "ten": "Chân dung cận cảnh — ít lỗi nhất",
        "prompt": (
            "Close-up portrait of a young Vietnamese woman, natural skin with visible pores, "
            "soft window light from the left, 85mm lens, shallow depth of field, "
            "head and shoulders framing, plain warm backdrop, subtle film grain"
        ),
        "size": [832, 1216],
        "pipeline": "flux_q5_standard",
        "rui_ro": THAP,
        "ghi_chu": "Không có tay trong khung → gần như không có lỗi giải phẫu.",
    },
    {
        "id": "ban_than_cam_coc",
        "ten": "Bán thân, hai tay cầm cốc",
        "prompt": (
            "Half-body portrait of a young Vietnamese woman sitting in a cafe, "
            "both hands wrapped around a ceramic coffee cup resting on the table, "
            "natural skin texture, soft afternoon window light, 50mm lens, "
            "shallow depth of field, blurred cafe background, film grain"
        ),
        "size": [832, 1216],
        "pipeline": "flux_q5_quality",
        "rui_ro": TRUNG_BINH,
        "ghi_chu": "Tay có vật bám (cốc) → render ổn định hơn tay trôi nổi.",
    },
    {
        "id": "toan_than_tui_quan",
        "ten": "Toàn thân đứng, tay trong túi",
        "prompt": (
            "Full-body photograph of a young Vietnamese woman standing on a quiet street, "
            "hands tucked into the pockets of a beige trench coat, relaxed natural pose, "
            "soft golden hour light, 35mm lens, full figure in frame from head to shoes, "
            "shallow depth of field, film grain"
        ),
        "size": [832, 1216],
        "pipeline": "flux_q5_standard",
        "rui_ro": THAP,
        "ghi_chu": "Tay giấu trong túi → không lộ ngón tay.",
    },
    {
        "id": "toan_than_ngoi",
        "ten": "Toàn thân ngồi, tay đan trên đùi",
        "prompt": (
            "Full-body photograph of a young Vietnamese woman sitting on a wooden bench "
            "in a park, hands clasped together resting on her lap, relaxed posture, "
            "dappled sunlight through leaves, 35mm lens, entire figure in frame, "
            "natural colors, film grain"
        ),
        "size": [832, 1216],
        "pipeline": "flux_q5_standard",
        "rui_ro": THAP,
        "ghi_chu": "Tay đan thành một khối kín → dễ render, ít lỗi ngón.",
    },
    {
        "id": "thoi_trang",
        "ten": "Thời trang, tay xách túi",
        "prompt": (
            "Fashion editorial photograph of a Vietnamese woman in a flowing white dress "
            "standing against a textured concrete wall, one hand holding a small leather "
            "handbag at her side, dramatic side lighting, 85mm lens, full body in frame, "
            "high fashion magazine aesthetic, film grain"
        ),
        "size": [832, 1216],
        "pipeline": "flux_q5_quality",
        "rui_ro": TRUNG_BINH,
        "ghi_chu": "Tay xách túi có điểm tựa; vẫn nên chạy quality để sửa tay.",
    },
    {
        "id": "duong_pho",
        "ten": "Đời thường đường phố, xách túi tote",
        "prompt": (
            "Candid street photograph of a young Vietnamese woman walking through a "
            "Hanoi old quarter street, carrying a canvas tote bag in her right hand, "
            "natural walking pose, overcast soft light, 35mm lens, full body in frame, "
            "documentary photography style, film grain"
        ),
        "size": [832, 1216],
        "pipeline": "flux_q5_quality",
        "rui_ro": TRUNG_BINH,
        "ghi_chu": "Đi bộ + túi xách: tay có việc để làm, ít sinh ngón thừa.",
    },
    {
        "id": "phong_canh",
        "ten": "Phong cảnh (không có người)",
        "prompt": (
            "Wide landscape photograph of terraced rice fields in northern Vietnam at "
            "sunrise, mist drifting between the hills, warm golden light, 24mm wide angle, "
            "deep depth of field, high detail, no people"
        ),
        "size": [1216, 832],
        "pipeline": "flux_q5_fast",
        "rui_ro": THAP,
        "ghi_chu": "Không có người → không có lỗi giải phẫu. 'no people' là cụm rất hiệu lực.",
    },
    {
        "id": "san_pham",
        "ten": "Sản phẩm studio",
        "prompt": (
            "Studio product photograph of a matte black ceramic coffee mug on a light oak "
            "table, soft diffused lighting from the left, subtle shadow, seamless light grey "
            "backdrop, 100mm macro lens, sharp focus, commercial photography"
        ),
        "size": [1024, 1024],
        "pipeline": "flux_q5_fast",
        "rui_ro": THAP,
        "ghi_chu": "Vật vô tri → không có lỗi giải phẫu, dùng fast cho nhanh.",
    },
    {
        "id": "anh_minh_hoa",
        "ten": "Minh họa anime",
        "prompt": (
            "Anime illustration of a girl with long silver hair and aqua eyes in a school "
            "uniform, standing under cherry blossoms, soft afternoon sunlight, clean linework, "
            "cel shading, detailed background, studio anime key visual"
        ),
        "size": [832, 1216],
        "pipeline": "flux_q5_fast",
        "rui_ro": CAO,
        "ghi_chu": ("YOLO mặt (face_yolov8m.pt) được huấn luyện trên mặt người thật nên "
                    "có thể KHÔNG nhận diện được mặt anime → FaceDetailer vô tác dụng. "
                    "Dùng fast, hoặc tự sửa bằng Cell 7."),
    },
]

# Những cụm nên TRÁNH — kèm lý do
ANTI_PATTERNS = [
    ("five fingers, perfect hands, detailed fingers",
     "Nhấn mạnh số ngón khiến model chưng cất hay sinh THÊM ngón. Hãy tả tay đang làm gì."),
    ("best quality, masterpiece, 8k, ultra detailed",
     "Tag chất lượng kiểu SDXL không ăn với FLUX; làm prompt dài mà không thêm chi tiết gì."),
    ("Mọi thứ bạn KHÔNG muốn (negative prompt)",
     "cfg=1.0 → comfy/samplers.py:610 bỏ hẳn nhánh negative. Viết vào cũng không được đọc."),
    ("Prompt quá 60 từ",
     "T5-XXL cắt ở 512 token, nhưng prompt dài làm loãng ý chính. Giữ 30-45 từ."),
    ("Kích thước quá xa 1 megapixel",
     "VD 512×512 hay 2048×2048: schnell sinh lỗi cấu trúc. Giữ ~1MP: 832×1216 / 1024²."),
]


def build() -> dict:
    return {
        "negative": NEGATIVE,
        "luu_y": [
            "cfg=1.0 trên FLUX.1-schnell → negative prompt KHÔNG được dùng (comfy/samplers.py:610).",
            "Cách tránh lỗi tay hiệu quả nhất: tả tay đang cầm/giấu/đan, đừng đòi 'five fingers'.",
            "Viết câu tự nhiên: chủ thể → tư thế → bối cảnh → ánh sáng → ống kính → khung hình.",
            "Giữ ~1 megapixel: 832×1216 (dọc), 1216×832 (ngang), 1024×1024 (vuông).",
        ],
        "anti_patterns": [{"cum": a, "ly_do": b} for a, b in ANTI_PATTERNS],
        "presets": PRESETS,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "workflows", "prompts.json"))
    ap.add_argument("--print", dest="just_print", action="store_true")
    args = ap.parse_args()

    data = build()
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if args.just_print:
        print(text)
        return 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"✅ {os.path.relpath(args.out, ROOT)} ({len(PRESETS)} preset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
