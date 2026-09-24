#!/usr/bin/env python3
"""Sinh toàn bộ workflow FLUX.1-schnell (GGUF Q5) cho pipeline tối ưu.

Đây là NGUỒN DUY NHẤT của workflows/*.json — cùng đoạn code này được nhúng vào
Cell 4 của notebook Colab (xem scripts/make_notebook.py), nên workflow chạy trên Colab
và workflow đã kiểm tra tĩnh trong repo luôn giống hệt nhau.

    python3 scripts/build_workflows.py            # ghi workflows/*.json
    python3 scripts/build_workflows.py --print    # chỉ in ra, không ghi
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

# ----------------------------------------------------------------- model (khớp Cell 3)
UNET = "flux1-schnell-Q5_K_S.gguf"
CLIP_L = "clip_l.safetensors"
T5 = "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
VAE = "ae.safetensors"
YOLO_FACE = "bbox/face_yolov8m.pt"          # Impact Subpack BẮT BUỘC tiền tố bbox/
YOLO_HAND = "bbox/hand_yolov8s.pt"
SAM = "sam_vit_b_01ec64.pth"

# ----------------------------------------------------------------- tham số tối ưu T4 16GB
BASE_STEPS = 4          # FLUX.1-schnell là model distill 4 bước
BASE_CFG = 1.0          # schnell không dùng CFG → bỏ luôn nhánh negative khi sample
DETAIL_STEPS = 4
FACE_DENOISE = 0.22
HAND_DENOISE = 0.28
HIRES_DENOISE = 0.35

POS_DEFAULT = (
    "photorealistic portrait of a young Vietnamese woman, natural skin texture with visible pores, "
    "soft window light, 85mm lens, shallow depth of field, detailed eyes and hands, five fingers, "
    "casual linen shirt, warm neutral background, film grain, high detail"
)
NEG_DEFAULT = ""   # schnell + cfg 1.0 → negative bị bỏ qua, để trống cho T5 encode nhanh


# ----------------------------------------------------------------- khối node dùng chung
def loaders() -> Dict[str, Any]:
    """1 lần nạp model, dùng lại cho mọi node phía sau (tiết kiệm VRAM + thời gian)."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "2": {"class_type": "DualCLIPLoaderGGUF", "inputs": {
            "clip_name1": CLIP_L, "clip_name2": T5, "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }


def cond(pos: str = POS_DEFAULT, neg: str = NEG_DEFAULT) -> Dict[str, Any]:
    return {
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
    }


def latent(nid: str = "4", width: int = 1024, height: int = 1024,
           batch: int = 1) -> Dict[str, Any]:
    return {nid: {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": batch}}}


def ksampler(nid: str, latent_id: str = "4", seed: int = 0, steps: int = BASE_STEPS,
             cfg: float = BASE_CFG, denoise: float = 1.0, model: str = "1",
             pos: str = "5", neg: str = "6") -> Dict[str, Any]:
    return {nid: {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "denoise": denoise,
        "model": [model, 0], "positive": [pos, 0], "negative": [neg, 0],
        "latent_image": [latent_id, 0]}}}


def detailer(nid: str, image_id: str, detector_id: str, seed: int, denoise: float,
             feather: int, crop_factor: float, threshold: float = 0.5, dilation: int = 8,
             sam_id: str | None = None, guide_size: int = 384, max_size: int = 768,
             steps: int = DETAIL_STEPS) -> Dict[str, Any]:
    """FaceDetailer đầy đủ input required (thiếu `wildcard`/`cycle` là ComfyUI báo lỗi)."""
    ins: Dict[str, Any] = {
        "image": [image_id, 0], "model": ["1", 0], "clip": ["2", 0], "vae": ["3", 0],
        "guide_size": guide_size, "guide_size_for": True, "max_size": max_size,
        "seed": seed, "steps": steps, "cfg": BASE_CFG,
        "sampler_name": "euler", "scheduler": "simple",
        "positive": ["5", 0], "negative": ["6", 0],
        "denoise": denoise, "feather": feather, "noise_mask": True, "force_inpaint": True,
        "bbox_threshold": threshold, "bbox_dilation": dilation,
        "bbox_crop_factor": crop_factor,
        "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
        "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
        "sam_mask_hint_use_negative": "False",
        "drop_size": 10, "bbox_detector": [detector_id, 0], "wildcard": "", "cycle": 1,
    }
    if sam_id:
        ins["sam_model_opt"] = [sam_id, 0]
    # tiled encode/decode: crop nhỏ nên không tốn VRAM, tránh OOM trên T4
    ins["tiled_encode"] = True
    ins["tiled_decode"] = True
    return {nid: {"class_type": "FaceDetailer", "inputs": ins}}


def detector(nid: str, model_name: str) -> Dict[str, Any]:
    return {nid: {"class_type": "UltralyticsDetectorProvider",
                  "inputs": {"model_name": model_name}}}


def sam_loader(nid: str = "13") -> Dict[str, Any]:
    return {nid: {"class_type": "SAMLoader",
                  "inputs": {"model_name": SAM, "device_mode": "AUTO"}}}


def save(nid: str, prefix: str, from_id: str) -> Dict[str, Any]:
    return {nid: {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": prefix, "images": [from_id, 0]}}}


# ----------------------------------------------------------------- 5 pipeline
def build_fast(width: int = 1024, height: int = 1024,
               pos: str = POS_DEFAULT, neg: str = NEG_DEFAULT) -> Dict[str, Any]:
    """Nhanh nhất: 4 bước, không detailer. ~15-20s/ảnh 1024² trên T4."""
    wf: Dict[str, Any] = {}
    wf.update(loaders())
    wf.update(cond(pos, neg))
    wf.update(latent("4", width, height))
    wf.update(ksampler("7"))
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf.update(save("9", "flux/fast", "8"))
    return wf


def build_standard(width: int = 1024, height: int = 1024,
                   pos: str = POS_DEFAULT, neg: str = NEG_DEFAULT) -> Dict[str, Any]:
    """Mặc định: 4 bước + sửa mặt bằng SAM. ~25-35s/ảnh."""
    wf = build_fast(width, height, pos, neg)
    wf.update(detector("10", YOLO_FACE))
    wf.update(sam_loader("13"))
    wf.update(save("30", "flux/base", "8"))
    wf.update(detailer("20", "8", "10", seed=100, denoise=FACE_DENOISE,
                       feather=8, crop_factor=3.0, sam_id="13"))
    wf["9"] = save("9", "flux/standard", "20")["9"]
    return wf


def build_quality(width: int = 1024, height: int = 1024,
                  pos: str = POS_DEFAULT, neg: str = NEG_DEFAULT) -> Dict[str, Any]:
    """Chất lượng: 4 bước + sửa mặt (SAM) + sửa tay. ~40-55s/ảnh."""
    wf = build_standard(width, height, pos, neg)
    wf.update(detector("11", YOLO_HAND))
    wf.update(detailer("21", "20", "11", seed=101, denoise=HAND_DENOISE,
                       feather=16, crop_factor=2.5, threshold=0.35, dilation=6))
    wf["9"] = save("9", "flux/quality", "21")["9"]
    return wf


def build_hires(width: int = 832, height: int = 1216,
                pos: str = POS_DEFAULT, neg: str = NEG_DEFAULT) -> Dict[str, Any]:
    """Ảnh nét cỡ lớn: base + sửa mặt, upscale 1.5× rồi lấy lại chi tiết bằng 4 bước
    denoise thấp. Chỉ dùng node có sẵn trong ComfyUI (không cần ESRGAN/UltimateSDUpscale)."""
    wf = build_standard(width, height, pos, neg)
    wf["40"] = {"class_type": "ImageScaleBy", "inputs": {
        "image": ["20", 0], "upscale_method": "lanczos", "scale_by": 1.5}}
    wf["41"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["40", 0], "vae": ["3", 0]}}
    wf.update(ksampler("42", latent_id="41", seed=200, denoise=HIRES_DENOISE))
    wf["43"] = {"class_type": "VAEDecode", "inputs": {"samples": ["42", 0], "vae": ["3", 0]}}
    wf["9"] = save("9", "flux/hires", "43")["9"]
    wf["30"] = save("30", "flux/hires_base", "20")["30"]
    return wf


def build_inpaint(image: str = "input_image.png", mask: str = "mask.png",
                  pos: str = ("detailed human hand, five fingers, natural fingernails, "
                              "realistic skin texture, photorealistic, sharp focus"),
                  neg: str = NEG_DEFAULT, denoise: float = 0.5,
                  steps: int = 6, grow_mask_by: int = 12) -> Dict[str, Any]:
    """Sửa vùng tô (tay/mặt/chân) trên ảnh có sẵn."""
    wf: Dict[str, Any] = {}
    wf.update(loaders())
    wf.update(cond(pos, neg))
    wf["4"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    wf["5m"] = {"class_type": "LoadImage", "inputs": {"image": mask}}
    wf["6m"] = {"class_type": "ImageToMask", "inputs": {"image": ["5m", 0], "channel": "red"}}
    wf["9e"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
        "pixels": ["4", 0], "vae": ["3", 0], "mask": ["6m", 0],
        "grow_mask_by": grow_mask_by}}
    wf.update(ksampler("7", latent_id="9e", steps=steps, denoise=denoise))
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf.update(save("10", "flux/inpaint", "8"))
    return wf


PIPELINES = {
    "flux_q5_fast": build_fast,
    "flux_q5_standard": build_standard,
    "flux_q5_quality": build_quality,
    "flux_q5_hires": build_hires,
    "flux_q5_inpaint": build_inpaint,
}


def build_all() -> Dict[str, Dict[str, Any]]:
    return {name: fn() for name, fn in PIPELINES.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workflows")
    ap.add_argument("--print", dest="just_print", action="store_true")
    args = ap.parse_args()

    for name, wf in build_all().items():
        text = json.dumps(wf, ensure_ascii=False, indent=1)
        if args.just_print:
            print(f"===== {name} ({len(wf)} node) =====")
            print(text)
            continue
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"✅ {path} ({len(wf)} node)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
