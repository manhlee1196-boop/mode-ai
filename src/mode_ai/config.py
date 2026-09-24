"""
Cấu hình model và workflow cho FLUX
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json

@dataclass
class ModelConfig:
    """Cấu hình model FLUX GGUF"""
    # UNET GGUF - chọn Q5_K_S là tốt nhất dưới 15GB
    unet_name: str = "flux1-schnell-Q5_K_S.gguf"
    unet_size_gb: float = 8.26
    unet_quant: str = "Q5_K_S"
    
    # T5-XXL GGUF
    t5_name: str = "t5-v1_1-xxl-encoder-Q4_K_M.gguf"
    t5_size_gb: float = 2.9
    
    # CLIP_L
    clip_l_name: str = "clip_l.safetensors"
    clip_l_size_mb: int = 250
    
    # VAE
    vae_name: str = "ae.safetensors"
    vae_size_mb: int = 320
    
    # YOLO detectors
    yolo_face: str = "face_yolov8m.pt"
    yolo_hand: str = "hand_yolov8s.pt"
    yolo_foot: str = "foot_anime_yolo11m_v3.pt"
    
    # SAM
    sam_model: str = "sam_vit_b_01ec64.pth"
    sam_size_mb: int = 375
    
    # Paths
    base_dir: str = "AI_Models"
    
    @property
    def total_size_gb(self) -> float:
        return self.unet_size_gb + self.t5_size_gb + (self.clip_l_size_mb + self.vae_size_mb + self.sam_size_mb) / 1024 + 0.114  # YOLO ~114MB
    
    def get_model_paths(self) -> Dict[str, str]:
        return {
            "unet": f"{self.base_dir}/gguf/{self.unet_name}",
            "t5": f"{self.base_dir}/clip/{self.t5_name}",
            "clip_l": f"{self.base_dir}/clip/{self.clip_l_name}",
            "vae": f"{self.base_dir}/vae/{self.vae_name}",
            "yolo_face": f"{self.base_dir}/ultralytics/bbox/{self.yolo_face}",
            "yolo_hand": f"{self.base_dir}/ultralytics/bbox/{self.yolo_hand}",
            "yolo_foot": f"{self.base_dir}/ultralytics/bbox/{self.yolo_foot}",
            "sam": f"{self.base_dir}/sams/{self.sam_model}",
        }
    
    def to_dict(self) -> dict:
        return {
            "unet": self.unet_name,
            "t5": self.t5_name,
            "clip_l": self.clip_l_name,
            "vae": self.vae_name,
            "detectors": {
                "face": self.yolo_face,
                "hand": self.yolo_hand,
                "foot": self.yolo_foot,
            },
            "sam": self.sam_model,
            "total_gb": round(self.total_size_gb, 2),
        }

@dataclass
class KSamplerConfig:
    """Cấu hình KSampler cho FLUX schnell"""
    sampler_name: str = "euler"
    scheduler: str = "simple"
    steps: int = 4  # schnell chuẩn 4 steps
    cfg: float = 1.0  # FLUX schnell yêu cầu CFG=1
    denoise: float = 1.0
    seed: int = 42
    
    def to_dict(self) -> dict:
        return {
            "sampler_name": self.sampler_name,
            "scheduler": self.scheduler,
            "steps": self.steps,
            "cfg": self.cfg,
            "denoise": self.denoise,
            "seed": self.seed,
        }

@dataclass
class FaceDetailerConfig:
    """Cấu hình FaceDetailer"""
    guide_size: int = 512
    guide_size_for: bool = True
    max_size: int = 1024
    steps: int = 6
    cfg: float = 1.0
    sampler_name: str = "euler"
    scheduler: str = "simple"
    denoise: float = 0.18
    feather: int = 5
    bbox_threshold: float = 0.5
    bbox_dilation: int = 10
    bbox_crop_factor: float = 3.0
    sam_threshold: float = 0.93
    sam_dilation: int = 0
    tiled_encode: bool = True
    tiled_decode: bool = True
    use_sam: bool = True

@dataclass
class WorkflowConfig:
    """Cấu hình workflow tổng"""
    name: str = "flux_schnell_gguf_toi_uu"
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    sampler: KSamplerConfig = field(default_factory=KSamplerConfig)
    face_detailer: FaceDetailerConfig = field(default_factory=FaceDetailerConfig)
    hand_detailer: FaceDetailerConfig = field(default_factory=lambda: FaceDetailerConfig(
        denoise=0.25, feather=24, bbox_crop_factor=2.8, use_sam=False
    ))
    foot_detailer: FaceDetailerConfig = field(default_factory=lambda: FaceDetailerConfig(
        denoise=0.30, feather=24, bbox_crop_factor=3.0, use_sam=False
    ))
    enable_face: bool = True
    enable_hand: bool = True
    enable_foot: bool = True
    save_prefix: str = "FLUX_Q5_schnell"
    
    # Presets
    @classmethod
    def portrait_realistic(cls) -> "WorkflowConfig":
        return cls(
            name="flux_schnell_realistic",
            width=832,
            height=1216,
            sampler=KSamplerConfig(steps=4, cfg=1.0, seed=12345),
            face_detailer=FaceDetailerConfig(denoise=0.22, steps=8, max_size=1216),
            enable_hand=False,
            enable_foot=False,
            save_prefix="FLUX_Q5_realistic"
        )
    
    @classmethod
    def square_aesthetic(cls) -> "WorkflowConfig":
        return cls(
            name="flux_schnell_aesthetic",
            width=1024,
            height=1024,
            sampler=KSamplerConfig(steps=4, cfg=1.0),
            save_prefix="FLUX_Q5_aesthetic"
        )
    
    @classmethod
    def landscape(cls) -> "WorkflowConfig":
        return cls(
            name="flux_schnell_landscape",
            width=1216,
            height=832,
            save_prefix="FLUX_Q5_landscape"
        )
    
    @classmethod
    def simple(cls) -> "WorkflowConfig":
        return cls(
            name="flux_schnell_simple",
            width=1024,
            height=1024,
            enable_face=False,
            enable_hand=False,
            enable_foot=False,
            save_prefix="FLUX_Q5_simple"
        )

# Default model configs for different VRAM levels
MODEL_PRESETS = {
    "q5_optimal": ModelConfig(
        unet_name="flux1-schnell-Q5_K_S.gguf",
        unet_size_gb=8.26,
        t5_name="t5-v1_1-xxl-encoder-Q4_K_M.gguf",
    ),
    "q4_balanced": ModelConfig(
        unet_name="flux1-schnell-Q4_K_S.gguf",
        unet_size_gb=6.8,
        t5_name="t5-v1_1-xxl-encoder-Q4_K_M.gguf",
    ),
    "q6_high": ModelConfig(
        unet_name="flux1-schnell-Q6_K.gguf",
        unet_size_gb=9.8,
        t5_name="t5-v1_1-xxl-encoder-Q4_K_M.gguf",
    ),
    "q8_max": ModelConfig(
        unet_name="flux1-schnell-Q8_0.gguf",
        unet_size_gb=12.7,
        t5_name="t5-v1_1-xxl-encoder-Q4_K_M.gguf",
    ),
}
