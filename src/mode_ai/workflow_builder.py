"""
Workflow Builder cho FLUX.1-schnell GGUF - Tạo ComfyUI workflow JSON programmatically
"""
import json
from pathlib import Path
from typing import Dict, Optional, Any
from .config import ModelConfig, WorkflowConfig, KSamplerConfig, FaceDetailerConfig
from .prompt_enhancer import PromptEnhancer

class FluxWorkflowBuilder:
    """Builder cho FLUX workflows"""
    
    def __init__(self, model_config: Optional[ModelConfig] = None, workflow_config: Optional[WorkflowConfig] = None):
        self.model_config = model_config or ModelConfig()
        self.workflow_config = workflow_config or WorkflowConfig()
        self.prompt_enhancer = PromptEnhancer()
        self._nodes: Dict[str, Dict] = {}
        self._next_id = 1
    
    def _add_node(self, class_type: str, inputs: Dict[str, Any], title: str, node_id: Optional[str] = None) -> str:
        """Add node và return id"""
        if node_id is None:
            node_id = str(self._next_id)
            self._next_id += 1
        else:
            # Ensure next_id is ahead
            try:
                nid_int = int(node_id)
                if nid_int >= self._next_id:
                    self._next_id = nid_int + 1
            except:
                pass
        
        self._nodes[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {"title": title}
        }
        return node_id
    
    def build_simple(self, prompt: str, negative_prompt: str = "", seed: int = 42) -> Dict:
        """Build simple workflow - chỉ generate, không detailer"""
        self._nodes = {}
        self._next_id = 1
        
        # Loaders
        unet_id = self._add_node("UnetLoaderGGUF", {
            "unet_name": self.model_config.unet_name
        }, f"Unet Loader - {self.model_config.unet_name}")
        
        clip_id = self._add_node("DualCLIPLoaderGGUF", {
            "clip_name1": self.model_config.t5_name,
            "clip_name2": self.model_config.clip_l_name,
            "type": "flux"
        }, "DualCLIPLoader GGUF")
        
        vae_id = self._add_node("VAELoader", {
            "vae_name": self.model_config.vae_name
        }, "VAE Loader")
        
        # Latent
        latent_id = self._add_node("EmptyLatentImage", {
            "width": self.workflow_config.width,
            "height": self.workflow_config.height,
            "batch_size": self.workflow_config.batch_size
        }, f"Empty Latent {self.workflow_config.width}x{self.workflow_config.height}")
        
        # Prompts
        pos_id = self._add_node("CLIPTextEncode", {
            "text": prompt,
            "clip": [clip_id, 0]
        }, "Positive Prompt")
        
        neg_id = self._add_node("CLIPTextEncode", {
            "text": negative_prompt,
            "clip": [clip_id, 0]
        }, "Negative Prompt")
        
        # KSampler
        ksampler_id = self._add_node("KSampler", {
            "seed": seed,
            "steps": self.workflow_config.sampler.steps,
            "cfg": self.workflow_config.sampler.cfg,
            "sampler_name": self.workflow_config.sampler.sampler_name,
            "scheduler": self.workflow_config.sampler.scheduler,
            "denoise": self.workflow_config.sampler.denoise,
            "model": [unet_id, 0],
            "positive": [pos_id, 0],
            "negative": [neg_id, 0],
            "latent_image": [latent_id, 0]
        }, f"KSampler {self.workflow_config.sampler.steps} steps")
        
        # Decode
        decode_id = self._add_node("VAEDecode", {
            "samples": [ksampler_id, 0],
            "vae": [vae_id, 0]
        }, "VAE Decode")
        
        # Save
        save_id = self._add_node("SaveImage", {
            "filename_prefix": self.workflow_config.save_prefix,
            "images": [decode_id, 0]
        }, "Save Image")
        
        return self._nodes
    
    def build_optimized(self, prompt: str, negative_prompt: str = "", seed: int = 42) -> Dict:
        """Build optimized workflow với FaceDetailer face+hand+foot (tối ưu nhất)"""
        self._nodes = {}
        self._next_id = 1
        
        # Loaders
        unet_id = self._add_node("UnetLoaderGGUF", {
            "unet_name": self.model_config.unet_name
        }, f"Unet Loader - {self.model_config.unet_name}", "1")
        
        clip_id = self._add_node("DualCLIPLoaderGGUF", {
            "clip_name1": self.model_config.t5_name,
            "clip_name2": self.model_config.clip_l_name,
            "type": "flux"
        }, "DualCLIP GGUF", "2")
        
        vae_id = self._add_node("VAELoader", {
            "vae_name": self.model_config.vae_name
        }, "VAE Loader", "3")
        
        latent_id = self._add_node("EmptyLatentImage", {
            "width": self.workflow_config.width,
            "height": self.workflow_config.height,
            "batch_size": self.workflow_config.batch_size
        }, f"Latent {self.workflow_config.width}x{self.workflow_config.height}", "4")
        
        pos_id = self._add_node("CLIPTextEncode", {
            "text": prompt,
            "clip": [clip_id, 0]
        }, "Positive", "5")
        
        neg_id = self._add_node("CLIPTextEncode", {
            "text": negative_prompt,
            "clip": [clip_id, 0]
        }, "Negative", "6")
        
        ksampler_id = self._add_node("KSampler", {
            "seed": seed,
            "steps": self.workflow_config.sampler.steps,
            "cfg": self.workflow_config.sampler.cfg,
            "sampler_name": self.workflow_config.sampler.sampler_name,
            "scheduler": self.workflow_config.sampler.scheduler,
            "denoise": self.workflow_config.sampler.denoise,
            "model": [unet_id, 0],
            "positive": [pos_id, 0],
            "negative": [neg_id, 0],
            "latent_image": [latent_id, 0]
        }, "KSampler Main", "7")
        
        decode_id = self._add_node("VAEDecode", {
            "samples": [ksampler_id, 0],
            "vae": [vae_id, 0]
        }, "VAE Decode Base", "8")
        
        # Detectors
        face_detector_id = self._add_node("UltralyticsDetectorProvider", {
            "model_name": self.model_config.yolo_face
        }, "YOLO Face Detector", "10")
        
        hand_detector_id = self._add_node("UltralyticsDetectorProvider", {
            "model_name": self.model_config.yolo_hand
        }, "YOLO Hand Detector", "11")
        
        foot_detector_id = self._add_node("UltralyticsDetectorProvider", {
            "model_name": self.model_config.yolo_foot
        }, "YOLO Foot Detector", "12")
        
        sam_id = self._add_node("SAMLoader", {
            "model_name": self.model_config.sam_model,
            "device_mode": "AUTO"
        }, "SAM Loader", "13")
        
        # FaceDetailer chain
        last_image_id = decode_id
        
        if self.workflow_config.enable_face:
            face_cfg = self.workflow_config.face_detailer
            face_detailer_id = self._add_node("FaceDetailer", {
                "image": [last_image_id, 0],
                "model": [unet_id, 0],
                "clip": [clip_id, 0],
                "vae": [vae_id, 0],
                "guide_size": face_cfg.guide_size,
                "guide_size_for": face_cfg.guide_size_for,
                "max_size": face_cfg.max_size,
                "seed": seed + 100,
                "steps": face_cfg.steps,
                "cfg": face_cfg.cfg,
                "sampler_name": face_cfg.sampler_name,
                "scheduler": face_cfg.scheduler,
                "positive": [pos_id, 0],
                "negative": [neg_id, 0],
                "denoise": face_cfg.denoise,
                "feather": face_cfg.feather,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": face_cfg.bbox_threshold,
                "bbox_dilation": face_cfg.bbox_dilation,
                "bbox_crop_factor": face_cfg.bbox_crop_factor,
                "sam_detection_hint": "center-1",
                "sam_dilation": face_cfg.sam_dilation,
                "sam_threshold": face_cfg.sam_threshold,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "bbox_detector": [face_detector_id, 0],
                "sam_model_opt": [sam_id, 0] if face_cfg.use_sam else None,
                "tiled_encode": face_cfg.tiled_encode,
                "tiled_decode": face_cfg.tiled_decode
            }, f"FaceDetailer FACE denoise {face_cfg.denoise}", "20")
            # Remove None values
            self._nodes[face_detailer_id]["inputs"] = {k: v for k, v in self._nodes[face_detailer_id]["inputs"].items() if v is not None}
            last_image_id = face_detailer_id
        
        if self.workflow_config.enable_hand:
            hand_cfg = self.workflow_config.hand_detailer
            hand_detailer_id = self._add_node("FaceDetailer", {
                "image": [last_image_id, 0],
                "model": [unet_id, 0],
                "clip": [clip_id, 0],
                "vae": [vae_id, 0],
                "guide_size": hand_cfg.guide_size,
                "guide_size_for": hand_cfg.guide_size_for,
                "max_size": hand_cfg.max_size,
                "seed": seed + 101,
                "steps": hand_cfg.steps,
                "cfg": hand_cfg.cfg,
                "sampler_name": hand_cfg.sampler_name,
                "scheduler": hand_cfg.scheduler,
                "positive": [pos_id, 0],
                "negative": [neg_id, 0],
                "denoise": hand_cfg.denoise,
                "feather": hand_cfg.feather,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": hand_cfg.bbox_threshold,
                "bbox_dilation": hand_cfg.bbox_dilation,
                "bbox_crop_factor": hand_cfg.bbox_crop_factor,
                "sam_detection_hint": "center-1",
                "sam_dilation": hand_cfg.sam_dilation,
                "sam_threshold": hand_cfg.sam_threshold,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "bbox_detector": [hand_detector_id, 0],
                "tiled_encode": hand_cfg.tiled_encode,
                "tiled_decode": hand_cfg.tiled_decode
            }, f"FaceDetailer HAND denoise {hand_cfg.denoise}", "21")
            last_image_id = hand_detailer_id
        
        if self.workflow_config.enable_foot:
            foot_cfg = self.workflow_config.foot_detailer
            foot_detailer_id = self._add_node("FaceDetailer", {
                "image": [last_image_id, 0],
                "model": [unet_id, 0],
                "clip": [clip_id, 0],
                "vae": [vae_id, 0],
                "guide_size": foot_cfg.guide_size,
                "guide_size_for": foot_cfg.guide_size_for,
                "max_size": foot_cfg.max_size,
                "seed": seed + 102,
                "steps": foot_cfg.steps,
                "cfg": foot_cfg.cfg,
                "sampler_name": foot_cfg.sampler_name,
                "scheduler": foot_cfg.scheduler,
                "positive": [pos_id, 0],
                "negative": [neg_id, 0],
                "denoise": foot_cfg.denoise,
                "feather": foot_cfg.feather,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": foot_cfg.bbox_threshold,
                "bbox_dilation": foot_cfg.bbox_dilation,
                "bbox_crop_factor": foot_cfg.bbox_crop_factor,
                "sam_detection_hint": "center-1",
                "sam_dilation": foot_cfg.sam_dilation,
                "sam_threshold": foot_cfg.sam_threshold,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "bbox_detector": [foot_detector_id, 0],
                "tiled_encode": foot_cfg.tiled_encode,
                "tiled_decode": foot_cfg.tiled_decode
            }, f"FaceDetailer FOOT denoise {foot_cfg.denoise}", "22")
            last_image_id = foot_detailer_id
        
        # Save final
        save_id = self._add_node("SaveImage", {
            "filename_prefix": self.workflow_config.save_prefix,
            "images": [last_image_id, 0]
        }, "Save Final Image", "9")
        
        # Save base for comparison
        base_save_id = self._add_node("SaveImage", {
            "filename_prefix": f"{self.workflow_config.save_prefix}_base",
            "images": [decode_id, 0]
        }, "Save Base Image", "30")
        
        return self._nodes
    
    def build_inpaint(self, prompt: str, negative_prompt: str = "", seed: int = 42, grow_mask: int = 12) -> Dict:
        """Build inpaint workflow cho Gradio"""
        self._nodes = {}
        self._next_id = 1
        
        unet_id = self._add_node("UnetLoaderGGUF", {"unet_name": self.model_config.unet_name}, "Unet GGUF", "1")
        clip_id = self._add_node("DualCLIPLoaderGGUF", {
            "clip_name1": self.model_config.t5_name,
            "clip_name2": self.model_config.clip_l_name,
            "type": "flux"
        }, "DualCLIP", "2")
        vae_id = self._add_node("VAELoader", {"vae_name": self.model_config.vae_name}, "VAE", "3")
        
        load_image_id = self._add_node("LoadImage", {"image": "input_image.png"}, "Load Image", "4")
        load_mask_id = self._add_node("LoadImage", {"image": "mask.png"}, "Load Mask", "5")
        mask_id = self._add_node("ImageToMask", {"image": [load_mask_id, 0], "channel": "red"}, "Image to Mask", "6")
        
        pos_id = self._add_node("CLIPTextEncode", {"text": prompt, "clip": [clip_id, 0]}, "Positive", "7")
        neg_id = self._add_node("CLIPTextEncode", {"text": negative_prompt, "clip": [clip_id, 0]}, "Negative", "8")
        
        encode_id = self._add_node("VAEEncodeForInpaint", {
            "pixels": [load_image_id, 0],
            "vae": [vae_id, 0],
            "mask": [mask_id, 0],
            "grow_mask_by": grow_mask
        }, f"VAE Encode Inpaint grow {grow_mask}", "9")
        
        ksampler_id = self._add_node("KSampler", {
            "seed": seed,
            "steps": 8,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.5,
            "model": [unet_id, 0],
            "positive": [pos_id, 0],
            "negative": [neg_id, 0],
            "latent_image": [encode_id, 0]
        }, "KSampler Inpaint 8 steps denoise 0.5", "10")
        
        decode_id = self._add_node("VAEDecode", {"samples": [ksampler_id, 0], "vae": [vae_id, 0]}, "VAE Decode", "11")
        save_id = self._add_node("SaveImage", {"filename_prefix": "FLUX_Q5_inpaint_fixed", "images": [decode_id, 0]}, "Save Inpainted", "12")
        
        return self._nodes
    
    def build_with_style(self, prompt: str, style: str = "aesthetic_anime", seed: int = 42, optimized: bool = True) -> Dict:
        """Build workflow với style preset"""
        enhancer = PromptEnhancer(style=style)
        enhanced_prompt = enhancer.enhance(prompt)
        negative = enhancer.get_negative()
        
        if optimized:
            return self.build_optimized(enhanced_prompt, negative, seed)
        else:
            return self.build_simple(enhanced_prompt, negative, seed)
    
    def save(self, workflow: Dict, path: str):
        """Save workflow to JSON"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved workflow to {p} ({len(workflow)} nodes)")
    
    def load(self, path: str) -> Dict:
        """Load workflow from JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def validate(self, workflow: Dict) -> tuple[bool, list[str]]:
        """Validate workflow"""
        errors = []
        
        # Check required nodes
        class_types = [node.get("class_type") for node in workflow.values()]
        
        required = ["UnetLoaderGGUF", "DualCLIPLoaderGGUF", "VAELoader", "KSampler", "VAEDecode", "SaveImage"]
        for req in required:
            if req not in class_types:
                errors.append(f"Missing required node: {req}")
        
        # Check FLUX specific params
        for node_id, node in workflow.items():
            if node.get("class_type") == "KSampler":
                cfg = node.get("inputs", {}).get("cfg")
                steps = node.get("inputs", {}).get("steps")
                if cfg != 1.0:
                    errors.append(f"Node {node_id} KSampler cfg should be 1.0 for FLUX, got {cfg}")
                if steps and steps < 4:
                    errors.append(f"Node {node_id} KSampler steps too low for schnell: {steps}")
        
        return len(errors) == 0, errors
    
    def get_stats(self, workflow: Dict) -> Dict:
        """Get workflow stats"""
        return {
            "total_nodes": len(workflow),
            "node_types": list(set(node.get("class_type") for node in workflow.values())),
            "has_face_detailer": any(n.get("class_type") == "FaceDetailer" for n in workflow.values()),
            "has_sam": any(n.get("class_type") == "SAMLoader" for n in workflow.values()),
            "has_yolo": any(n.get("class_type") == "UltralyticsDetectorProvider" for n in workflow.values()),
        }
