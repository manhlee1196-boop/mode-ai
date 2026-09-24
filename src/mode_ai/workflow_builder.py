"""
Workflow Builder cho FLUX.1-schnell GGUF - Tạo ComfyUI workflow JSON programmatically
Hỗ trợ cả GGUF (cần custom nodes) và BUILTIN (chỉ built-in nodes, không cần cài thêm)
"""
import json
from pathlib import Path
from typing import Dict, Optional, Any
from .config import ModelConfig, WorkflowConfig, KSamplerConfig, FaceDetailerConfig
from .prompt_enhancer import PromptEnhancer

class FluxWorkflowBuilder:
    """Builder cho FLUX workflows"""
    
    def __init__(self, model_config: Optional[ModelConfig] = None, workflow_config: Optional[WorkflowConfig] = None, use_builtin: bool = False):
        """
        Args:
            model_config: Cấu hình model
            workflow_config: Cấu hình workflow
            use_builtin: Nếu True, dùng UNETLoader + DualCLIPLoader built-in (không cần ComfyUI-GGUF)
                         Nếu False, dùng UnetLoaderGGUF + DualCLIPLoaderGGUF (cần cài ComfyUI-GGUF)
        """
        self.model_config = model_config or ModelConfig()
        self.workflow_config = workflow_config or WorkflowConfig()
        self.prompt_enhancer = PromptEnhancer()
        self.use_builtin = use_builtin
        self._nodes: Dict[str, Dict] = {}
        self._next_id = 1
    
    def _add_node(self, class_type: str, inputs: Dict[str, Any], title: str, node_id: Optional[str] = None) -> str:
        """Add node và return id"""
        if node_id is None:
            node_id = str(self._next_id)
            self._next_id += 1
        else:
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
    
    def _get_unet_loader(self):
        """Get UNET loader config based on use_builtin flag"""
        if self.use_builtin:
            # Built-in UNETLoader - dùng fp8 model, không cần custom nodes
            # Model name cho builtin: flux1-schnell-fp8.safetensors
            unet_name = self.model_config.unet_name.replace(".gguf", "-fp8.safetensors")
            # Nếu vẫn là .gguf thì fallback sang fp8
            if unet_name.endswith(".gguf"):
                unet_name = "flux1-schnell-fp8.safetensors"
            return "UNETLoader", {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}, f"Load Diffusion Model - {unet_name} (BUILTIN)"
        else:
            # GGUF loader - cần ComfyUI-GGUF custom node
            return "UnetLoaderGGUF", {"unet_name": self.model_config.unet_name}, f"Unet Loader GGUF - {self.model_config.unet_name}"
    
    def _get_clip_loader(self):
        """Get CLIP loader config"""
        if self.use_builtin:
            # Built-in DualCLIPLoader
            # t5 gguf -> t5xxl_fp8
            t5_name = self.model_config.t5_name.replace(".gguf", "").replace("t5-v1_1-xxl-encoder-", "")
            # Map Q4_K_M etc to fp8
            t5_name = "t5xxl_fp8_e4m3fn.safetensors"
            return "DualCLIPLoader", {
                "clip_name1": t5_name,
                "clip_name2": self.model_config.clip_l_name,
                "type": "flux"
            }, "DualCLIPLoader - T5 + CLIP_L (BUILTIN)"
        else:
            return "DualCLIPLoaderGGUF", {
                "clip_name1": self.model_config.t5_name,
                "clip_name2": self.model_config.clip_l_name,
                "type": "flux"
            }, "DualCLIPLoader GGUF"
    
    def build_simple(self, prompt: str, negative_prompt: str = "", seed: int = 42) -> Dict:
        """Build simple workflow - chỉ generate, không detailer"""
        self._nodes = {}
        self._next_id = 1
        
        # Loaders - tự động chọn GGUF hoặc BUILTIN
        unet_class, unet_inputs, unet_title = self._get_unet_loader()
        clip_class, clip_inputs, clip_title = self._get_clip_loader()
        
        unet_id = self._add_node(unet_class, unet_inputs, unet_title)
        clip_id = self._add_node(clip_class, clip_inputs, clip_title)
        
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
        prefix = "FLUX_builtin_fp8_simple" if self.use_builtin else self.workflow_config.save_prefix
        save_id = self._add_node("SaveImage", {
            "filename_prefix": prefix,
            "images": [decode_id, 0]
        }, "Save Image")
        
        return self._nodes
    
    def build_optimized(self, prompt: str, negative_prompt: str = "", seed: int = 42) -> Dict:
        """Build optimized workflow với FaceDetailer face+hand+foot (tối ưu nhất)"""
        self._nodes = {}
        self._next_id = 1
        
        # Loaders
        unet_class, unet_inputs, unet_title = self._get_unet_loader()
        clip_class, clip_inputs, clip_title = self._get_clip_loader()
        
        unet_id = self._add_node(unet_class, unet_inputs, unet_title, "1")
        clip_id = self._add_node(clip_class, clip_inputs, clip_title, "2")
        
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
        
        # Detectors (cần Impact Pack)
        face_detector_id = self._add_node("UltralyticsDetectorProvider", {
            "model_name": self.model_config.yolo_face
        }, "YOLO Face Detector - Impact Subpack", "10")
        
        hand_detector_id = self._add_node("UltralyticsDetectorProvider", {
            "model_name": self.model_config.yolo_hand
        }, "YOLO Hand Detector - Impact Subpack", "11")
        
        foot_detector_id = self._add_node("UltralyticsDetectorProvider", {
            "model_name": self.model_config.yolo_foot
        }, "YOLO Foot Detector - Impact Subpack", "12")
        
        sam_id = self._add_node("SAMLoader", {
            "model_name": self.model_config.sam_model,
            "device_mode": "AUTO"
        }, "SAM Loader - Impact Pack", "13")
        
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
            }, f"FaceDetailer FACE denoise {face_cfg.denoise} - Impact Pack", "20")
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
            }, f"FaceDetailer HAND denoise {hand_cfg.denoise} - Impact Pack", "21")
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
            }, f"FaceDetailer FOOT denoise {foot_cfg.denoise} - Impact Pack", "22")
            last_image_id = foot_detailer_id
        
        # Save final
        prefix = "FLUX_builtin_fp8_toi_uu" if self.use_builtin else self.workflow_config.save_prefix
        save_id = self._add_node("SaveImage", {
            "filename_prefix": prefix,
            "images": [last_image_id, 0]
        }, "Save Final Image", "9")
        
        base_save_id = self._add_node("SaveImage", {
            "filename_prefix": f"{prefix}_base",
            "images": [decode_id, 0]
        }, "Save Base Image", "30")
        
        return self._nodes
    
    def build_inpaint(self, prompt: str, negative_prompt: str = "", seed: int = 42, grow_mask: int = 12) -> Dict:
        """Build inpaint workflow cho Gradio"""
        self._nodes = {}
        self._next_id = 1
        
        unet_class, unet_inputs, unet_title = self._get_unet_loader()
        clip_class, clip_inputs, clip_title = self._get_clip_loader()
        
        unet_id = self._add_node(unet_class, unet_inputs, unet_title, "1")
        clip_id = self._add_node(clip_class, clip_inputs, clip_title, "2")
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
        print(f"✅ Saved workflow to {p} ({len(workflow)} nodes) - {'BUILTIN (no custom nodes)' if self.use_builtin else 'GGUF (needs ComfyUI-GGUF)'}")
    
    def load(self, path: str) -> Dict:
        """Load workflow from JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def validate(self, workflow: Dict) -> tuple[bool, list[str]]:
        """Validate workflow"""
        errors = []
        
        class_types = [node.get("class_type") for node in workflow.values()]
        
        # Check có ít nhất 1 UNET loader (GGUF hoặc builtin)
        has_unet = any(ct in ["UnetLoaderGGUF", "UNETLoader", "Load Diffusion Model"] for ct in class_types)
        if not has_unet:
            errors.append("Missing UNET loader: need UnetLoaderGGUF or UNETLoader")
        
        # Check CLIP loader
        has_clip = any(ct in ["DualCLIPLoaderGGUF", "DualCLIPLoader"] for ct in class_types)
        if not has_clip:
            errors.append("Missing CLIP loader: need DualCLIPLoaderGGUF or DualCLIPLoader")
        
        required_builtin = ["VAELoader", "KSampler", "VAEDecode", "SaveImage"]
        for req in required_builtin:
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
        node_types = list(set(node.get("class_type") for node in workflow.values()))
        
        # Phân loại custom nodes
        gguf_nodes = [ct for ct in node_types if "GGUF" in ct]
        impact_nodes = [ct for ct in node_types if ct in ["FaceDetailer", "SAMLoader", "UltralyticsDetectorProvider"]]
        builtin_only = len(gguf_nodes) == 0 and len(impact_nodes) == 0
        
        return {
            "total_nodes": len(workflow),
            "node_types": node_types,
            "has_face_detailer": any(n.get("class_type") == "FaceDetailer" for n in workflow.values()),
            "has_sam": any(n.get("class_type") == "SAMLoader" for n in workflow.values()),
            "has_yolo": any(n.get("class_type") == "UltralyticsDetectorProvider" for n in workflow.values()),
            "has_gguf": len(gguf_nodes) > 0,
            "gguf_nodes": gguf_nodes,
            "impact_nodes": impact_nodes,
            "builtin_only": builtin_only,
            "custom_nodes_required": gguf_nodes + impact_nodes,
        }
    
    def get_install_guide(self, workflow: Dict) -> str:
        """Get install guide based on workflow requirements"""
        stats = self.get_stats(workflow)
        
        if stats["builtin_only"]:
            return "✅ Workflow này chỉ dùng built-in nodes, không cần cài thêm gì! Load và chạy ngay."
        
        guide = []
        if stats["has_gguf"]:
            guide.append("🔧 Cần cài ComfyUI-GGUF (fix 2 nodes lỗi của bạn):")
            guide.append("   cd ComfyUI/custom_nodes && git clone https://github.com/city96/ComfyUI-GGUF.git")
            guide.append("   pip install -r ComfyUI-GGUF/requirements.txt")
        
        if stats["has_face_detailer"] or stats["has_sam"] or stats["has_yolo"]:
            guide.append("\n🔧 Cần cài Impact Pack (cho FaceDetailer):")
            guide.append("   git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git")
            guide.append("   git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git")
        
        guide.append("\n🔄 Sau đó restart ComfyUI")
        guide.append("\n💡 Hoặc dùng script tự động:")
        guide.append("   python scripts/install_comfyui_nodes.py --comfyui-path /path/to/ComfyUI")
        guide.append("   bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI")
        
        if stats["has_gguf"]:
            guide.append("\n🎯 Chỉ muốn fix 2 nodes lỗi nhanh:")
            guide.append("   bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI --only-gguf")
        
        return "\n".join(guide)
