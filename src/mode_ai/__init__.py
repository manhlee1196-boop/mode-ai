"""
mode-ai - FLUX.1-schnell GGUF workflow toolkit
Tối ưu cho Colab T4 16GB, tổng model <15GB
"""

__version__ = "0.2.0"
__author__ = "mode-ai"

from .workflow_builder import FluxWorkflowBuilder
from .comfy_api import ComfyUIClient
from .config import ModelConfig, WorkflowConfig
from .prompt_enhancer import PromptEnhancer

__all__ = [
    "FluxWorkflowBuilder",
    "ComfyUIClient",
    "ModelConfig",
    "WorkflowConfig",
    "PromptEnhancer",
]
