#!/usr/bin/env python3
"""
Script tạo ảnh với FLUX workflow
Usage:
  python scripts/generate.py --prompt "1girl, cherry blossoms" --style aesthetic_anime --workflow optimized
  python scripts/generate.py --prompt "Vietnamese woman in ao dai" --style realistic_vietnamese --width 832 --height 1216
"""
import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mode_ai import FluxWorkflowBuilder, ModelConfig, WorkflowConfig
from mode_ai.config import MODEL_PRESETS
from mode_ai.prompt_enhancer import PromptEnhancer, PRESET_PROMPTS
from mode_ai.comfy_api import get_client

def main():
    parser = argparse.ArgumentParser(description="FLUX.1-schnell Image Generation")
    parser.add_argument("--prompt", type=str, default="", help="Prompt tạo ảnh")
    parser.add_argument("--preset", type=str, choices=list(PRESET_PROMPTS.keys()), help="Dùng preset prompt")
    parser.add_argument("--style", type=str, default="aesthetic_anime", choices=PromptEnhancer.list_styles(), help="Style preset")
    parser.add_argument("--negative", type=str, default="", help="Negative prompt extra")
    parser.add_argument("--workflow", type=str, default="optimized", choices=["simple", "optimized", "realistic", "hires", "inpaint"], help="Loại workflow")
    parser.add_argument("--width", type=int, default=1024, help="Width")
    parser.add_argument("--height", type=int, default=1024, help="Height")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--steps", type=int, default=4, help="Steps (schnell chuẩn 4)")
    parser.add_argument("--model", type=str, default="q5_optimal", choices=list(MODEL_PRESETS.keys()), help="Model preset")
    parser.add_argument("--output", type=str, default="workflows/generated.json", help="Output workflow JSON path")
    parser.add_argument("--server", type=str, default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--no-queue", action="store_true", help="Chỉ tạo workflow JSON, không queue vào ComfyUI")
    parser.add_argument("--list-styles", action="store_true", help="List available styles")
    parser.add_argument("--list-presets", action="store_true", help="List preset prompts")
    
    args = parser.parse_args()
    
    if args.list_styles:
        print("Available styles:")
        for style in PromptEnhancer.list_styles():
            info = PromptEnhancer.get_style_info(style)
            print(f"  - {style}: prefix={info.get('prefix','')[:50]}...")
        return
    
    if args.list_presets:
        print("Available preset prompts:")
        for k, v in PRESET_PROMPTS.items():
            print(f"  - {k}: {v[:80]}...")
        return
    
    # Get prompt
    prompt = args.prompt
    if args.preset:
        prompt = PRESET_PROMPTS[args.preset]
        print(f"Using preset '{args.preset}': {prompt}")
    
    if not prompt:
        prompt = PRESET_PROMPTS["girl_cherry"]
        print(f"No prompt provided, using default: {prompt}")
    
    # Model config
    model_config = MODEL_PRESETS.get(args.model, MODEL_PRESETS["q5_optimal"])
    print(f"Model: {model_config.unet_name} ({model_config.total_size_gb:.1f}GB total)")
    
    # Workflow config
    if args.workflow == "realistic" or args.style in ["photorealistic", "realistic_vietnamese"]:
        wf_config = WorkflowConfig.portrait_realistic()
    elif args.workflow == "simple":
        wf_config = WorkflowConfig.simple()
    elif args.workflow == "optimized":
        wf_config = WorkflowConfig.square_aesthetic()
    else:
        wf_config = WorkflowConfig()
    
    wf_config.width = args.width
    wf_config.height = args.height
    wf_config.sampler.seed = args.seed
    wf_config.sampler.steps = args.steps
    
    # Builder
    builder = FluxWorkflowBuilder(model_config=model_config, workflow_config=wf_config)
    
    # Enhance prompt
    enhancer = PromptEnhancer(style=args.style)
    enhanced_prompt = enhancer.enhance(prompt)
    negative_prompt = enhancer.get_negative(extra_negative=args.negative)
    
    print(f"\n📝 Original prompt: {prompt}")
    print(f"✨ Enhanced prompt: {enhanced_prompt}")
    print(f"🚫 Negative: {negative_prompt}")
    print(f"🎨 Style: {args.style}, Workflow: {args.workflow}, Size: {args.width}x{args.height}, Seed: {args.seed}")
    
    # Build workflow
    if args.workflow == "simple":
        workflow = builder.build_simple(enhanced_prompt, negative_prompt, seed=args.seed)
    elif args.workflow == "inpaint":
        workflow = builder.build_inpaint(enhanced_prompt, negative_prompt, seed=args.seed)
    else:
        workflow = builder.build_optimized(enhanced_prompt, negative_prompt, seed=args.seed)
    
    # Validate
    is_valid, errors = builder.validate(workflow)
    stats = builder.get_stats(workflow)
    print(f"\n📊 Workflow stats: {stats}")
    if not is_valid:
        print(f"❌ Validation errors: {errors}")
        return
    else:
        print("✅ Workflow validation passed")
    
    # Save
    builder.save(workflow, args.output)
    
    # Queue to ComfyUI if requested
    if not args.no_queue:
        try:
            client = get_client(args.server, allow_mock=True)
            result = client.generate_image(workflow, wait=False)
            print(f"🚀 Queued to ComfyUI: {result}")
        except Exception as e:
            print(f"⚠️ Could not queue to ComfyUI: {e}")
            print("   Workflow JSON saved, you can manually load it in ComfyUI")
    
    print(f"\n✅ Done! Workflow saved to {args.output}")
    print("   Load this JSON in ComfyUI: Drag & drop or Load button")

if __name__ == "__main__":
    main()
