#!/usr/bin/env python3
"""
Gradio App - FLUX.1-schnell Workflow Generator & ComfyUI Client
Chạy local hoặc Colab để tạo workflow và generate ảnh
"""
import sys
import json
import random
from pathlib import Path

# Add src
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("Gradio not installed, install with: pip install gradio")

from mode_ai import FluxWorkflowBuilder, ModelConfig, WorkflowConfig
from mode_ai.config import MODEL_PRESETS
from mode_ai.prompt_enhancer import PromptEnhancer, PRESET_PROMPTS
from mode_ai.comfy_api import get_client

def list_workflows():
    wf_dir = Path("workflows")
    if not wf_dir.exists():
        return []
    return [str(p) for p in wf_dir.glob("*.json")]

def generate_workflow_fn(prompt, preset, style, workflow_type, width, height, seed, model_preset, use_preset, use_builtin):
    # Use preset if selected
    if use_preset and preset in PRESET_PROMPTS:
        prompt = PRESET_PROMPTS[preset]
    
    if not prompt:
        return "❌ Vui lòng nhập prompt hoặc chọn preset", "", ""
    
    if seed == 0:
        seed = random.randint(0, 2**32 - 1)
    
    try:
        model_config = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["q5_optimal"])
        
        if workflow_type == "realistic" or style in ["photorealistic", "realistic_vietnamese"]:
            wf_config = WorkflowConfig.portrait_realistic()
        elif workflow_type == "simple":
            wf_config = WorkflowConfig.simple()
        else:
            wf_config = WorkflowConfig.square_aesthetic()
        
        wf_config.width = int(width)
        wf_config.height = int(height)
        wf_config.sampler.seed = int(seed)
        
        builder = FluxWorkflowBuilder(model_config=model_config, workflow_config=wf_config, use_builtin=use_builtin)
        
        enhancer = PromptEnhancer(style=style)
        enhanced = enhancer.enhance(prompt)
        negative = enhancer.get_negative()
        
        if workflow_type == "simple":
            workflow = builder.build_simple(enhanced, negative, seed=int(seed))
        elif workflow_type == "inpaint":
            workflow = builder.build_inpaint(enhanced, negative, seed=int(seed))
        else:
            workflow = builder.build_optimized(enhanced, negative, seed=int(seed))
        
        is_valid, errors = builder.validate(workflow)
        stats = builder.get_stats(workflow)
        
        # Save
        mode_tag = "builtin" if use_builtin else "gguf"
        output_path = Path(f"workflows/generated_{mode_tag}_{seed}.json")
        builder.save(workflow, str(output_path))
        
        mode_str = "🔧 BUILTIN (UNETLoader/DualCLIPLoader - KHÔNG cần cài custom nodes)" if use_builtin else "🔧 GGUF (UnetLoaderGGUF/DualCLIPLoaderGGUF - cần ComfyUI-GGUF)"
        
        custom_warn = ""
        if stats["custom_nodes_required"]:
            custom_warn = f"\n⚠️ Cần cài custom nodes: {stats['custom_nodes_required']}\nChạy: bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI"
        else:
            custom_warn = "\n✅ Không cần cài custom nodes - load và chạy ngay trong ComfyUI!"
        
        info = f"""✅ Workflow tạo thành công!

{mode_str}
📝 Original: {prompt}
✨ Enhanced: {enhanced}
🚫 Negative: {negative}
🎨 Style: {style}
📐 Size: {width}x{height}
🎲 Seed: {seed}
🤖 Model: {model_config.unet_name} ({model_config.total_size_gb:.1f}GB)
📊 Stats: {stats}
🔍 Valid: {is_valid}
📁 Saved: {output_path}

Errors: {errors if errors else 'None'}
{custom_warn}
"""
        
        workflow_json = json.dumps(workflow, indent=2, ensure_ascii=False)
        
        return info, workflow_json, str(output_path)
    
    except Exception as e:
        import traceback
        return f"❌ Lỗi: {e}\n{traceback.format_exc()}", "", ""

def queue_to_comfy_fn(workflow_json, server_address):
    if not workflow_json:
        return "❌ Chưa có workflow JSON"
    
    try:
        workflow = json.loads(workflow_json)
        client = get_client(server_address, allow_mock=True)
        
        if not client.is_server_running():
            # Mock
            result = client.queue_prompt(workflow)
            return f"🎭 Mock mode (ComfyUI không chạy tại {server_address})\nWorkflow đã lưu mock, kiểm tra thư mục output/\nResult: {result}"
        
        result = client.generate_image(workflow, wait=False)
        return f"✅ Đã queue tới ComfyUI {server_address}\nResult: {json.dumps(result, indent=2)}"
    
    except Exception as e:
        import traceback
        return f"❌ Lỗi queue: {e}\n{traceback.format_exc()}"

def load_workflow_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        return json.dumps(workflow, indent=2, ensure_ascii=False), f"✅ Loaded {file_path} ({len(workflow)} nodes)"
    except Exception as e:
        return "", f"❌ Lỗi load: {e}"

def create_ui():
    if not HAS_GRADIO:
        print("Gradio not available")
        return None
    
    with gr.Blocks(title="mode-ai - FLUX Workflow Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 mode-ai - FLUX.1-schnell GGUF Workflow Generator
        Tạo workflow ComfyUI tối ưu cho FLUX schnell Q5_K_S (tổng <15GB, chạy T4 16GB)
        
        **Workflow chính**: `flux_schnell_gguf_toi_uu.json` - 4 steps + FaceDetailer mặt (SAM) + tay + chân
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Prompt")
                prompt_input = gr.Textbox(label="Prompt", placeholder="1girl, long hair, school uniform, cherry blossoms...", lines=3)
                preset_dropdown = gr.Dropdown(choices=list(PRESET_PROMPTS.keys()), label="Preset prompts", value=None)
                use_preset_checkbox = gr.Checkbox(label="Dùng preset thay vì prompt trên", value=False)
                
                gr.Markdown("""#### 🔧 Chế độ nodes
- **BUILTIN**: `UNETLoader` + `DualCLIPLoader` — **không cần cài custom nodes**, fix lỗi *Missing node type*. Dùng model `flux1-schnell-fp8.safetensors`.
- **GGUF**: `UnetLoaderGGUF` + `DualCLIPLoaderGGUF` — model `flux1-schnell-Q5_K_S.gguf`, nhẹ hơn (~12.3GB), nhưng phải cài **ComfyUI-GGUF**.
""")
                use_builtin_checkbox = gr.Checkbox(label="Dùng BUILTIN (không cần cài custom nodes) - fix lỗi Missing node type", value=True)
                
                with gr.Row():
                    style_dropdown = gr.Dropdown(choices=PromptEnhancer.list_styles(), value="aesthetic_anime", label="Style")
                    workflow_type_dropdown = gr.Dropdown(choices=["simple", "optimized", "realistic", "hires", "inpaint"], value="optimized", label="Workflow type")
                
                with gr.Row():
                    width_slider = gr.Slider(minimum=512, maximum=1536, step=64, value=1024, label="Width")
                    height_slider = gr.Slider(minimum=512, maximum=1536, step=64, value=1024, label="Height")
                
                with gr.Row():
                    seed_input = gr.Number(value=42, label="Seed (0=random)", precision=0)
                    model_preset_dropdown = gr.Dropdown(choices=list(MODEL_PRESETS.keys()), value="q5_optimal", label="Model preset")
                
                generate_btn = gr.Button("🚀 Tạo Workflow", variant="primary")
                
                gr.Markdown("### 📂 Load workflow có sẵn")
                workflow_files = list_workflows()
                existing_wf_dropdown = gr.Dropdown(choices=workflow_files, label="Workflows có sẵn")
                load_btn = gr.Button("📂 Load file")
            
            with gr.Column():
                gr.Markdown("### 📊 Kết quả")
                info_output = gr.Textbox(label="Info", lines=15)
                workflow_json_output = gr.Code(label="Workflow JSON", language="json")
                file_path_output = gr.Textbox(label="Saved path")
                
                gr.Markdown("### 🌐 ComfyUI")
                server_input = gr.Textbox(value="127.0.0.1:8188", label="ComfyUI server")
                queue_btn = gr.Button("📤 Queue tới ComfyUI")
                queue_output = gr.Textbox(label="Queue result", lines=5)
        
        # Examples
        gr.Markdown("### 💡 Ví dụ prompt")
        gr.Examples(
            examples=[
                ["1girl, long silver hair, aqua eyes, school uniform, cherry blossoms, soft sunlight", "girl_cherry", "aesthetic_anime", "optimized", 1024, 1024, 42],
                ["beautiful Vietnamese woman, white ao dai, Hanoi old quarter", "vietnamese_aodai", "realistic_vietnamese", "realistic", 832, 1216, 123],
                ["close-up portrait of beautiful woman, detailed face, bokeh", "portrait_closeup", "photorealistic", "realistic", 832, 1216, 0],
            ],
            inputs=[prompt_input, preset_dropdown, style_dropdown, workflow_type_dropdown, width_slider, height_slider, seed_input],
        )
        
        # Events
        generate_btn.click(
            fn=generate_workflow_fn,
            inputs=[prompt_input, preset_dropdown, style_dropdown, workflow_type_dropdown, width_slider, height_slider, seed_input, model_preset_dropdown, use_preset_checkbox, use_builtin_checkbox],
            outputs=[info_output, workflow_json_output, file_path_output]
        )
        
        load_btn.click(
            fn=load_workflow_file,
            inputs=[existing_wf_dropdown],
            outputs=[workflow_json_output, info_output]
        )
        
        queue_btn.click(
            fn=queue_to_comfy_fn,
            inputs=[workflow_json_output, server_input],
            outputs=[queue_output]
        )
    
    return demo

if __name__ == "__main__":
    if not HAS_GRADIO:
        print("Installing gradio...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio", "requests"])
        import gradio as gr
    
    demo = create_ui()
    if demo:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
