#!/usr/bin/env python3
"""
Validate tất cả workflows trong thư mục workflows/
Phân loại BUILTIN (không cần custom nodes) và GGUF (cần cài)
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mode_ai import FluxWorkflowBuilder

def validate_file(path: Path):
    print(f"\n{'='*70}")
    print(f"🔍 Validating: {path.name}")
    print(f"{'='*70}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return False, {}
    
    builder = FluxWorkflowBuilder()
    is_valid, errors = builder.validate(workflow)
    stats = builder.get_stats(workflow)
    
    print(f"📊 Stats:")
    print(f"   - Total nodes: {stats['total_nodes']}")
    print(f"   - Node types: {', '.join(stats['node_types'])}")
    print(f"   - Has FaceDetailer: {stats['has_face_detailer']}")
    print(f"   - Has SAM: {stats['has_sam']}")
    print(f"   - Has YOLO: {stats['has_yolo']}")
    print(f"   - Has GGUF: {stats['has_gguf']} {stats['gguf_nodes']}")
    print(f"   - Impact nodes: {stats['impact_nodes']}")
    print(f"   - BUILTIN only (0 custom nodes): {stats['builtin_only']}")
    if stats['custom_nodes_required']:
        print(f"   - Custom nodes required: {stats['custom_nodes_required']}")
    else:
        print(f"   - ✅ No custom nodes needed - chạy ngay!")
    
    # Check FLUX specific
    for node_id, node in workflow.items():
        ct = node.get('class_type', '')
        inputs = node.get('inputs', {})
        if ct in ['UnetLoaderGGUF', 'UNETLoader']:
            print(f"   - UNET: {inputs.get('unet_name')} ({ct})")
        if ct in ['DualCLIPLoaderGGUF', 'DualCLIPLoader']:
            print(f"   - CLIP: {inputs.get('clip_name1')}, {inputs.get('clip_name2')} (type={inputs.get('type')}, {ct})")
        if ct == 'KSampler':
            print(f"   - KSampler {node_id}: steps={inputs.get('steps')}, cfg={inputs.get('cfg')}, sampler={inputs.get('sampler_name')}/{inputs.get('scheduler')}")
    
    if is_valid:
        print("✅ VALID")
        # Show install guide if needed
        if stats['custom_nodes_required']:
            print(f"\n📦 Install guide:")
            print(builder.get_install_guide(workflow))
    else:
        print("❌ INVALID:")
        for err in errors:
            print(f"   - {err}")
    
    return is_valid, stats

def main():
    workflows_dir = Path(__file__).parent.parent / "workflows"
    json_files = list(workflows_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {workflows_dir}")
        return
    
    print(f"Found {len(json_files)} workflow files in {workflows_dir}")
    
    results = []
    builtin_count = 0
    gguf_count = 0
    
    for jf in sorted(json_files):
        ok, stats = validate_file(jf)
        results.append((jf.name, ok, stats))
        if stats.get('builtin_only'):
            builtin_count += 1
        if stats.get('has_gguf'):
            gguf_count += 1
    
    print(f"\n{'='*70}")
    print("📋 SUMMARY")
    print(f"{'='*70}")
    for name, ok, stats in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        custom = "BUILTIN (0 custom)" if stats.get('builtin_only') else f"Needs {stats.get('custom_nodes_required')}"
        print(f"{status} - {name} - {custom}")
    
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nTotal: {passed}/{len(results)} passed")
    print(f"BUILTIN (no custom nodes needed): {builtin_count} workflows")
    print(f"GGUF (needs ComfyUI-GGUF): {gguf_count} workflows")
    
    if passed == len(results):
        print("🎉 All workflows valid!")
    
    print(f"\n💡 Fix lỗi '2 nodes affected - Missing node type':")
    print(f"   - Nguyên nhân: Chưa cài ComfyUI-GGUF (UnetLoaderGGUF, DualCLIPLoaderGGUF)")
    print(f"   - Fix nhanh: bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI --only-gguf")
    print(f"   - Hoặc dùng workflow BUILTIN không cần cài gì:")
    print(f"     workflows/flux_builtin_fp8_simple.json (0 custom nodes)")

if __name__ == "__main__":
    main()
