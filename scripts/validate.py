#!/usr/bin/env python3
"""
Validate tất cả workflows trong thư mục workflows/
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mode_ai import FluxWorkflowBuilder

def validate_file(path: Path):
    print(f"\n{'='*60}")
    print(f"🔍 Validating: {path.name}")
    print(f"{'='*60}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return False
    
    builder = FluxWorkflowBuilder()
    is_valid, errors = builder.validate(workflow)
    stats = builder.get_stats(workflow)
    
    print(f"📊 Stats:")
    print(f"   - Total nodes: {stats['total_nodes']}")
    print(f"   - Node types: {', '.join(stats['node_types'])}")
    print(f"   - Has FaceDetailer: {stats['has_face_detailer']}")
    print(f"   - Has SAM: {stats['has_sam']}")
    print(f"   - Has YOLO: {stats['has_yolo']}")
    
    # Check FLUX specific
    for node_id, node in workflow.items():
        ct = node.get('class_type', '')
        inputs = node.get('inputs', {})
        if ct == 'UnetLoaderGGUF':
            print(f"   - UNET: {inputs.get('unet_name')}")
        if ct == 'DualCLIPLoaderGGUF':
            print(f"   - CLIP: {inputs.get('clip_name1')}, {inputs.get('clip_name2')} (type={inputs.get('type')})")
        if ct == 'KSampler':
            print(f"   - KSampler {node_id}: steps={inputs.get('steps')}, cfg={inputs.get('cfg')}, sampler={inputs.get('sampler_name')}/{inputs.get('scheduler')}")
    
    if is_valid:
        print("✅ VALID")
        return True
    else:
        print("❌ INVALID:")
        for err in errors:
            print(f"   - {err}")
        return False

def main():
    workflows_dir = Path(__file__).parent.parent / "workflows"
    json_files = list(workflows_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {workflows_dir}")
        return
    
    print(f"Found {len(json_files)} workflow files in {workflows_dir}")
    
    results = []
    for jf in sorted(json_files):
        ok = validate_file(jf)
        results.append((jf.name, ok))
    
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, ok in results if ok)
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if passed == len(results):
        print("🎉 All workflows valid!")
    else:
        print("⚠️ Some workflows have issues")

if __name__ == "__main__":
    main()
