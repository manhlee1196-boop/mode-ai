#!/usr/bin/env python3
"""
Auto-install ComfyUI custom nodes cần thiết cho FLUX workflows
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Danh sách custom nodes cần thiết
REQUIRED_NODES = {
    "ComfyUI-GGUF": {
        "repo": "https://github.com/city96/ComfyUI-GGUF.git",
        "description": "UnetLoaderGGUF, DualCLIPLoaderGGUF - Bắt buộc cho GGUF workflows (2 nodes lỗi của bạn)",
        "required_for": ["flux_schnell_simple.json", "flux_schnell_gguf_toi_uu.json", "flux_schnell_realistic.json", "flux_schnell_realistic_hires.json", "flux_schnell_inpaint.json"],
        "pip_requirements": True,
    },
    "ComfyUI-Impact-Pack": {
        "repo": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "description": "FaceDetailer, SAMLoader - Cho workflow tối ưu mặt/tay/chân",
        "required_for": ["flux_schnell_gguf_toi_uu.json", "flux_builtin_fp8_toi_uu.json", "flux_schnell_realistic.json"],
        "pip_requirements": True,
        "install_script": "install.py",
    },
    "ComfyUI-Impact-Subpack": {
        "repo": "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git",
        "description": "UltralyticsDetectorProvider - YOLO face/hand/foot detector",
        "required_for": ["flux_schnell_gguf_toi_uu.json", "flux_builtin_fp8_toi_uu.json"],
        "pip_requirements": True,
    },
    "ComfyUI_UltimateSDUpscale": {
        "repo": "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git",
        "description": "UltimateSDUpscale - Cho hires workflow (optional)",
        "required_for": ["flux_schnell_realistic_hires.json"],
        "pip_requirements": False,
    },
    "ComfyUI-Manager": {
        "repo": "https://github.com/ltdrdata/ComfyUI-Manager.git",
        "description": "ComfyUI Manager - Quản lý custom nodes dễ dàng",
        "required_for": ["Tất cả - khuyến nghị cài để dễ quản lý"],
        "pip_requirements": False,
        "optional": True,
    }
}

def run_cmd(cmd, cwd=None):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0

def install_node(node_name, node_info, custom_nodes_dir: Path, pip_cmd="pip"):
    print(f"\n{'='*60}")
    print(f"📦 Installing {node_name}")
    print(f"   {node_info['description']}")
    print(f"   Repo: {node_info['repo']}")
    print(f"{'='*60}")
    
    node_dir = custom_nodes_dir / node_name
    
    if node_dir.exists():
        print(f"⚠️ {node_name} đã tồn tại tại {node_dir}")
        print("   Pulling latest...")
        run_cmd("git pull", cwd=str(node_dir))
    else:
        print(f"⬇️ Cloning {node_name}...")
        if not run_cmd(f"git clone {node_info['repo']} {node_name}", cwd=str(custom_nodes_dir)):
            print(f"❌ Failed to clone {node_name}")
            return False
    
    # pip install requirements
    if node_info.get("pip_requirements"):
        req_file = node_dir / "requirements.txt"
        if req_file.exists():
            print(f"📦 Installing requirements for {node_name}...")
            run_cmd(f"{pip_cmd} install -r requirements.txt", cwd=str(node_dir))
    
    # Run install.py if exists
    install_script = node_info.get("install_script")
    if install_script:
        script_path = node_dir / install_script
        if script_path.exists():
            print(f"🔧 Running {install_script} for {node_name}...")
            run_cmd(f"{sys.executable} {install_script}", cwd=str(node_dir))
    
    print(f"✅ {node_name} installed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Cài đặt ComfyUI custom nodes cho FLUX workflows")
    parser.add_argument("--comfyui-path", type=str, default=None, help="Đường dẫn tới ComfyUI (ví dụ: /content/ComfyUI hoặc ./ComfyUI)")
    parser.add_argument("--only-gguf", action="store_true", help="Chỉ cài ComfyUI-GGUF (fix 2 nodes lỗi)")
    parser.add_argument("--skip-manager", action="store_true", help="Bỏ qua ComfyUI-Manager")
    parser.add_argument("--pip", type=str, default="pip", help="pip command")
    parser.add_argument("--list", action="store_true", help="Liệt kê các nodes cần thiết")
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 Các custom nodes cần thiết:")
        for name, info in REQUIRED_NODES.items():
            print(f"\n- {name}:")
            print(f"  Mô tả: {info['description']}")
            print(f"  Repo: {info['repo']}")
            print(f"  Dùng cho: {', '.join(info['required_for'])}")
        return
    
    # Tìm ComfyUI path
    comfyui_path = None
    if args.comfyui_path:
        comfyui_path = Path(args.comfyui_path)
    else:
        # Tự động tìm
        candidates = [
            Path("/content/ComfyUI"),
            Path("./ComfyUI"),
            Path("../ComfyUI"),
            Path("~/ComfyUI").expanduser(),
            Path("ComfyUI"),
        ]
        for cand in candidates:
            if cand.exists() and (cand / "main.py").exists():
                comfyui_path = cand
                break
    
    if not comfyui_path or not comfyui_path.exists():
        print("❌ Không tìm thấy ComfyUI!")
        print("   Hãy chỉ định đường dẫn: --comfyui-path /path/to/ComfyUI")
        print("   Hoặc chạy trong thư mục chứa ComfyUI")
        # Vẫn tiếp tục với thư mục hiện tại nếu không tìm thấy, để user có thể cài manual
        if args.comfyui_path:
            comfyui_path = Path(args.comfyui_path)
            comfyui_path.mkdir(parents=True, exist_ok=True)
            (comfyui_path / "custom_nodes").mkdir(exist_ok=True)
        else:
            print("\n💡 Nếu bạn chưa có ComfyUI, cài đặt:")
            print("   git clone https://github.com/comfyanonymous/ComfyUI.git")
            print("   Sau đó chạy lại script này")
            return
    
    custom_nodes_dir = comfyui_path / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 ComfyUI path: {comfyui_path}")
    print(f"📁 Custom nodes dir: {custom_nodes_dir}")
    
    # Xác định nodes cần cài
    nodes_to_install = {}
    if args.only_gguf:
        nodes_to_install["ComfyUI-GGUF"] = REQUIRED_NODES["ComfyUI-GGUF"]
        print("\n🎯 Chỉ cài ComfyUI-GGUF để fix 2 nodes lỗi")
    else:
        nodes_to_install = REQUIRED_NODES.copy()
        if args.skip_manager:
            nodes_to_install.pop("ComfyUI-Manager", None)
    
    # Cài từng node
    success = []
    failed = []
    
    for name, info in nodes_to_install.items():
        try:
            if install_node(name, info, custom_nodes_dir, pip_cmd=args.pip):
                success.append(name)
            else:
                failed.append(name)
        except Exception as e:
            print(f"❌ Lỗi cài {name}: {e}")
            failed.append(name)
    
    print(f"\n{'='*60}")
    print("📊 Kết quả cài đặt")
    print(f"{'='*60}")
    print(f"✅ Thành công ({len(success)}): {', '.join(success)}")
    if failed:
        print(f"❌ Thất bại ({len(failed)}): {', '.join(failed)}")
    
    print(f"\n🔄 Hãy restart ComfyUI:")
    print(f"   cd {comfyui_path}")
    print(f"   python main.py --enable-manager")
    
    print(f"\n💡 Sau khi restart, load lại workflow JSON trong workflows/")
    print(f"   - Nếu chỉ cài GGUF: dùng flux_schnell_simple.json (đã fix 2 nodes lỗi)")
    print(f"   - Nếu cài đầy đủ: dùng flux_schnell_gguf_toi_uu.json (tối ưu)")
    print(f"   - Nếu không muốn cài gì: dùng flux_builtin_fp8_simple.json (0 custom nodes)")

if __name__ == "__main__":
    main()
