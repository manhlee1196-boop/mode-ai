#!/bin/bash
# Auto-install ComfyUI custom nodes cho FLUX workflows
# Fix lỗi: 2 nodes affected - Missing node type

set -e

COMFYUI_PATH=${1:-""}
ONLY_GGUF=${2:-""}

# Tìm ComfyUI path nếu không chỉ định
if [ -z "$COMFYUI_PATH" ]; then
    for candidate in "/content/ComfyUI" "./ComfyUI" "../ComfyUI" "$HOME/ComfyUI" "ComfyUI"; do
        if [ -f "$candidate/main.py" ]; then
            COMFYUI_PATH=$candidate
            break
        fi
    done
fi

if [ -z "$COMFYUI_PATH" ] || [ ! -d "$COMFYUI_PATH" ]; then
    echo "❌ Không tìm thấy ComfyUI!"
    echo "   Sử dụng: bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI"
    echo "   Hoặc: bash scripts/install_comfyui_nodes.sh /path/to/ComfyUI --only-gguf (chỉ fix 2 nodes lỗi)"
    exit 1
fi

CUSTOM_NODES_DIR="$COMFYUI_PATH/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"
cd "$CUSTOM_NODES_DIR"

echo "📁 ComfyUI: $COMFYUI_PATH"
echo "📁 Custom nodes: $CUSTOM_NODES_DIR"

install_node() {
    local name=$1
    local repo=$2
    echo ""
    echo "============================================================"
    echo "📦 Installing $name"
    echo "   Repo: $repo"
    echo "============================================================"
    
    if [ -d "$name" ]; then
        echo "⚠️ $name đã tồn tại, pulling latest..."
        cd "$name"
        git pull || echo "⚠️ git pull failed, bỏ qua"
        cd ..
    else
        echo "⬇️ Cloning $name..."
        git clone "$repo" "$name" || {
            echo "❌ Failed to clone $name"
            return 1
        }
    fi
    
    # pip install requirements nếu có
    if [ -f "$name/requirements.txt" ]; then
        echo "📦 Installing requirements for $name..."
        pip install -r "$name/requirements.txt" || pip3 install -r "$name/requirements.txt" || echo "⚠️ pip install failed"
    fi
    
    # install.py nếu có
    if [ -f "$name/install.py" ]; then
        echo "🔧 Running install.py for $name..."
        python "$name/install.py" || python3 "$name/install.py" || echo "⚠️ install.py failed"
    fi
    
    echo "✅ $name installed"
}

# Fix 2 nodes lỗi chính: ComfyUI-GGUF
echo ""
echo "🎯 Cài ComfyUI-GGUF để fix 2 nodes lỗi: UnetLoaderGGUF, DualCLIPLoaderGGUF"
install_node "ComfyUI-GGUF" "https://github.com/city96/ComfyUI-GGUF.git"

# Nếu chỉ cần fix 2 nodes, dừng ở đây
if [ "$ONLY_GGUF" = "--only-gguf" ]; then
    echo ""
    echo "✅ Đã fix 2 nodes lỗi! (ComfyUI-GGUF)"
    echo "🔄 Restart ComfyUI: cd $COMFYUI_PATH && python main.py --enable-manager"
    echo "📂 Load workflow: workflows/flux_schnell_simple.json"
    exit 0
fi

# Cài thêm các nodes khác cho workflow tối ưu
echo ""
echo "📦 Cài thêm Impact Pack cho workflow tối ưu..."
install_node "ComfyUI-Impact-Pack" "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
install_node "ComfyUI-Impact-Subpack" "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git"
install_node "ComfyUI_UltimateSDUpscale" "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git"
install_node "ComfyUI-Manager" "https://github.com/ltdrdata/ComfyUI-Manager.git"

echo ""
echo "============================================================"
echo "📊 Hoàn tất cài đặt!"
echo "============================================================"
echo "✅ Đã cài:"
echo "   - ComfyUI-GGUF (fix 2 nodes lỗi)"
echo "   - ComfyUI-Impact-Pack (FaceDetailer, SAMLoader)"
echo "   - ComfyUI-Impact-Subpack (UltralyticsDetectorProvider)"
echo "   - ComfyUI_UltimateSDUpscale (UltimateSDUpscale)"
echo "   - ComfyUI-Manager (quản lý nodes)"
echo ""
echo "🔄 Restart ComfyUI:"
echo "   cd $COMFYUI_PATH"
echo "   python main.py --enable-manager"
echo ""
echo "📂 Sau restart, load workflow:"
echo "   - Simple (đã fix): workflows/flux_schnell_simple.json"
echo "   - Tối ưu đầy đủ: workflows/flux_schnell_gguf_toi_uu.json"
echo "   - Không cần custom nodes: workflows/flux_builtin_fp8_simple.json"
