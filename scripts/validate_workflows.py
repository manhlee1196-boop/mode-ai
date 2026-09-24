#!/usr/bin/env python3
"""Kiểm tra tĩnh toàn bộ workflow FLUX của repo — KHÔNG cần GPU, không cần ComfyUI chạy.

Đối chiếu trực tiếp với spec node trích từ mã nguồn thật (scripts/node_spec.py):
  • class_type có tồn tại không
  • đủ input required chưa (lỗi hay gặp nhất: FaceDetailer thiếu `wildcard`)
  • input thừa / sai tên
  • mọi link trỏ tới node + slot có thật, kiểu dữ liệu khớp
  • giá trị enum hợp lệ (sampler_name, scheduler, type, channel, device_mode...)
  • giá trị số nằm trong min/max của node
  • graph không có chu trình, có node xuất ảnh
  • quy ước tên file model (VD UltralyticsDetectorProvider phải là `bbox/face_yolov8m.pt`)
  • file UI format đọc ngược lại phải ra đúng graph API gốc (round-trip)
  • notebook chỉ tham chiếu những workflow thật sự có trong repo

    python3 scripts/validate_workflows.py
    python3 scripts/validate_workflows.py --models-root /content/ComfyUI/models
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_to_ui import api_to_ui, ui_to_api  # noqa: E402

OUTPUT_NODES = {"SaveImage", "PreviewImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "VHS_VideoCombine"}

# input là "đường nối" chứ không phải widget
CONN_RE = re.compile(r"(MODEL|CLIP|VAE|LATENT|CONDITIONING|IMAGE|MASK|DETECTOR|SAM_MODEL|PIPE|"
                     r"CONTROL_NET|UPSCALE_MODEL|NOISE|GUIDER|SAMPLER|SIGMAS|SEGS|BBOX|HOOK|OPT$)")

# `control_after_generate` là combo do GIAO DIỆN ComfyUI tự chèn sau widget seed
# (xem nodes.py: 'seed': ('INT', {'control_after_generate': True})).
# Nó không nằm trong INPUT_TYPES nhưng là input hợp lệ của KSampler/KSamplerAdvanced.
UI_ONLY_INPUTS = {"control_after_generate"}


class Report:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warns: List[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)


def _is_link(v: Any) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and isinstance(v[0], (str, int)) and isinstance(v[1], int))


def _spec_of(spec: Dict[str, Any], cls: str) -> Dict[str, Any]:
    return spec["nodes"].get(cls, {"required": {}, "optional": {}, "returns": []})


def _check_enum(rep: Report, where: str, name: str, value: Any, ispec: Dict[str, Any],
                spec: Dict[str, Any]) -> None:
    if "choices" in ispec and value not in ispec["choices"]:
        rep.err(f"{where}: `{name}` = {value!r} không hợp lệ (chấp nhận: {ispec['choices'][:8]}...)")
    dyn = ispec.get("dynamic")
    if isinstance(dyn, str) and dyn in spec.get("dynamic_enums", {}):
        allowed = spec["dynamic_enums"][dyn]
        if allowed and value not in allowed:
            rep.err(f"{where}: `{name}` = {value!r} không hợp lệ (chấp nhận: {allowed})")


def _check_range(rep: Report, where: str, name: str, value: Any, ispec: Dict[str, Any]) -> None:
    if isinstance(value, bool):
        return
    if "min" in ispec and isinstance(value, (int, float)) and value < ispec["min"]:
        rep.err(f"{where}: `{name}` = {value} < min {ispec['min']}")
    if "max" in ispec and isinstance(value, (int, float)) and value > ispec["max"]:
        rep.err(f"{where}: `{name}` = {value} > max {ispec['max']}")


def validate_api(prompt: Dict[str, Any], spec: Dict[str, Any], rep: Report,
                 label: str, models_root: Optional[str] = None) -> None:
    known = spec["nodes"]
    schnell = False

    # 1) class_type
    for nid, node in prompt.items():
        where = f"{label}[{nid}]"
        cls = node.get("class_type")
        if not cls:
            rep.err(f"{where}: thiếu class_type")
            continue
        if cls not in known:
            rep.err(f"{where}: class_type `{cls}` không có trong spec "
                    f"(thiếu custom node hoặc sai tên)")
        if "schnell" in json.dumps(node.get("inputs", {})).lower():
            schnell = True

    # 2) inputs
    for nid, node in prompt.items():
        cls = node.get("class_type", "")
        where = f"{label}[{nid}:{cls}]"
        nspec = _spec_of(spec, cls)
        req, opt = nspec["required"], nspec["optional"]
        inputs = node.get("inputs", {})

        for name in req:
            if name not in inputs:
                rep.err(f"{where}: THIẾU input required `{name}`")
        for name, val in inputs.items():
            ispec = req.get(name) or opt.get(name)
            if ispec is None and (req or opt):
                if name not in UI_ONLY_INPUTS:
                    rep.warn(f"{where}: input `{name}` không có trong INPUT_TYPES (thừa/sai tên)")
                continue
            if ispec is None:
                continue
            if _is_link(val):
                src, slot = str(val[0]), int(val[1])
                if src not in prompt:
                    rep.err(f"{where}: `{name}` nối tới node {src} không tồn tại")
                    continue
                rets = _spec_of(spec, prompt[src].get("class_type", ""))["returns"]
                if rets and slot >= len(rets):
                    rep.err(f"{where}: `{name}` lấy slot {slot} của node {src} "
                            f"(chỉ có {len(rets)} output)")
                    continue
                if rets:
                    got = rets[slot]
                    want = ispec.get("type")
                    if want and got not in ("*",) and want not in ("*", got) \
                            and not (want in ("IMAGE_NAME", "CLIP_NAME", "VAE_NAME")):
                        rep.err(f"{where}: `{name}` cần {want} nhưng nhận {got} từ {src}")
            else:
                _check_enum(rep, where, name, val, ispec, spec)
                _check_range(rep, where, name, val, ispec)

    # 3) chu trình
    graph = {nid: {str(v[0]) for v in node.get("inputs", {}).values() if _is_link(v)}
             for nid, node in prompt.items()}
    color: Dict[str, int] = {}

    def dfs(n: str, stack: List[str]) -> bool:
        color[n] = 1
        for m in graph.get(n, ()):
            if m not in graph:
                continue
            if color.get(m) == 1:
                rep.err(f"{label}: graph có chu trình ({' -> '.join(stack + [n, m])})")
                return True
            if color.get(m, 0) == 0 and dfs(m, stack + [n]):
                return True
        color[n] = 2
        return False

    for n in list(graph):
        if color.get(n, 0) == 0:
            dfs(n, [])

    # 4) có node xuất ảnh
    if not any(n.get("class_type") in OUTPUT_NODES for n in prompt.values()):
        rep.warn(f"{label}: không có node xuất ảnh (SaveImage/PreviewImage)")

    # 5) quy ước riêng của Impact Pack / pipeline FLUX
    for nid, node in prompt.items():
        cls = node.get("class_type")
        ins = node.get("inputs", {})
        where = f"{label}[{nid}:{cls}]"
        if cls == "UltralyticsDetectorProvider":
            name = ins.get("model_name", "")
            if not (name.startswith("bbox/") or name.startswith("segm/")):
                rep.err(f"{where}: model_name phải có tiền tố `bbox/` hoặc `segm/` "
                        f"(đang là {name!r}) — nếu không Impact Subpack sẽ không tìm thấy file")
        if cls == "FaceDetailer":
            if float(ins.get("denoise", 0.5)) < 0.05:
                rep.warn(f"{where}: denoise {ins.get('denoise')} quá thấp, gần như không sửa gì")
        if cls == "DualCLIPLoaderGGUF" or cls == "DualCLIPLoader":
            t1, t2 = str(ins.get("clip_name1", "")), str(ins.get("clip_name2", ""))
            if "t5" in (t1 + t2).lower() and ins.get("type") != "flux":
                rep.err(f"{where}: có T5 mà type != 'flux' ({ins.get('type')!r})")
        if cls == "KSampler" and schnell:
            cfg = float(ins.get("cfg", 1.0))
            if abs(cfg - 1.0) > 1e-6:
                rep.warn(f"{where}: FLUX.1-schnell là model distill → cfg nên = 1.0 (đang {cfg})")
            steps = int(ins.get("steps", 4))
            if steps > 8:
                rep.warn(f"{where}: schnell chỉ cần 4 bước; {steps} bước sẽ chậm mà không đẹp hơn")

    # 6) file model có thật không (chỉ khi được cung cấp thư mục models)
    if models_root:
        for nid, node in prompt.items():
            for key, val in node.get("inputs", {}).items():
                if _is_link(val) or not isinstance(val, str):
                    continue
                if not re.search(r"\.(gguf|safetensors|pth|pt|ckpt|bin)$", val):
                    continue
                hits = glob.glob(os.path.join(models_root, "**", os.path.basename(val)), recursive=True)
                if not hits:
                    rep.warn(f"{label}[{nid}:{node.get('class_type')}]: chưa thấy file model "
                             f"`{val}` trong {models_root}")


def validate_ui(ui: Dict[str, Any], spec: Dict[str, Any], rep: Report, label: str) -> None:
    ids = {str(n["id"]) for n in ui.get("nodes", [])}
    seen_links = set()
    for entry in ui.get("links") or []:
        if len(entry) < 5:
            rep.err(f"{label}: link sai cấu trúc: {entry}")
            continue
        lid, src, sslot, tgt, tslot = entry[0], str(entry[1]), entry[2], str(entry[3]), entry[4]
        if lid in seen_links:
            rep.err(f"{label}: link id {lid} trùng lặp")
        seen_links.add(lid)
        if src not in ids:
            rep.err(f"{label}: link {lid} trỏ tới node {src} không tồn tại")
        if tgt not in ids:
            rep.err(f"{label}: link {lid} trỏ vào node {tgt} không tồn tại")
    for node in ui.get("nodes", []):
        for spec_in in node.get("inputs") or []:
            lid = spec_in.get("link")
            if lid is not None and lid not in seen_links:
                rep.err(f"{label}: node {node['id']} input {spec_in.get('name')} "
                        f"tham chiếu link {lid} không tồn tại")
    # round-trip: UI → API phải tái tạo đúng graph gốc (chênh lệch duy nhất được phép
    # là những input bị lược vì trùng default của node, và combo control_after_generate)
    api1 = ui_to_api(ui, spec)
    orig_api = json.loads(json.dumps(api1))
    api2 = ui_to_api(api_to_ui(orig_api, spec), spec)
    for nid, node in orig_api.items():
        other = api2.get(nid)
        if other is None or other["class_type"] != node["class_type"]:
            rep.err(f"{label}: round-trip làm mất/đổi class của node {nid}")
            continue
        nspec = _spec_of(spec, node["class_type"])
        for name, val in node["inputs"].items():
            if other["inputs"].get(name) != val:
                rep.err(f"{label}: round-trip lệch node {nid} input `{name}`: "
                        f"{val!r} → {other['inputs'].get(name)!r}")
        for name, val in other["inputs"].items():
            if name in node["inputs"] or name == "control_after_generate":
                continue
            ispec = nspec["required"].get(name) or nspec["optional"].get(name) or {}
            if val == ispec.get("default"):
                continue
            rep.err(f"{label}: round-trip sinh input lạ `{name}`={val!r} ở node {nid}")
    validate_api(api1, spec, rep, label + "(api)")


def check_notebook_refs(repo: str, rep: Report) -> List[str]:
    """Mọi tên workflow .json được nhắc trong notebook phải tồn tại trong repo."""
    found: List[str] = []
    for nb in glob.glob(os.path.join(repo, "*.ipynb")):
        try:
            data = json.load(open(nb, encoding="utf-8"))
        except Exception as exc:
            rep.err(f"{os.path.basename(nb)}: không đọc được JSON ({exc})")
            continue
        text = "\n".join("".join(c.get("source", [])) for c in data.get("cells", []))
        for name in sorted(set(re.findall(r"[\w./-]*flux[\w./-]*\.json", text, re.I))):
            base = os.path.basename(name)
            found.append(base)
            if not glob.glob(os.path.join(repo, "**", base), recursive=True):
                rep.err(f"{os.path.basename(nb)}: nhắc tới `{base}` nhưng repo không có file này")
    return sorted(set(found))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="workflows/node_spec.json")
    ap.add_argument("--workflows", default="workflows")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--models-root", default=None,
                    help="thư mục models của ComfyUI (nếu có) để kiểm file model tồn tại")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.spec):
        print(f"❌ thiếu spec {args.spec} — chạy scripts/node_spec.py trước", file=sys.stderr)
        return 2
    spec = json.load(open(args.spec, encoding="utf-8"))
    print(f"Spec: {len(spec['nodes'])} node class "
          f"({len(spec.get('generated_from', []))} tệp nguồn)")

    files = sorted(glob.glob(os.path.join(args.workflows, "*.json"))
                   + glob.glob(os.path.join(args.workflows, "ui", "*.json")))
    files = [f for f in files if not f.endswith("node_spec.json")]
    if not files:
        print("❌ không tìm thấy workflow nào", file=sys.stderr)
        return 2

    total_err = total_warn = skipped = 0
    for path in files:
        rep = Report()
        rel = os.path.relpath(path, args.repo)
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as exc:
            print(f"❌ {rel}: JSON lỗi — {exc}")
            total_err += 1
            continue
        if "nodes" in data:
            validate_ui(data, spec, rep, rel)
        elif all(isinstance(v, dict) and "class_type" in v for v in data.values()):
            validate_api(data, spec, rep, rel, args.models_root)
        else:
            # file dữ liệu đi kèm (node_spec.json, prompts.json) — KHÔNG phải workflow
            print(f"·  {rel:<48} bỏ qua (không phải workflow)")
            skipped += 1
            continue

        status = "✅" if not rep.errors else "❌"
        print(f"{status} {rel:<48} "
              f"{len(data.get('nodes', data)) if isinstance(data, dict) else 0:>3} node  "
              f"{len(rep.errors)} lỗi, {len(rep.warns)} cảnh báo")
        for e in rep.errors:
            print(f"     ✗ {e}")
        for w in rep.warns:
            if not args.quiet:
                print(f"     ⚠ {w}")
        total_err += len(rep.errors)
        total_warn += len(rep.warns)

    rep = Report()
    refs = check_notebook_refs(args.repo, rep)
    print(f"\nNotebook tham chiếu {len(refs)} workflow: {', '.join(refs) if refs else '(không)'}")
    for e in rep.errors:
        print(f"     ✗ {e}")
    total_err += len(rep.errors)

    print(f"\n{'=' * 60}\nTỔNG: {total_err} lỗi, {total_warn} cảnh báo "
          f"trên {len(files) - skipped} workflow")
    return 1 if total_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
