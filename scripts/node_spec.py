#!/usr/bin/env python3
"""Trích xuất "node spec" (INPUT_TYPES / RETURN_TYPES) trực tiếp từ mã nguồn ComfyUI
và custom node, để workflow JSON có thể được kiểm tra tĩnh mà KHÔNG cần GPU.

Nguồn spec thật (không phải bảng kê khai viết tay):
  - ComfyUI:            nodes.py, comfy_extras/*.py
  - ComfyUI-GGUF:       nodes.py
  - ComfyUI-Impact-Pack:modules/impact/*.py
  - ComfyUI-Impact-Subpack: modules/*.py

Cách dùng:
    python3 scripts/node_spec.py --comfy /tmp/ComfyUI --gguf /tmp/ComfyUI-GGUF \
        --impact /tmp/Impact-Pack --subpack /tmp/Impact-Subpack -o workflows/node_spec.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

# Các danh sách enum động (không phải literal trong INPUT_TYPES) nhưng cần validate.
# Được đọc trực tiếp từ comfy/samplers.py khi có nguồn ComfyUI.
DYNAMIC_ENUM_SOURCES = {
    "comfy.samplers.KSampler.SAMPLERS": ("comfy/samplers.py", "SAMPLER_NAMES"),
    "comfy.samplers.KSampler.SCHEDULERS": ("comfy/samplers.py", "SCHEDULER_NAMES"),
}

# Vài node khai báo INPUT_TYPES bằng hàm gọi (folder_paths...) nên element[0] là Call.
# Khi đó ta chỉ ghi được tên input; kiểu dữ liệu lấy từ bảng fallback tối thiểu này.
# (Chỉ dùng khi không trích xuất được từ nguồn — KHÔNG bịa input không có thật.)
FALLBACK_TYPES = {
    "VAELoader": {"vae_name": "VAE_NAME"},
    "DualCLIPLoader": {"clip_name1": "CLIP_NAME", "clip_name2": "CLIP_NAME", "type": "CLIP_TYPE"},
    "LoadImage": {"image": "IMAGE_NAME"},
}


def _name_of(node: ast.AST) -> Optional[str]:
    return node.id if isinstance(node, ast.Name) else None


def _io_type_map(comfy_dir: Optional[str]) -> Dict[str, str]:
    """`@comfytype(io_type="LATENT") class Latent` → {'Latent': 'LATENT'} (API V3 của ComfyUI)."""
    out: Dict[str, str] = {}
    if not comfy_dir:
        return out
    path = os.path.join(comfy_dir, "comfy_api", "latest", "_io.py")
    if not os.path.isfile(path):
        return out
    for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if not isinstance(node, ast.ClassDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "io_type" and isinstance(kw.value, ast.Constant):
                        out[node.name] = str(kw.value.value)
    return out


def _v3_type(func: ast.AST, io_map: Dict[str, str]) -> str:
    """IO.Image.Input(...) → 'IMAGE'."""
    chain: List[str] = []
    cur = func
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    chain.append(getattr(cur, "id", ""))
    # chain = ['Input', 'Image', 'IO'] (ngược)
    cls = next((c for c in chain if c in io_map), None)
    if cls:
        return io_map[cls]
    for c in chain:
        if c and c not in ("IO", "Input", "Output", "Schema"):
            return c.upper()
    return "*"


def extract_v3(path: str, io_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Trích node khai báo theo API V3: class X(IO.ComfyNode): define_schema() → IO.Schema(...)."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "define_schema"):
                continue
            call = None
            for stmt in ast.walk(fn):
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute) \
                        and stmt.func.attr == "Schema":
                    call = stmt
                    break
            if call is None:
                continue
            spec: Dict[str, Any] = {"required": {}, "optional": {}, "returns": []}
            node_id = cls.name
            for kw in call.keywords:
                if kw.arg == "node_id" and isinstance(kw.value, ast.Constant):
                    node_id = str(kw.value.value)
                elif kw.arg == "inputs" and isinstance(kw.value, ast.List):
                    for item in kw.value.elts:
                        if not isinstance(item, ast.Call):
                            continue
                        name = None
                        if item.args and isinstance(item.args[0], ast.Constant):
                            name = str(item.args[0].value)
                        ispec: Dict[str, Any] = {"type": _v3_type(item.func, io_map)}
                        for k2 in item.keywords:
                            if k2.arg == "options":
                                opts = _const(k2.value)
                                if isinstance(opts, list) and all(isinstance(o, str) for o in opts):
                                    ispec = {"choices": opts}
                            elif k2.arg in ("default", "min", "max", "step"):
                                val = _const(k2.value)
                                if val is not None:
                                    ispec[k2.arg] = val
                        optional = any(k2.arg == "optional" and _const(k2.value) is True
                                       for k2 in item.keywords)
                        if name:
                            spec["optional" if optional else "required"][name] = ispec
                elif kw.arg == "outputs" and isinstance(kw.value, ast.List):
                    for item in kw.value.elts:
                        if isinstance(item, ast.Call):
                            spec["returns"].append(_v3_type(item.func, io_map))
            if spec["required"] or spec["optional"] or spec["returns"]:
                out[node_id] = spec
                if cls.name != node_id:
                    out[cls.name] = spec
    return out


def _const(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _input_spec(value: ast.AST) -> Dict[str, Any]:
    """(TYPE, {opts}) -> {'type': ...} | {'choices': [...]} (+ default/min/max)."""
    out: Dict[str, Any] = {}
    if isinstance(value, ast.Tuple) and value.elts:
        head = value.elts[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            out["type"] = head.value
        elif isinstance(head, (ast.List, ast.Tuple)):
            choices = [_const(e) for e in head.elts]
            if all(isinstance(c, str) for c in choices):
                out["choices"] = choices
            else:
                out["dynamic"] = "list"
        elif isinstance(head, (ast.Name, ast.Attribute)):
            out["dynamic"] = ast.unparse(head)
        elif isinstance(head, ast.Call):
            out["dynamic"] = "call"
        else:
            out["dynamic"] = "unknown"
        if len(value.elts) > 1 and isinstance(value.elts[1], ast.Dict):
            for k, v in zip(value.elts[1].keys, value.elts[1].values):
                key = _const(k)
                if key in ("default", "min", "max", "step"):
                    val = _const(v)
                    if val is not None:
                        out[key] = val
    else:
        out["dynamic"] = "unknown"
    return out


def _section(spec: Dict[str, Any], name: str, node: ast.AST) -> None:
    if not isinstance(node, ast.Dict):
        return
    for k, v in zip(node.keys, node.values):
        key = _const(k)
        if not isinstance(key, str):
            continue
        # giá trị có thể là ("TYPE", {...}), (["a","b"],) hoặc 1 biến/hàm gọi (danh sách file)
        spec[name][key] = _input_spec(v)


def _parse_dict_return(fn: ast.FunctionDef) -> Optional[ast.Dict]:
    for stmt in reversed(fn.body):
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
            return stmt.value
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict):
            # kiểu: base = {...}; base["required"].update(...); return base
            for k in stmt.value.keys:
                if _const(k) == "required":
                    return stmt.value
    # quét mọi Dict được gán/return trong thân hàm
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Dict) and any(_const(k) == "required" for k in stmt.keys):
            return stmt
    return None


def extract_classes(path: str) -> Dict[str, Dict[str, Any]]:
    """Trả về {class_name: {'required': {...}, 'optional': {...}, 'returns': [...]}}."""
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as exc:  # pragma: no cover
        print(f"  ! bỏ qua {path}: {exc}", file=sys.stderr)
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        spec: Dict[str, Any] = {"required": {}, "optional": {}, "returns": []}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                tgt = _name_of(stmt.targets[0])
                if tgt == "RETURN_TYPES":
                    vals = _const(stmt.value)
                    if isinstance(vals, (list, tuple)):
                        spec["returns"] = [v for v in vals if isinstance(v, str)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.target.id == "RETURN_TYPES" and stmt.value is not None:
                    vals = _const(stmt.value)
                    if isinstance(vals, (list, tuple)):
                        spec["returns"] = [v for v in vals if isinstance(v, str)]
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "INPUT_TYPES":
                d = _parse_dict_return(stmt)
                if d is None:
                    continue
                for k, v in zip(d.keys, d.values):
                    key = _const(k)
                    if key == "required":
                        _section(spec, "required", v)
                    elif key == "optional":
                        _section(spec, "optional", v)
        if spec["required"] or spec["optional"]:
            # node con kế thừa INPUT_TYPES của node cha → gộp spec cha nếu biết
            out[node.name] = spec

    # kế thừa: node con tự khai báo lại `required` (đầy đủ) nên CHỈ kế thừa optional + RETURN_TYPES.
    # (nếu gộp cả required của cha sẽ sinh báo động giả, VD DualCLIPLoaderGGUF ← CLIPLoaderGGUF)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in out:
            for base in node.bases:
                bname = base.id if isinstance(base, ast.Name) else None
                if bname and bname in out and bname != node.name:
                    parent = out[bname]
                    for k, v in parent["optional"].items():
                        out[node.name]["optional"].setdefault(k, v)
                    if not out[node.name]["returns"]:
                        out[node.name]["returns"] = parent["returns"]
    return out


def _dynamic_enums(comfy_dir: Optional[str]) -> Dict[str, List[str]]:
    enums: Dict[str, List[str]] = {}
    if not comfy_dir:
        return enums
    for expr, (rel, var) in DYNAMIC_ENUM_SOURCES.items():
        path = os.path.join(comfy_dir, rel)
        if not os.path.isfile(path):
            continue
        tree = ast.parse(open(path, encoding="utf-8").read())
        # gom mọi list literal + key của dict literal gán cho biến (kể cả A = B + [...])
        def collect(value: ast.AST) -> List[str]:
            found: List[str] = []
            for sub in ast.walk(value):
                if isinstance(sub, (ast.List, ast.Tuple)):
                    lit = _const(sub)
                    if isinstance(lit, (list, tuple)):
                        found += [x for x in lit if isinstance(x, str)]
                elif isinstance(sub, ast.Dict):
                    found += [k for k in (_const(x) for x in sub.keys) if isinstance(k, str)]
                elif isinstance(sub, ast.Name):  # tham chiếu biến khác trong cùng module
                    for other in tree.body:
                        if isinstance(other, ast.Assign) and _name_of(other.targets[0]) == sub.id:
                            found += collect(other.value)
            return found

        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and _name_of(stmt.targets[0]) == var:
                vals = sorted(set(collect(stmt.value)))
                if vals:
                    enums[expr] = vals
    return enums


def build_spec(comfy: Optional[str], gguf: Optional[str], impact: Optional[str],
               subpack: Optional[str]) -> Dict[str, Any]:
    nodes: Dict[str, Any] = {}
    sources: List[str] = []

    io_map = _io_type_map(comfy)

    def add_dir(root: Optional[str], rels: Iterable[str], label: str) -> None:
        if not root:
            return
        for rel in rels:
            p = os.path.join(root, rel)
            if not os.path.isfile(p):
                continue
            got = extract_classes(p)
            got_v3 = extract_v3(p, io_map)
            for cls, spec in got_v3.items():
                old = got.get(cls)
                # API V3 đầy đủ hơn (có cả kiểu dữ liệu của input động) → ưu tiên
                if old is None or len(spec["required"]) + len(spec["optional"]) >= \
                        len(old["required"]) + len(old["optional"]):
                    got[cls] = spec
            if got:
                sources.append(f"{label}:{rel} ({len(got)} node)")
            for cls, spec in got.items():
                # không ghi đè spec đầy đủ hơn
                old = nodes.get(cls)
                if old is None or len(spec["required"]) + len(spec["optional"]) > \
                        len(old["required"]) + len(old["optional"]):
                    nodes[cls] = spec

    if comfy:
        add_dir(comfy, ["nodes.py"], "ComfyUI")
        extras = os.path.join(comfy, "comfy_extras")
        if os.path.isdir(extras):
            add_dir(comfy, [f"comfy_extras/{f}" for f in sorted(os.listdir(extras)) if f.endswith(".py")], "ComfyUI")
    if gguf:
        add_dir(gguf, ["nodes.py"], "ComfyUI-GGUF")
    if impact:
        files = ["__init__.py"]
        mod = os.path.join(impact, "modules", "impact")
        if os.path.isdir(mod):
            files += [f"modules/impact/{f}" for f in sorted(os.listdir(mod)) if f.endswith(".py")]
        add_dir(impact, files, "ComfyUI-Impact-Pack")
    if subpack:
        files = ["__init__.py"]
        mod = os.path.join(subpack, "modules")
        if os.path.isdir(mod):
            files += [f"modules/{f}" for f in sorted(os.listdir(mod)) if f.endswith(".py")]
        add_dir(subpack, files, "ComfyUI-Impact-Subpack")

    # bổ sung kiểu dữ liệu tối thiểu cho node khai báo INPUT_TYPES bằng hàm gọi
    for cls, table in FALLBACK_TYPES.items():
        if cls in nodes:
            for name, typ in table.items():
                nodes[cls]["required"].setdefault(name, {"type": typ})

    return {
        "generated_from": sources,
        "dynamic_enums": _dynamic_enums(comfy),
        "nodes": nodes,
    }


def load_spec(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", help="thư mục ComfyUI đã clone")
    ap.add_argument("--gguf", help="thư mục ComfyUI-GGUF đã clone")
    ap.add_argument("--impact", help="thư mục ComfyUI-Impact-Pack đã clone")
    ap.add_argument("--subpack", help="thư mục ComfyUI-Impact-Subpack đã clone")
    ap.add_argument("-o", "--out", default="workflows/node_spec.json")
    args = ap.parse_args()

    if not any([args.comfy, args.gguf, args.impact, args.subpack]):
        print("cần ít nhất 1 đường dẫn nguồn (--comfy/--gguf/--impact/--subpack)", file=sys.stderr)
        return 2

    spec = build_spec(args.comfy, args.gguf, args.impact, args.subpack)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        # KHÔNG sort_keys: thứ tự input phải giữ nguyên theo INPUT_TYPES (dùng cho UI widgets_values)
        json.dump(spec, fh, ensure_ascii=False, indent=1)
    print(f"✅ {len(spec['nodes'])} node class → {args.out}")
    for line in spec["generated_from"]:
        print("   -", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
