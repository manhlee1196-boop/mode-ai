#!/usr/bin/env python3
"""Chuyển workflow ComfyUI giữa 2 định dạng:

  API format  : {"7": {"class_type": "KSampler", "inputs": {...}}}   → dùng cho POST /prompt
  UI  format  : {"nodes": [...], "links": [...], "version": 0.4}     → kéo-thả vào giao diện

    python3 scripts/api_to_ui.py workflows/flux_q5_quality.json -o workflows/ui/flux_q5_quality.json
    python3 scripts/api_to_ui.py --to-api workflows/ui/flux_q5_quality.json   # chiều ngược lại (để kiểm tra)

Lưu ý: phần toạ độ/thứ tự widget là "best effort" — kết nối (link) và tham số thì chính xác
tuyệt đối, vì `--to-api` đọc ngược lại phải ra đúng graph gốc (được validate_workflows.py kiểm).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

LINK_TYPES = {
    "*", "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "IMAGE", "MASK",
    "BBOX_DETECTOR", "SEGM_DETECTOR", "SAM_MODEL", "DETAILER_PIPE", "CONTROL_NET",
    "UPSCALE_MODEL", "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "INT", "FLOAT",
}

# Node có widget seed mà giao diện ComfyUI tự chèn thêm combo "control_after_generate"
SEED_CONTROL_CLASSES = {"KSampler", "KSamplerAdvanced", "KSamplerSelect", "RepeatLatentBatch"}

WIDGET_DEFAULT = {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False, "STRING": ""}

# input là ô nhập liệu (widget) hay là "ổ cắm" nối dây (socket)
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "SAMPLER", "SCHEDULER",
                "VAE_NAME", "CLIP_NAME", "IMAGE_NAME", "CLIP_TYPE"}


def is_widget(name: str, ispec: Dict[str, Any]) -> bool:
    """Phân biệt widget (nằm trong widgets_values) và socket (nằm trong inputs, link=null)."""
    if not ispec:
        return True
    if "choices" in ispec:
        return True
    if ispec.get("type") in WIDGET_TYPES:
        return True
    # danh sách file model (dynamic/call) cũng là widget; còn *_opt/*_ref là socket
    if "dynamic" in ispec and not name.endswith(("_opt", "_ref")):
        return True
    return False

_MISSING = object()


def _is_link(value: Any) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], (str, int)) and isinstance(value[1], int))


# --------------------------------------------------------------------------- API → UI
def api_to_ui(prompt: Dict[str, Any], spec: Optional[Dict[str, Any]] = None,
              title_prefix: str = "") -> Dict[str, Any]:
    nodes_spec = (spec or {}).get("nodes", {})
    ids = sorted(prompt.keys(), key=lambda x: (len(x), x))

    # độ sâu topo để xếp cột
    depth: Dict[str, int] = {}

    def depth_of(nid: str, seen: Optional[set] = None) -> int:
        seen = seen or set()
        if nid in depth:
            return depth[nid]
        if nid in seen:
            return 0
        seen.add(nid)
        deps = [v[0] for v in prompt.get(nid, {}).get("inputs", {}).values()
                if _is_link(v) and str(v[0]) in prompt]
        d = 0 if not deps else 1 + max(depth_of(str(x), seen) for x in deps)
        depth[nid] = d
        return d

    for nid in ids:
        depth_of(nid)

    ordered = sorted(ids, key=lambda n: (depth[n], int(n) if n.isdigit() else 0))

    links: List[List[Any]] = []
    ui_nodes: List[Dict[str, Any]] = []
    link_id = 1
    # map (target_node, input_name) -> link_id
    pending: Dict[Tuple[str, str], int] = {}

    for nid in ordered:
        depth_of(nid)
    # tạo link trước để node input/output tham chiếu đúng id
    for nid in ordered:
        for name, val in prompt[nid].get("inputs", {}).items():
            if _is_link(val):
                pending[(nid, name)] = link_id
                links.append([link_id, str(val[0]), int(val[1]), nid, 0, "*"])
                link_id += 1

    col_rows: Dict[int, int] = {}
    for nid in ordered:
        cls = prompt[nid].get("class_type", "?")
        inputs = prompt[nid].get("inputs", {})
        nspec = nodes_spec.get(cls, {})
        order_keys = list(nspec.get("required", {})) + list(nspec.get("optional", {}))
        for k in inputs:
            if k not in order_keys:
                order_keys.append(k)

        ui_inputs: List[Dict[str, Any]] = []
        widgets: List[Any] = []
        for name in order_keys:
            ispec = (nspec.get("required", {}).get(name)
                     or nspec.get("optional", {}).get(name) or {})
            val = inputs.get(name, _MISSING)
            if val is not _MISSING and _is_link(val):
                lid = pending.get((nid, name))
                src = prompt.get(str(val[0]), {})
                src_cls = src.get("class_type", "")
                rets = nodes_spec.get(src_cls, {}).get("returns", [])
                typ = rets[int(val[1])] if int(val[1]) < len(rets) else "*"
                ui_inputs.append({"name": name, "type": typ, "link": lid})
                links[lid - 1][5] = typ
                links[lid - 1][4] = len(ui_inputs) - 1
                continue
            if not is_widget(name, ispec):
                # socket tuỳ chọn chưa nối dây → vẫn phải có mặt trong inputs (link: null)
                ui_inputs.append({"name": name,
                                  "type": ispec.get("type", "*"),
                                  "link": None})
                continue
            # widget: dùng giá trị trong prompt, nếu không có thì dùng default của node
            if val is _MISSING:
                val = ispec.get("default", "" if ispec.get("type") == "STRING" else None)
                if val is None:
                    val = ispec["choices"][0] if ispec.get("choices") else WIDGET_DEFAULT.get(
                        ispec.get("type", ""), "")
            widgets.append(val)
            if name in ("seed", "noise_seed") and cls in SEED_CONTROL_CLASSES:
                widgets.append(inputs.get("control_after_generate", "randomize"))
        # input không có trong spec (node lạ) mà cũng không phải link → nối vào cuối widget
        for name, val in inputs.items():
            if name not in order_keys and not _is_link(val):
                widgets.append(val)

        # outputs: gom theo slot
        out_slots: Dict[int, List[int]] = {}
        for lid, src, slot, tgt, tslot, typ in links:
            if src == nid:
                out_slots.setdefault(int(slot), []).append(lid)
        rets = nspec.get("returns", [])
        ui_outputs: List[Dict[str, Any]] = []
        n_out = max([len(rets)] + [s + 1 for s in out_slots] or [1])
        for slot in range(n_out):
            typ = rets[slot] if slot < len(rets) else "*"
            ls = out_slots.get(slot, [])
            ui_outputs.append({"name": f"out{slot}" if not rets else typ.lower(),
                               "type": typ, "links": ls or None,
                               "slot_index": slot if ls else None})

        row = col_rows.get(depth[nid], 0)
        col_rows[depth[nid]] = row + 1
        ui_nodes.append({
            "id": int(nid) if nid.isdigit() else nid,
            "type": cls,
            "title": (title_prefix + cls) if title_prefix else cls,
            "pos": [80 + 320 * depth[nid], 80 + 220 * row],
            "size": [320, 140],
            "flags": {},
            "order": ordered.index(nid),
            "mode": 0,
            "inputs": ui_inputs,
            "outputs": ui_outputs,
            "properties": {"Node name for S&R": cls},
            "widgets_values": widgets,
        })

    return {
        "last_node_id": max([n["id"] for n in ui_nodes if isinstance(n["id"], int)] or [1]),
        "last_link_id": max([l[0] for l in links] or [0]),
        "nodes": ui_nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


# --------------------------------------------------------------------------- UI → API
def ui_to_api(ui: Dict[str, Any], spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Đọc ngược UI format về API format.

    Trong UI format widget KHÔNG nằm trong `inputs` mà nằm ở `widgets_values` theo đúng
    thứ tự INPUT_TYPES của node → cần node spec để đặt lại tên cho từng giá trị.
    """
    if "nodes" not in ui:
        raise ValueError("không phải UI format (thiếu 'nodes')")
    nodes_spec = (spec or {}).get("nodes", {})
    link_map: Dict[int, Tuple[str, int]] = {}
    for entry in ui.get("links") or []:
        lid, src, sslot = entry[0], entry[1], entry[2]
        link_map[lid] = (str(src), int(sslot))

    prompt: Dict[str, Any] = {}
    for node in ui["nodes"]:
        if node.get("mode", 0) in (2, 4):  # muted / bypass
            continue
        nid = str(node["id"])
        cls = node.get("type", "")
        nspec = nodes_spec.get(cls, {})
        linked = {i.get("name"): i.get("link") for i in (node.get("inputs") or [])}

        inputs: Dict[str, Any] = {}
        for name, lid in linked.items():
            if lid is not None and lid in link_map:
                inputs[name] = [link_map[lid][0], link_map[lid][1]]

        widget_names = [k for sec in ("required", "optional") for k, v in nspec.get(sec, {}).items()
                        if k not in linked and is_widget(k, v)]
        widgets = list(node.get("widgets_values") or [])
        for i, name in enumerate(widget_names):
            if i >= len(widgets):
                break
            inputs[name] = widgets[i]
            if name in ("seed", "noise_seed") and cls in SEED_CONTROL_CLASSES:
                pass  # widget control_after_generate không phải input của node
        if cls in SEED_CONTROL_CLASSES and widgets:
            # giao diện chèn combo control_after_generate ngay sau seed/noise_seed
            for i, name in enumerate(widget_names):
                if name in ("seed", "noise_seed") and i + 1 < len(widgets):
                    inputs["control_after_generate"] = widgets[i + 1]
                    # mọi widget sau đó bị lệch 1 ô
                    for j in range(i + 1, len(widget_names)):
                        if j + 1 < len(widgets):
                            inputs[widget_names[j]] = widgets[j + 1]
                    break
        prompt[nid] = {"class_type": cls, "inputs": inputs}
    return prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out")
    ap.add_argument("--to-api", action="store_true", help="nguồn là UI format, xuất API format")
    ap.add_argument("--spec", default="workflows/node_spec.json")
    args = ap.parse_args()

    data = json.load(open(args.src, encoding="utf-8"))
    spec = json.load(open(args.spec, encoding="utf-8")) if os.path.isfile(args.spec) else None

    if args.to_api:
        out = ui_to_api(data, spec)
        default = args.src.replace("/ui/", "/").replace(".json", "_api.json")
    else:
        if "nodes" in data:
            print("nguồn đã là UI format", file=sys.stderr)
            return 2
        out = api_to_ui(data, spec)
        base = os.path.basename(args.src)
        default = os.path.join(os.path.dirname(args.src), "ui", base)

    dest = args.out or default
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"✅ {args.src} → {dest}  ({len(out)} khoá)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
