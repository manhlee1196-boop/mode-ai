#!/usr/bin/env python3
"""Kiểm tra mọi artifact sinh ra có còn KHỚP với nguồn không — không cần GPU, không cần mạng.

Vì workflows/ và notebook đều ĐƯỢC SINH từ scripts/, chúng có thể lệch khỏi nguồn khi ai đó
sửa tay file JSON hoặc sửa `build_workflows.py` mà quên chạy lại. Script này sinh lại toàn bộ
vào thư mục tạm rồi so sánh từng byte với bản đã commit.

    python3 scripts/check_sync.py          # báo lỗi (exit 1) nếu lệch
    python3 scripts/check_sync.py --fix    # ghi đè lại cho khớp

Các bước:
  1. workflows/*.json        ← scripts/build_workflows.py
  2. workflows/ui/*.json     ← scripts/api_to_ui.py  (từ API format)
  3. ComfyUI_Colab_WAI_fixed.ipynb ← scripts/make_notebook.py
     (đồng thời kiểm tra builder nhúng trong notebook giống hệt build_workflows.py)
  4. python3 scripts/validate_workflows.py
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

API_DIR = "workflows"
UI_DIR = "workflows/ui"
NOTEBOOK = "ComfyUI_Colab_WAI_fixed.ipynb"
SPEC = "workflows/node_spec.json"

# file sinh ra từ API, không phải nguồn độc lập
GENERATED_PREFIXES = ("flux_q5_",)


def run(cmd: list[str], cwd: str = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _run_or_die(cmd: list[str], what: str) -> None:
    r = run(cmd)
    if r.returncode:
        print(r.stdout.rstrip())
        print(r.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(f"\n❌ {what} thất bại — sửa lỗi trên rồi chạy lại.")
    print(r.stdout.rstrip())


def gen_workflows(tmp: str) -> dict[str, str]:
    """Sinh workflows/ (API) + workflows/ui/ vào tmp."""
    api_tmp = os.path.join(tmp, "workflows")
    ui_tmp = os.path.join(api_tmp, "ui")
    os.makedirs(ui_tmp, exist_ok=True)

    _run_or_die([sys.executable, os.path.join(HERE, "build_workflows.py"), "--out", api_tmp],
                "scripts/build_workflows.py")
    _run_or_die([sys.executable, os.path.join(HERE, "prompt_presets.py"),
                 "--out", os.path.join(api_tmp, "prompts.json")],
                "scripts/prompt_presets.py")
    # node_spec.json là ĐẦU VÀO của api_to_ui (không phải artifact sinh ra)
    shutil.copy2(os.path.join(ROOT, SPEC), os.path.join(api_tmp, "node_spec.json"))

    # chỉ file workflow mới qua api_to_ui; prompts.json / node_spec.json thì không
    for name in sorted(f for f in os.listdir(api_tmp)
                       if f.startswith("flux_q5_") and f.endswith(".json")):
        _run_or_die([sys.executable, os.path.join(HERE, "api_to_ui.py"),
                     os.path.join(api_tmp, name),
                     "-o", os.path.join(ui_tmp, name),
                     "--spec", os.path.join(api_tmp, "node_spec.json")],
                    f"scripts/api_to_ui.py ({name})")

    out: dict[str, str] = {}
    for name in sorted(f for f in os.listdir(api_tmp)
                       if f.endswith(".json") and f != "node_spec.json"):
        out[f"{API_DIR}/{name}"] = os.path.join(api_tmp, name)
    for name in sorted(os.listdir(ui_tmp)):
        out[f"{UI_DIR}/{name}"] = os.path.join(ui_tmp, name)
    return out


def gen_notebook(tmp: str) -> str:
    nb_tmp = os.path.join(tmp, NOTEBOOK)
    _run_or_die([sys.executable, os.path.join(HERE, "make_notebook.py"), "-o", nb_tmp],
                "scripts/make_notebook.py")
    return nb_tmp


def compare(produced: dict[str, str], label: str, fix: bool) -> tuple[list[str], int]:
    """So sánh artifact đã sinh với bản trong repo. Trả về (danh sách lệch, số khớp)."""
    import difflib

    missing = [r for r in produced if not os.path.isfile(os.path.join(ROOT, r))]
    drifted = [r for r in produced
               if r not in missing
               and not filecmp.cmp(os.path.join(ROOT, r), produced[r], shallow=False)]
    ok = len(produced) - len(missing) - len(drifted)

    print(f"\nĐối chiếu {label}:")
    for rel in missing:
        print(f"  ➕ THIẾU  {rel}")
    for rel in drifted:
        print(f"  ✗ LỆCH   {rel}")
        a = open(os.path.join(ROOT, rel), encoding="utf-8").read().splitlines()
        b = open(produced[rel], encoding="utf-8").read().splitlines()
        diff = list(difflib.unified_diff(a, b, "đã commit", "sinh lại", lineterm="", n=1))
        for line in diff[:20]:
            print("      " + line)
        if len(diff) > 20:
            print(f"      … và {len(diff) - 20} dòng nữa")
    if ok:
        print(f"  ✅ {ok} file khớp từng byte")

    if (missing or drifted) and fix:
        for rel in missing + drifted:
            os.makedirs(os.path.dirname(os.path.join(ROOT, rel)) or ".", exist_ok=True)
            shutil.copy2(produced[rel], os.path.join(ROOT, rel))
            print(f"  🔧 đã ghi lại {rel}")
    return missing + drifted, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true",
                    help="ghi đè artifact trong repo cho khớp nguồn, thay vì chỉ báo lỗi")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="mode-ai-sync-")
    try:
        # --- 1) workflows (API) + bản UI -------------------------------------
        print("① Sinh lại workflows/ từ scripts/build_workflows.py …")
        wf = gen_workflows(tmp)
        bad, _ = compare(wf, "workflows/", fix=args.fix)
        if bad and not args.fix:
            print("\n❌ workflows/ lệch khỏi nguồn. Sửa bằng:")
            print("     python3 scripts/build_workflows.py")
            print("     for f in workflows/flux_q5_*.json; do "
                  "python3 scripts/api_to_ui.py \"$f\"; done")
            print("   hoặc: python3 scripts/check_sync.py --fix")
            return 1

        # --- 2) notebook (kiểm tra luôn builder nhúng vs build_workflows.py) --
        print("\n② Sinh lại notebook từ scripts/make_notebook.py …")
        nb = gen_notebook(tmp)
        bad_nb, _ = compare({NOTEBOOK: nb}, "notebook", fix=args.fix)
        if bad_nb and not args.fix:
            print("\n❌ Notebook lệch khỏi nguồn. Sửa bằng:")
            print("     python3 scripts/make_notebook.py")
            print("   hoặc: python3 scripts/check_sync.py --fix")
            return 1

        total = len(wf) + 1
        if args.fix and (bad or bad_nb):
            print(f"\n✅ Đã ghi lại {len(bad) + len(bad_nb)}/{total} artifact")
        elif not bad and not bad_nb:
            print(f"\n✅ Tất cả {total} artifact khớp từng byte với nguồn")

        # --- 3) kiểm tra tĩnh -------------------------------------------------
        print("\n③ Kiểm tra tĩnh (đối chiếu INPUT_TYPES thật):")
        r = run([sys.executable, os.path.join(HERE, "validate_workflows.py")])
        print(r.stdout.rstrip())
        if r.returncode:
            print(r.stderr, file=sys.stderr)
            return r.returncode
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
