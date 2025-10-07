#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train EasyOCR recognition model on your dataset_for_easyocr/*/gt.txt folders.

Pipeline:
1) Scan DATASET_ROOT subfolders (name/, fam/, …), each must contain gt.txt with lines "filename<TAB>text".
2) Build unified list of (image_path, label), shuffle, train/val split.
3) Auto-build alphabet from labels (can be edited later in workdir/alphabet.txt).
4) Create LMDB datasets via clovaai/deep-text-recognition-benchmark (DTRB).
5) Run DTRB training (CTC: ResNet + BiLSTM) with your alphabet.
6) Install the best .pth along with .yaml and .py into ~/.EasyOCR/user_network/<rec_name>.*

Usage example:
  python3 train_easyocr.py \
    --dataset_root /path/to/dataset_for_easyocr \
    --workdir /path/to/work_eocr_train \
    --rec_name passport_ru_g2 \
    --epochs 15 --batch_size 256 --val_split 0.1 --imgH 32 --imgW 160 --workers 4

Prereqs:
  - Python 3.9–3.12 (for 3.12 DTRB may require a tiny patch to dataset.py: replace
    'from torch._utils import _accumulate' with 'from itertools import accumulate as _accumulate')
  - PyTorch installed (GPU or CPU)
  - pip packages: easyocr, lmdb, pillow, opencv-python
  - git (to clone DTRB)
"""

import argparse
import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from random import shuffle, seed
from datetime import datetime

# --------------------------
# Helpers
# --------------------------

def run(cmd, cwd=None, env=None):
    print(f"[RUN] {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if r.returncode != 0:
        sys.exit(f"Command failed with code {r.returncode}: {cmd}")

def read_gt_file(gt_path: Path):
    """Yield (abs_image_path, label) for each line in gt.txt (filename<TAB>text)."""
    folder = gt_path.parent
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            fname = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            img_path = folder / fname
            if img_path.exists():
                yield (str(img_path.resolve()), text)

def scan_dataset_root(dataset_root: Path):
    """Collect all (img_path, label) from every subfolder/<gt.txt>."""
    all_pairs = []
    for sub in sorted(dataset_root.iterdir()):
        if not sub.is_dir():
            continue
        gt = sub / "gt.txt"
        if gt.exists():
            all_pairs.extend(list(read_gt_file(gt)))
    return all_pairs

def build_charset(pairs):
    """Derive sorted unique character set from labels; normalize and clean tabs/newlines."""
    charset = set()
    for _, lab in pairs:
        lab = lab.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        lab = unicodedata.normalize("NFKC", lab)
        for ch in lab:
            charset.add(ch)

    # Preferred ordering improves readability; rest appended sorted by codepoint.
    digits = list("0123456789")
    latin = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    cyr_cap = [chr(c) for c in range(ord('А'), ord('Я')+1)] + ['Ё']
    cyr_low = [chr(c) for c in range(ord('а'), ord('я')+1)] + ['ё']
    symbols = [' ', '-', '.', ',', ':', ';', '/', '\\', '(', ')', '<', '>', '_']
    preferred = digits + latin + cyr_cap + cyr_low + symbols

    ordered = []
    for ch in preferred:
        if ch in charset:
            ordered.append(ch); charset.remove(ch)
    ordered += sorted(list(charset))

    # Deduplicate preserving order
    seen = set(); final = []
    for ch in ordered:
        if ch not in seen:
            final.append(ch); seen.add(ch)
    return "".join(final)

def write_list_file(pairs, out_path: Path):
    """Write lines path<TAB>label (absolute image paths are fine for DTRB)."""
    with out_path.open("w", encoding="utf-8") as f:
        for p, lab in pairs:
            lab = unicodedata.normalize("NFKC", lab).replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{p}\t{lab}\n")

def ensure_repo_cloned(repo_url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(f"git clone {repo_url} {dest}")

def escape_yaml_string(s: str) -> str:
    """Escape double quotes for YAML inline string."""
    return s.replace('"', '\\"')

def copy_model_files_to_user_network(pth_src: Path, rec_name: str, alphabet: str, imgH: int, imgW: int):
    """Create ~/.EasyOCR/user_network/{rec_name}.pth/.yaml/.py."""
    user_dir = Path.home() / ".EasyOCR" / "user_network"
    user_dir.mkdir(parents=True, exist_ok=True)

    # Weights
    pth_dst = user_dir / f"{rec_name}.pth"
    shutil.copy2(pth_src, pth_dst)

    # Minimal stub .py (compat placeholder)
    py_text = f'''# Auto-generated for EasyOCR custom recog_network: {rec_name}
# Standard CRNN (ResNet + BiLSTM + CTC). EasyOCR will use internal modules.
'''
    (user_dir / f"{rec_name}.py").write_text(py_text, encoding="utf-8")

    # YAML with alphabet and meta (tune batch_max_length if needed)
    yaml_text = (
        f'character_list: "{escape_yaml_string(alphabet)}"\n'
        f"imgH: {imgH}\n"
        f"imgW: {imgW}\n"
        f"batch_max_length: 25\n"
        f"prediction: CTC\n"
        f"model_backbone: ResNet\n"
        f"seq_modeling: BiLSTM\n"
    )
    (user_dir / f"{rec_name}.yaml").write_text(yaml_text, encoding="utf-8")
    return user_dir

# --------------------------
# Main pipeline
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="Root with subfolders (name/, fam/, …), each containing gt.txt and images.")
    ap.add_argument("--workdir", required=True, help="Workspace where LMDB and training outputs will be created.")
    ap.add_argument("--rec_name", default="passport_ru_g2", help="Name for recog_network in EasyOCR.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--imgH", type=int, default=32)
    ap.add_argument("--imgW", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo_url", default="https://github.com/clovaai/deep-text-recognition-benchmark.git")
    args = ap.parse_args()

    seed(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    workdir = Path(args.workdir).resolve()
    dtrb_dir = workdir / "deep-text-recognition-benchmark"
    lmdb_root = workdir / "lmdb"
    lists_dir = workdir / "lists"

    for p in [workdir, lmdb_root, lists_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # 1) Gather pairs
    print(f"[INFO] Scanning dataset_root: {dataset_root}")
    pairs_all = scan_dataset_root(dataset_root)
    if not pairs_all:
        sys.exit("No labeled pairs found. Ensure subfolders contain gt.txt and images.")
    print(f"[INFO] Collected {len(pairs_all)} labeled images.")

    # 2) Shuffle and split
    shuffle(pairs_all)
    n_total = len(pairs_all)
    n_val = max(1, int(n_total * args.val_split))
    val_pairs = pairs_all[:n_val]
    train_pairs = pairs_all[n_val:]
    print(f"[INFO] Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # 3) Build character set
    alphabet = build_charset(train_pairs + val_pairs)
    print(f"[INFO] Alphabet size: {len(alphabet)}")
    (workdir / "alphabet.txt").write_text(alphabet, encoding="utf-8")

    # 4) Write list files
    train_list = lists_dir / "train_gt.txt"
    val_list = lists_dir / "val_gt.txt"
    write_list_file(train_pairs, train_list)
    write_list_file(val_pairs, val_list)

    # 5) Clone DTRB repo if needed
    ensure_repo_cloned(args.repo_url, dtrb_dir)

    # 6) Create LMDB datasets (DTRB supports absolute paths in GT files)
    py = sys.executable
    run(f'{py} create_lmdb_dataset.py --inputPath "{dataset_root}" --gtFile "{train_list}" --outputPath "{lmdb_root / "training"}"', cwd=dtrb_dir)
    run(f'{py} create_lmdb_dataset.py --inputPath "{dataset_root}" --gtFile "{val_list}" --outputPath "{lmdb_root / "validation"}"', cwd=dtrb_dir)

    # 7) Train model (CTC: ResNet + BiLSTM). Use correct flags for DTRB.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.rec_name}_{ts}"

    train_cmd = (
        f'{py} train.py '
        f'--train_data "{lmdb_root / "training"}" '
        f'--valid_data "{lmdb_root / "validation"}" '
        f'--select_data "/" --batch_ratio "1.0" '
        f'--manualSeed {args.seed} '
        f'--Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC '
        f'--imgH {args.imgH} --imgW {args.imgW} '
        f'--character "{alphabet}" '
        f'--batch_max_length 25 '
        f'--workers {args.workers} '
        f'--batch_size {args.batch_size} '
        f'--num_iter {args.epochs * 2000} '
        f'--exp_name "{exp_name}" '
    )
    run(train_cmd, cwd=dtrb_dir)

    # 8) Pick best checkpoint from DTRB default location: saved_models/<exp_name>/
    exp_dir = dtrb_dir / "saved_models" / exp_name
    if not exp_dir.exists():
        # fallback: pick latest exp
        root_saved = dtrb_dir / "saved_models"
        if not root_saved.exists():
            sys.exit("Cannot find saved_models/ in DTRB repo. Check training logs.")
        candidates = sorted(root_saved.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        exp_dir = candidates[0] if candidates else None
    if not exp_dir or not exp_dir.exists():
        sys.exit("Cannot find experiment directory with .pth under saved_models/")

    pth_candidates = list(exp_dir.glob("*.pth"))
    best_pth = None
    # Prefer well-known names if present:
    for n in ["best_accuracy.pth", "best_norm_ED.pth"]:
        cand = exp_dir / n
        if cand.exists():
            best_pth = cand
            break
    if best_pth is None and pth_candidates:
        best_pth = sorted(pth_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    if not best_pth or not best_pth.exists():
        sys.exit("No .pth model produced, check training logs.")

    # 9) Install into EasyOCR user_network
    user_net_dir = copy_model_files_to_user_network(best_pth, args.rec_name, alphabet, args.imgH, args.imgW)
    print(f"[DONE] Installed custom EasyOCR model as recog_network='{args.rec_name}' in: {user_net_dir}")
    print("Use it like:\n  import easyocr\n  reader = easyocr.Reader(['ru','en'], recog_network='%s')\n" % args.rec_name)

if __name__ == "__main__":
    main()

