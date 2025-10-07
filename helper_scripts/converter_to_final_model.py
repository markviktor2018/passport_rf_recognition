#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a DTRB checkpoint (best_accuracy.pth) into EasyOCR custom model package.

Usage:
  python3 convert_to_easyocr.py \
      --pth /path/to/saved_models/exp_name/best_accuracy.pth \
      --alphabet /path/to/easyocr_train_work/alphabet.txt \
      --outdir /path/to/final_versions/runs/passport_ru_digits \
      --rec_name passport_ru_digits \
      --imgH 32 --imgW 192 --batch_max_length 20
"""

import argparse
import shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to best_accuracy.pth (from DTRB).")
    ap.add_argument("--alphabet", required=True, help="Path to alphabet.txt used in training.")
    ap.add_argument("--outdir", required=True, help="Target directory where EasyOCR model will be exported.")
    ap.add_argument("--rec_name", required=True, help="Name of recog_network for EasyOCR.")
    ap.add_argument("--imgH", type=int, default=32)
    ap.add_argument("--imgW", type=int, default=160)
    ap.add_argument("--batch_max_length", type=int, default=25)
    ap.add_argument("--backbone", default="ResNet")
    ap.add_argument("--seq_modeling", default="BiLSTM")
    ap.add_argument("--prediction", default="CTC")
    args = ap.parse_args()

    pth_src = Path(args.pth).resolve()
    alphabet = Path(args.alphabet).read_text(encoding="utf-8").strip()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # copy weights
    pth_dst = outdir / f"{args.rec_name}.pth"
    shutil.copy2(pth_src, pth_dst)

    # stub .py
    py_text = f"""# Auto-generated for EasyOCR custom recog_network: {args.rec_name}
# To use:
#   import easyocr
#   reader = easyocr.Reader(['ru','en'], recog_network='{args.rec_name}',
#                           user_network_directory=r'{outdir}')
"""
    (outdir / f"{args.rec_name}.py").write_text(py_text, encoding="utf-8")

    # yaml meta
    def escape(s): return s.replace('"', '\\"')
    yaml_text = (
        f'character_list: "{escape(alphabet)}"\n'
        f"imgH: {args.imgH}\n"
        f"imgW: {args.imgW}\n"
        f"batch_max_length: {args.batch_max_length}\n"
        f"prediction: {args.prediction}\n"
        f"model_backbone: {args.backbone}\n"
        f"seq_modeling: {args.seq_modeling}\n"
    )
    (outdir / f"{args.rec_name}.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"[DONE] Exported EasyOCR model to: {outdir}")
    print(f"Use with:\n"
          f"  import easyocr\n"
          f"  reader = easyocr.Reader(['ru','en'], recog_network='{args.rec_name}', "
          f"user_network_directory=r'{outdir}')")

if __name__ == "__main__":
    main()

