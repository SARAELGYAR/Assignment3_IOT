"""Create A3_IOT_submission.zip with all required deliverables (section 4.2)."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_NAME = ROOT.parent / "A3_IOT_submission.zip"

INCLUDE = [
    "train_cosine.py",
    "requirements.txt",
    "README.md",
    "results.txt",
    "cosine_float.tflite",
    "cosine_int8.tflite",
    "model.h",
    "cosine_predictor/cosine_predictor.ino",
    "cosine_predictor/model.h",
    "plots/loss_curves.png",
    "plots/float_prediction.png",
    "plots/int8_vs_float.png",
]


def main():
    missing = [p for p in INCLUDE if not (ROOT / p).exists()]
    if missing:
        print("Missing files (run train_cosine.py first):")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(1)

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE:
            zf.write(ROOT / rel, arcname=f"A3_IOT/{rel.replace(chr(92), '/')}")

    print(f"Created {ZIP_NAME} ({ZIP_NAME.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
