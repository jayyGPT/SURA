#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "tools" / "tmp_payload"
EXPECTED_SHA256 = "637862ceec8ddf26408f45c73c27e85601e1268374f350017d47fe0b6a53f256"

payload_b64 = "".join(path.read_text(encoding="utf-8") for path in sorted(PARTS.glob("part*.b64")))
payload = base64.b64decode(payload_b64, validate=True)
actual = hashlib.sha256(payload).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(f"payload checksum mismatch: {actual}")

with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    archive.extractall(ROOT, filter="data")

checkpoints = ROOT / "checkpoints"
checkpoints.mkdir(exist_ok=True)
shutil.copy2(
    ROOT / "archive/legacy_experiments/Models/dl_models/best_wifi_heatmap.pth",
    checkpoints / "wifi_heatmap.pth",
)
shutil.copy2(
    ROOT / "archive/legacy_experiments/Models/dl_models/best_mag_sequence.pth",
    checkpoints / "magnetic_sequence.pth",
)

print("Applied frozen-protocol corrections and promoted canonical checkpoints.")
