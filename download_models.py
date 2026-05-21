"""One-time download of pretrained model weights into data/models/.

Run once after installing requirements:

    python download_models.py

It redirects torch's hub/cache directory to data/models/ and instantiates the
LPIPS perceptual metric for both backbones (AlexNet and VGG), which triggers the
download of the torchvision feature weights into data/models/. After this, the
notebooks load the weights from data/models/ with no network access, as long as
they set TORCH_HOME to the same directory before using LPIPS.
"""

import os
from pathlib import Path

# Point torch's hub / cache directory at data/models BEFORE importing torch.
MODELS_DIR = Path(__file__).resolve().parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(MODELS_DIR)

import torch  # noqa: E402
import lpips  # noqa: E402

print(f"TORCH_HOME = {os.environ['TORCH_HOME']}")
print("Downloading LPIPS backbones (pulls the torchvision feature weights)...\n")

for net in ["alex", "vgg"]:
    print(f"  instantiating lpips.LPIPS(net='{net}') ...", flush=True)
    _ = lpips.LPIPS(net=net)

print("\nDone. Files now cached under data/models/:")
total = 0.0
for p in sorted(MODELS_DIR.rglob("*")):
    if p.is_file():
        size_mb = p.stat().st_size / 1e6
        total += size_mb
        print(f"  {p.relative_to(MODELS_DIR)}  ({size_mb:.1f} MB)")
print(f"\nTotal: {total:.1f} MB")
print("\nIn the notebooks, set this before using LPIPS:")
print('  os.environ["TORCH_HOME"] = os.path.abspath("../data/models")')
