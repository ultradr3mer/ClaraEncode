import sys
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claraenc.sandbox_paths import sandbox_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    if DEVICE.type != "cuda":
        print("WARNING: CUDA not available, running on CPU")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(DEVICE)}")

    base = sandbox_path("bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        # bytearray -> writable buffer, so torch.frombuffer doesn't warn
        x = torch.frombuffer(bytearray(buffer), dtype=torch.bfloat16).to(DEVICE)

        gen = torch.Generator(device=DEVICE).manual_seed(42)
        rand_vec = torch.rand(x.shape, generator=gen, device=DEVICE, dtype=torch.float32)

        print(x[0])
