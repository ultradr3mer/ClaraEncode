"""How many input_layernorm files can one DeepSquared model memorize?
Same setup as claraenc/DeepSquared.py, trained on the first k files concatenated.

usage: python claraenc\\DeepSquaredCapacity.py 1 2 3 5     (k values, run one after another)
"""
import re
import sys
import time
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claraenc.sandbox_paths import sandbox_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

R, G = 8, 32
H = G * R


def layernorm_files():
    """model.layers.*.input_layernorm.weight.bin, sorted by layer number."""
    return sorted(sandbox_path("bins").glob("model.layers.*.input_layernorm.weight.bin"),
                  key=lambda p: int(re.search(r"layers\.(\d+)\.", p.name).group(1)))


def bit_report(pred_bf, x_bf):
    """Compare bf16 prediction vs truth as raw 16-bit patterns."""
    pb = pred_bf.view(torch.int16).int() & 0xFFFF
    xb = x_bf.view(torch.int16).int() & 0xFFFF
    diff = pb ^ xb
    wrong = torch.zeros_like(diff)
    for b in range(16):
        wrong += (diff >> b) & 1
    # leading correct bits, MSB first (sign, 8 exponent, 7 mantissa)
    lead = torch.full_like(diff, 16)
    for b in range(16):
        bit_set = ((diff >> (15 - b)) & 1).bool()
        lead = torch.where(bit_set & (lead == 16), torch.full_like(lead, b), lead)
    # steps off: sign-magnitude -> ordered int, so neighbouring bf16 values differ by 1
    def ordered(v):
        return torch.where(v & 0x8000 != 0, -(v & 0x7FFF), v & 0x7FFF)
    steps = (ordered(pb) - ordered(xb)).abs()

    n = diff.numel()
    total = wrong.sum().item()
    print(f"  bits: {total} wrong of {n * 16} ({total / (n * 16):.3%}), "
          f"avg {total / n:.3f} wrong bits/value, values needing a fix {(wrong > 0).sum().item()}/{n}")
    print("  wrong bits per value:", {i: int((wrong == i).sum()) for i in range(17) if (wrong == i).any()})
    print("  leading correct bits:", {i: int((lead == i).sum()) for i in range(17) if (lead == i).any()})
    sq = steps.float()
    print(f"  bf16 steps off: max {steps.max().item()}  "
          f"<=1: {(steps <= 1).sum().item()}  <=4: {(steps <= 4).sum().item()}  "
          f"<=16: {(steps <= 16).sum().item()}  median {sq.median().item():.0f}")
    # spacing between neighbouring bf16 values at each x (x assumed nonzero)
    xf = x_bf.float().abs()
    ulp = torch.pow(2.0, torch.floor(torch.log2(xf)) - 7)
    qs = torch.quantile(ulp, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=ulp.device)).tolist()
    print("  bf16 step size at x (min/25%/50%/75%/max):", " ".join(f"{q:.2e}" for q in qs),
          f" relative step {2 ** -8:.2%}..{2 ** -7:.2%}")


def run(files, k):
    xs = [torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.bfloat16) for p in files[:k]]
    x = torch.cat(xs).to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    rand_vec = torch.rand((x.shape[0], R), generator=gen, device=DEVICE, dtype=torch.float32)
    groups = x.shape[0] // G
    inputs = rand_vec.reshape(groups, G * R)
    x_groups = x.reshape(groups, G)
    target = x_groups.float()

    model = torch.nn.Sequential(
        torch.nn.Linear(G * R, H), torch.nn.ReLU(),
        torch.nn.Linear(H, H), torch.nn.ReLU(),
        torch.nn.Linear(H, G),
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100, min_lr=1e-7)
    loss_fn = torch.nn.MSELoss()

    best = 0
    t0 = time.time()
    for epoch in range(5000):
        perm = torch.randperm(groups, generator=gen, device=DEVICE)
        for s in range(0, groups, 16):
            idx = perm[s:s + 16]
            loss = loss_fn(model(inputs[idx]), target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(inputs)
            full = loss_fn(pred, target).item()
        sched.step(full)
        if epoch % 250 == 0 or epoch == 4999:
            hits = (pred.to(torch.bfloat16) == x_groups)
            best = max(best, hits.sum().item())
            el = time.time() - t0
            per = el / (epoch + 1)
            print(f"  k={k} epoch {epoch:5d}  exact {hits.sum().item()}/{x.numel()}  "
                  f"elapsed {el:6.1f}s  {per * 1000:.1f}ms/epoch  eta {per * (4999 - epoch):6.1f}s", flush=True)
    bit_report(pred.to(torch.bfloat16).reshape(-1), x)
    per_file = hits.reshape(k, -1).sum(1).tolist()
    n = x.numel()
    print(f"k={k} values={n} final_exact={hits.sum().item()}/{n} ({hits.float().mean().item():.2%}) "
          f"best_exact={best} mse={full:.2e} per_file={per_file} time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    files = layernorm_files()
    ks = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 5, len(files)]
    print("files:", [p.name for p in files], flush=True)
    for k in ks:
        run(files, k)
