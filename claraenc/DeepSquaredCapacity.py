"""How many input_layernorm files can one DeepSquared model memorize?
Same setup as claraenc/DeepSquared.py, trained on the first k files concatenated.

usage: python claraenc\\DeepSquaredCapacity.py 1 2 3 5 [norm] [std|mean|quart|pct] [L1=1e-10] [epochs=5000] [dev=cpu|cuda]
    k values run one after another; "norm" adds LayerNorm after each hidden Linear;
    std|mean|quart|pct pick a per-file target scaling, see target_scaling();
    L1 = lasso strength on the change w - w_init (start weights are seeded)
"""
import re
import sys
import time
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claraenc.sandbox_paths import sandbox_path

# cpu is faster than cuda for this small net (23.9 vs 29.0 ms/epoch); override with dev=cuda
DEVICE = torch.device("cpu")

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


def weight_report(model, tag, init=None):
    """Magnitude of the Linear weight matrices, and how many are (near) zero.
    With `init` (the seeded start weights), also the change w - w_init: that change is
    what would need storing, since w_init can be regenerated from the seed."""
    for i, m in enumerate(l for l in model if isinstance(l, torch.nn.Linear)):
        w = m.weight.detach().abs()
        print(f"  weights {tag} L{i} {tuple(m.weight.shape)}: max {w.max().item():.3e}  "
              f"mean {w.mean().item():.3e}  <1e-3: {(w < 1e-3).float().mean().item():.1%}  "
              f"<1e-5: {(w < 1e-5).float().mean().item():.1%}", flush=True)
        if init is not None:
            w0 = init[i]
            d = (m.weight.detach() - w0).abs()
            # unchanged at bf16 precision -> nothing to store for that weight
            same_bf16 = (m.weight.detach().to(torch.bfloat16) == w0.to(torch.bfloat16)).float().mean().item()
            print(f"  change  {tag} L{i}: max {d.max().item():.3e}  mean {d.mean().item():.3e}  "
                  f"<1e-3: {(d < 1e-3).float().mean().item():.1%}  <1e-5: {(d < 1e-5).float().mean().item():.1%}  "
                  f"same as init in bf16: {same_bf16:.1%}", flush=True)


def target_scaling(xf, scale):
    """Per-file reversible scaling of the target; xf is (files, values) float.
    Returns (fwd, inv) working on (groups, 32) tensors, files in order.
    Piecewise linear through per-file anchors x_i -> fixed t_i (straight lines between
    anchors, outer segments extended beyond them); stores len(anchors) numbers per file:
      None:    0 -> 0, 1 -> 1                                     raw x
      "std":   mean -> 0, mean + std -> 1                         mean 0, std 1
      "mean":  0 -> 0, mean -> 0.5                                x / mean * 0.5
      "quart": p25 -> 0.25, median -> 0.5, p75 -> 0.75
      "pct":   p1 -> 0.01, p25 -> 0.25, median -> 0.5, p75 -> 0.75, p99 -> 0.99
    """
    k, n = xf.shape
    dev = xf.device
    ones = torch.ones(k, device=dev)
    if scale == "std":
        mean, std = xf.mean(1), xf.std(1)
        xk, tk = torch.stack([mean, mean + std], 1), [0.0, 1.0]
    elif scale == "mean":
        xk, tk = torch.stack([0 * ones, xf.mean(1)], 1), [0.0, 0.5]
    elif scale in ("quart", "pct"):
        tk = [0.25, 0.5, 0.75] if scale == "quart" else [0.01, 0.25, 0.5, 0.75, 0.99]
        xk = torch.quantile(xf, torch.tensor(tk, device=dev), dim=1).T
    else:
        xk, tk = torch.stack([0 * ones, ones], 1), [0.0, 1.0]
    tk = torch.tensor(tk, device=dev)

    # (k, m) -> (groups, m): each file's anchors repeated for its groups
    xk = xk.repeat_interleave(n // G, dim=0).contiguous()
    tk = tk.expand(xk.shape).contiguous()

    def interp(v, src, dst):
        # segment index per value, clamped so outer segments extend beyond the anchors
        i = torch.searchsorted(src, v.contiguous()).clamp(1, src.shape[1] - 1)
        s0, s1 = src.gather(1, i - 1), src.gather(1, i)
        d0, d1 = dst.gather(1, i - 1), dst.gather(1, i)
        return d0 + (v - s0) * (d1 - d0) / (s1 - s0)

    def fwd(x):
        return interp(x, xk, tk)

    def inv(t):
        return interp(t, tk, xk)
    return fwd, inv


def run(files, k, use_norm=False, l1=0.0, scale=None, epochs=5000):
    xs = [torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.bfloat16) for p in files[:k]]
    x = torch.cat(xs).to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    rand_vec = torch.rand((x.shape[0], R), generator=gen, device=DEVICE, dtype=torch.float32)
    groups = x.shape[0] // G
    inputs = rand_vec.reshape(groups, G * R)
    x_groups = x.reshape(groups, G)
    fwd, inv = target_scaling(torch.stack(xs).float().to(DEVICE), scale)
    target = fwd(x_groups.float())

    def norm():
        # LayerNorm per sample; BatchNorm would differ between train and eval
        return torch.nn.LayerNorm(H) if use_norm else torch.nn.Identity()

    torch.manual_seed(0)  # start weights must be regenerable from a seed
    model = torch.nn.Sequential(
        torch.nn.Linear(G * R, H), norm(), torch.nn.ReLU(),
        torch.nn.Linear(H, H), norm(), torch.nn.ReLU(),
        torch.nn.Linear(H, G),
    ).to(DEVICE)
    weights = [l.weight for l in model if isinstance(l, torch.nn.Linear)]
    init = [w.detach().clone() for w in weights]
    weight_report(model, "init")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100, min_lr=1e-7)
    loss_fn = torch.nn.MSELoss()

    best = 0
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(groups, generator=gen, device=DEVICE)
        for s in range(0, groups, 16):
            idx = perm[s:s + 16]
            loss = loss_fn(model(inputs[idx]), target[idx])
            if l1:
                # lasso on the change from the seeded start, not on w itself
                loss = loss + l1 * sum((w - w0).abs().sum() for w, w0 in zip(weights, init))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out = model(inputs)
            full = loss_fn(out, target).item()  # in scaled units when scale is set
            pred = inv(out)                     # back to x units
        sched.step(full)
        if epoch % 250 == 0 or epoch == epochs - 1:
            hits = (pred.to(torch.bfloat16) == x_groups)
            best = max(best, hits.sum().item())
            el = time.time() - t0
            per = el / (epoch + 1)
            print(f"  k={k} epoch {epoch:5d}  exact {hits.sum().item()}/{x.numel()} "
                  f"({hits.float().mean().item():.3%})  "
                  f"elapsed {el:6.1f}s  {per * 1000:.1f}ms/epoch  eta {per * (epochs - 1 - epoch):6.1f}s", flush=True)
    weight_report(model, "final", init)
    bit_report(pred.to(torch.bfloat16).reshape(-1), x)
    per_file = hits.reshape(k, -1).sum(1).tolist()
    n = x.numel()
    print(f"k={k} dev={DEVICE} norm={use_norm} scale={scale} L1={l1} values={n} final_exact={hits.sum().item()}/{n} ({hits.float().mean().item():.2%}) "
          f"best_exact={best} mse={full:.2e} per_file={per_file} time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    files = layernorm_files()
    argv = sys.argv[1:]
    use_norm = "norm" in argv
    scale = next((a for a in argv if a in ("std", "mean", "quart", "pct")), None)
    l1 = next((float(a[3:]) for a in argv if a.startswith("L1=")), 0.0)
    epochs = next((int(a[7:]) for a in argv if a.startswith("epochs=")), 5000)
    DEVICE = torch.device(next((a[4:] for a in argv if a.startswith("dev=")), "cpu"))
    ks = [int(a) for a in argv if a not in ("norm", "std", "mean", "quart", "pct") and "=" not in a] \
        or [1, 2, 3, 5, len(files)]
    print("files:", [p.name for p in files], flush=True)
    for k in ks:
        run(files, k, use_norm, l1, scale, epochs)
