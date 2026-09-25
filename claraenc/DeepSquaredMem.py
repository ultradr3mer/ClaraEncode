"""DeepSquared with recurrent memory: each step gets 32*R random values + M memory values,
outputs 32 x predictions + M new memory values (tanh) for the next step.
Each file is one sequence of its 128 groups; k files are stepped in parallel as the batch.

usage: python claraenc\\DeepSquaredMem.py k=5 R=8 M=32 T=16 epochs=5000 dev=cpu L1=0
    T=0 -> full 128-step BPTT; dev=cpu is ~2.5x faster than cuda for this small serial net
"""
import sys
import time
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claraenc.DeepSquaredCapacity import bit_report, layernorm_files, weight_report

G = 32


def unroll(model, inputs, mem, t_from, t_to):
    """Run steps t_from..t_to-1; returns predictions (k, steps, 32) and the last memory."""
    preds = []
    for t in range(t_from, t_to):
        out = model(torch.cat([inputs[:, t], mem], dim=1))   # 32*R random + M memory in
        preds.append(out[:, :G])                             # first 32 = x prediction
        mem = torch.tanh(out[:, G:])                         # last M = memory for next step
    return torch.stack(preds, dim=1), mem


def run(k, r, m, t, epochs, l1, device, eval_every):
    files = layernorm_files()
    xs = [torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.bfloat16) for p in files[:k]]
    x = torch.stack(xs).to(device)                      # (k, 4096)
    steps = x.shape[1] // G                             # 128 groups per file
    gen = torch.Generator(device=device).manual_seed(42)
    rand_vec = torch.rand((k * x.shape[1], r), generator=gen, device=device, dtype=torch.float32)
    inputs = rand_vec.reshape(k, steps, G * r)
    x_groups = x.reshape(k, steps, G)
    target = x_groups.float()

    width = G * r + m  # square net: hidden as wide as the input
    model = torch.nn.Sequential(
        torch.nn.Linear(width, width), torch.nn.ReLU(),
        torch.nn.Linear(width, width), torch.nn.ReLU(),
        torch.nn.Linear(width, G + m),                  # 32 outputs + m memory
    ).to(device)
    # learned initial memory: the one memory value that is a direct parameter
    init_mem = torch.nn.Parameter(torch.zeros(1, m, device=device))
    params = list(model.parameters()) + [init_mem]
    n_params = sum(p.numel() for p in params)
    weights = [l.weight for l in model if isinstance(l, torch.nn.Linear)]
    chunk = t or steps
    print(f"L1={l1} k={k} R={r} M={m} T={chunk} dev={device} net {width}->{width}->{width}->{G + m}  "
          f"params {n_params}  values {x.numel()}  params/value {n_params / x.numel():.2f}", flush=True)

    weight_report(model, "init")
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100, min_lr=1e-7)
    loss_fn = torch.nn.MSELoss()

    best = 0
    t0 = time.time()
    for epoch in range(epochs):
        mem = init_mem.expand(k, m)                     # every file starts from init_mem
        for c in range(0, steps, chunk):
            pred, mem = unroll(model, inputs, mem, c, min(c + chunk, steps))
            loss = loss_fn(pred, target[:, c:c + chunk])
            if l1:
                loss = loss + l1 * sum(w.abs().sum() for w in weights)
            opt.zero_grad()
            loss.backward()                             # gradient flows back through mem
            if epoch == 0 and c == 0:
                g = init_mem.grad
                print(f"  gradient check: init_mem.grad norm = {g.norm().item() if g is not None and g.numel() else 'n/a'}",
                      flush=True)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            mem = mem.detach()  # truncated BPTT: carry memory, cut the gradient

        # lr schedule on the full-sequence loss: the chunked training loss is too noisy
        with torch.no_grad():
            pred, _ = unroll(model, inputs, init_mem.expand(k, m), 0, steps)
            full = loss_fn(pred, target).item()
        sched.step(full)

        if epoch % eval_every == 0 or epoch == epochs - 1:
            hits = (pred.to(torch.bfloat16) == x_groups)
            best = max(best, hits.sum().item())
            el = time.time() - t0
            per = el / (epoch + 1)
            print(f"  epoch {epoch:5d}  lr {opt.param_groups[0]['lr']:.1e}  mse {full:.2e}  "
                  f"exact {hits.sum().item()}/{x.numel()}  elapsed {el:6.1f}s  "
                  f"{per * 1000:.1f}ms/epoch  eta {per * (epochs - 1 - epoch):6.1f}s", flush=True)

    weight_report(model, "final")
    bit_report(pred.to(torch.bfloat16).reshape(-1), x.reshape(-1))
    per_file = hits.reshape(k, -1).sum(1).tolist()
    print(f"RESULT k={k} R={r} M={m} T={chunk} params={n_params} "
          f"final_exact={hits.sum().item()}/{x.numel()} ({hits.float().mean().item():.2%}) best_exact={best} "
          f"mse={full:.2e} per_file={per_file} time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = dict(a.split("=") for a in sys.argv[1:])
    run(k=int(args.get("k", 5)),
        r=int(args.get("R", 8)),
        m=int(args.get("M", 32)),
        t=int(args.get("T", 16)),
        epochs=int(args.get("epochs", 5000)),
        l1=float(args.get("L1", 0.0)),  # lasso strength on the Linear weight matrices (not biases)
        device=torch.device(args.get("dev", "cpu")),
        eval_every=int(args.get("eval", 250)))
