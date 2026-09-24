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
        hyperparam_random_input = 8
        hyperparam_floor = 0.0
        hyperparam_pow = 1.0

        remainder = 1.0 - hyperparam_floor
        mult = 1 / remainder
        r_shape = (x.shape[0], hyperparam_random_input)
        rand_vec = torch.rand(r_shape, generator=gen, device=DEVICE, dtype=torch.float32)
        rand_vec = torch.pow((rand_vec - hyperparam_floor) * mult, hyperparam_pow)

        hyperparam_group = 32
        # square net: hidden layers are as wide as the input, so the size is
        # controlled by hyperparam_random_input alone
        hyperparam_hidden = hyperparam_group * hyperparam_random_input
        hyperparam_batch = 16
        hyperparam_epochs = 5000
        hyperparam_lr = 1e-3
        hyperparam_lr_factor = 0.5
        hyperparam_lr_patience = 100
        hyperparam_lr_min = 1e-7

        # one sample = 32 consecutive x values, predicted together from their
        # 32 * hyperparam_random_input random values
        n = x.shape[0]
        groups = n // hyperparam_group
        inputs = rand_vec[:groups * hyperparam_group].reshape(groups, hyperparam_group * hyperparam_random_input)
        x_groups = x[:groups * hyperparam_group].reshape(groups, hyperparam_group)
        # train in float32, bf16 is only the storage format of the target
        target = x_groups.float()

        model = torch.nn.Sequential(
            torch.nn.Linear(hyperparam_group * hyperparam_random_input, hyperparam_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hyperparam_hidden, hyperparam_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hyperparam_hidden, hyperparam_group),
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparam_lr)
        # halve the lr whenever the full-data loss stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=hyperparam_lr_factor, patience=hyperparam_lr_patience, min_lr=hyperparam_lr_min)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(hyperparam_epochs):
            perm = torch.randperm(groups, generator=gen, device=DEVICE)
            for start in range(0, groups, hyperparam_batch):
                idx = perm[start:start + hyperparam_batch]
                pred = model(inputs[idx])
                loss = loss_fn(pred, target[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_all = model(inputs)
                full_loss = loss_fn(pred_all, target).item()
            scheduler.step(full_loss)

            if epoch % 250 == 0 or epoch == hyperparam_epochs - 1:
                # exact hit = prediction rounds to the same bf16 value
                hits = (pred_all.to(torch.bfloat16) == x_groups).sum().item()
                lr = optimizer.param_groups[0]["lr"]
                print(f"{name} epoch {epoch:5d}  lr {lr:.1e}  mse {full_loss:.3e}  exact {hits}/{groups * hyperparam_group}")
