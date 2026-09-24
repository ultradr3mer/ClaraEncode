import numpy as np

from claraenc.sandbox_paths import sandbox_path

if __name__ == "__main__":
    base = sandbox_path("bins")

    for path in base.glob("model.layers.0.input_layernorm.weight.bin"):
        with open(path, "rb") as f:
            buffer = f.read()
        name = path.name

        x = np.frombuffer(buffer, dtype=np.uint16)

        rng = np.random.default_rng(seed=42)
        rand_b = rng.uniform(low=0, high=1, size=x.shape)
        rand_vec = np.array(rand_b, dtype=float)

        print(x[0])