import numpy as np
import numpy.typing as npt

from clarautils import NBitArray


def get_bitwise_entropy(a: NBitArray):
    return get_bitwise_entropy(a.get_array(), a.get_item_count())

def get_bitwise_entropy(v, bit_count) -> npt.NDArray[np.float32]:
    # if isinstance(v, torch.Tensor):
    #     # PyTorch Implementation
    #     if v.numel() == 0:
    #         return torch.zeros(bit_count, device=v.device)
    #
    #     v = v.to(torch.int32).ravel()
    #     p = torch.zeros(bit_count, device=v.device, dtype=torch.float32)
    #
    #     # Berechne p (Wahrscheinlichkeit für eine 1) für jedes Bit
    #     for i in range(bit_count):
    #         p[i] = ((v >> i) & 1).float().mean()
    #
    #     entropy = torch.zeros(bit_count, device=v.device, dtype=torch.float32)
    #     mask = (p > 0) & (p < 1)
    #     entropy[mask] = -p[mask] * torch.log2(p[mask]) - (1 - p[mask]) * torch.log2(1 - p[mask])
    #     return entropy
    #
    # else:
        # NumPy Implementation
        v = np.asarray(v, dtype=np.int32).ravel()
        if v.size == 0:
            return np.zeros(bit_count, dtype=np.float32)

        p = np.zeros(bit_count, dtype=np.float32)
        for i in range(bit_count):
            p[i] = np.mean((v >> i) & 1)

        entropy = np.zeros(bit_count, dtype=np.float32)
        mask = (p > 0) & (p < 1)
        entropy[mask] = -p[mask] * np.log2(p[mask]) - (1 - p[mask]) * np.log2(1 - p[mask])
        return entropy

# def individual_gini_optimized(v, bit_count):
#     """
#     Berechnet die Gini-Unreinheit für jedes Bit eines Integer-Vektors.
#
#     Rückgabe:
#         Tensor/Array der Länge bit_count.
#         0 bedeutet: Bit ist innerhalb der Daten konstant.
#         0.5 bedeutet: Bit ist maximal gemischt (50 % 0, 50 % 1).
#     """
#     if isinstance(v, torch.Tensor):
#         # PyTorch-Implementierung
#         if v.numel() == 0:
#             return torch.zeros(bit_count, device=v.device, dtype=torch.float32)
#
#         v = v.to(torch.int32).ravel()
#         p = torch.zeros(bit_count, device=v.device, dtype=torch.float32)
#
#         # Anteil der Einsen für jedes Bit
#         for i in range(bit_count):
#             p[i] = ((v >> i) & 1).float().mean()
#
#         # Gini = 1 - (p² + (1-p)²) = 2p(1-p)
#         return 2.0 * p * (1.0 - p)
#
#     else:
#         # NumPy-Implementierung
#         v = np.asarray(v, dtype=np.int32).ravel()
#         if v.size == 0:
#             return np.zeros(bit_count, dtype=np.float32)
#
#         p = np.zeros(bit_count, dtype=np.float32)
#
#         # Anteil der Einsen für jedes Bit
#         for i in range(bit_count):
#             p[i] = np.mean((v >> i) & 1)
#
#         # Gini = 1 - (p² + (1-p)²) = 2p(1-p)
#         return 2.0 * p * (1.0 - p)

def individual_gini_sum(v, bit_count):
    # if isinstance(v, torch.Tensor):
    #     return torch.sum(individual_gini_optimized(v, bit_count))
    # else:
        return np.sum(individual_gini_optimized(v, bit_count))

def individual_entropy_sum(v, bit_count):
    # if isinstance(v, torch.Tensor):
    #     return torch.sum(get_bitwise_entropy(v, bit_count))
    # else:
        return np.sum(get_bitwise_entropy(v, bit_count))
