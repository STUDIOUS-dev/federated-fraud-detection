"""Phase E: update clipping, client-level differential privacy, adaptive (omniscient) attacks,
and communication accounting. Pure NumPy; used by the Tracked* strategies in ``server.py``.

All functions work on *updates* (client parameters minus the current global parameters),
flattened to one vector per client.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Attacks that are crafted on the server side from the honest updates of the round
# (an omniscient / colluding attacker, the standard threat model for these attacks).
ADAPTIVE_ATTACKS = ("alie", "minmax")


# --------------------------------------------------------------------------- flatten helpers
def flatten(arrays: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def unflatten(vec: np.ndarray, like: Sequence[np.ndarray]) -> list[np.ndarray]:
    out, i = [], 0
    for a in like:
        out.append(vec[i:i + a.size].reshape(a.shape).astype(np.float32))
        i += a.size
    return out


# --------------------------------------------------------------------------- clipping / DP
def clip_update(delta: np.ndarray, clip_norm: float) -> np.ndarray:
    """Scale ``delta`` down so its L2 norm is at most ``clip_norm`` (no-op if clip_norm <= 0)."""
    if clip_norm <= 0:
        return delta
    norm = float(np.linalg.norm(delta))
    return delta if norm <= clip_norm else delta * (clip_norm / norm)


def gaussian_noise_std(noise_multiplier: float, clip_norm: float, n_clients: int) -> float:
    """Std of the Gaussian noise added to the *average* of ``n_clients`` clipped updates.

    The sum of clipped updates has L2 sensitivity ``clip_norm`` when one client is added or
    removed; the average divides by the (public) number of participants.
    """
    return float(noise_multiplier) * float(clip_norm) / max(1, int(n_clients))


def zcdp_epsilon(noise_multiplier: float, rounds: int, delta: float = 1e-5) -> float:
    """(epsilon, delta) for ``rounds`` Gaussian mechanisms with noise multiplier z.

    Each round is rho = 1 / (2 z^2)-zCDP; composition adds rho; conversion to (eps, delta)-DP
    uses eps = rho + 2 * sqrt(rho * ln(1/delta)) (Bun & Steinke 2016). No amplification by client
    sampling or dropout is claimed, so this is an upper bound. Returns inf when z == 0.
    """
    if noise_multiplier <= 0 or rounds <= 0:
        return math.inf
    rho = rounds / (2.0 * noise_multiplier ** 2)
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


# --------------------------------------------------------------------------- adaptive attacks
def alie_update(benign: np.ndarray, z: float = 1.0) -> np.ndarray:
    """'A Little Is Enough' (Baruch et al., 2019): mean - z * std, coordinate-wise.

    The paper derives z from the number of clients; with 4 clients that formula gives z = 0
    (a harmless update), so the strength is a fixed parameter here.
    """
    return benign.mean(axis=0) - z * benign.std(axis=0)


def minmax_update(benign: np.ndarray, iters: int = 60) -> np.ndarray:
    """Min-Max attack (Shejwalkar & Houmansadr, NDSS 2021), inverse-std perturbation.

    Returns mean + gamma * p with p = -std / ||std|| and the largest gamma for which the malicious
    update is no farther from any honest update than the honest updates are from each other.
    """
    mu = benign.mean(axis=0)
    sd = benign.std(axis=0)
    p = -sd / (np.linalg.norm(sd) + 1e-12)
    pair = max(float(np.linalg.norm(a - b)) for i, a in enumerate(benign) for b in benign[i + 1:])

    def ok(g):
        return float(np.max(np.linalg.norm(benign - (mu + g * p), axis=1))) <= pair

    lo, hi = 0.0, 10.0 * pair + 1e-12
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return mu + lo * p


def craft_adaptive(attack: str, benign: np.ndarray, alie_z: float = 1.0) -> np.ndarray | None:
    """Malicious update from the honest updates of the round, or None if it cannot be crafted."""
    if len(benign) < 2:  # needs a spread of honest updates (e.g. after heavy dropout)
        return None
    if attack == "alie":
        return alie_update(benign, alie_z)
    if attack == "minmax":
        return minmax_update(benign)
    raise ValueError(f"Unknown adaptive attack '{attack}'")


# --------------------------------------------------------------------------- communication
def payload_bytes(parameters) -> int:
    """Bytes of a serialized Flower ``Parameters`` object (what actually goes over the wire)."""
    return int(sum(len(t) for t in parameters.tensors))
