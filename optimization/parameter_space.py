"""
optimization/parameter_space.py

Parameter space definition, encode/decode, and hashing for GridParams.
Used by both the Optuna optimizer (sample_optuna_params) and any random
exploration helpers (random_params, params_to_vector, vector_to_params).

Constraints applied after decoding:
    max_open_levels = min(max_open_levels, n_levels)
"""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np

from strategy.grid_engine import GridParams

# ── Parameter space definition ─────────────────────────────────────────────────
# Format: name → (type, low, high)
# type ∈ {"int", "float", "bool"}
# For "bool": low=False, high=True (symbolic; ignored by most helpers)

PARAM_SPACE: dict[str, tuple] = {
    "atr_period":        ("int",   10,    50),
    "atr_multiplier":    ("float", 0.3,   2.0),
    "geometric_ratio":   ("float", 1.0,   2.0),
    "n_levels":          ("int",   2,     8),
    "pullback_pct":      ("float", 0.5,   5.0),
    "hurst_window":      ("int",   50,    200),
    "hurst_threshold":   ("float", 0.45,  0.65),
    "adx_period":        ("int",   10,    30),
    "adx_threshold":     ("float", 20.0,  40.0),
    "vpvr_window":       ("int",   50,    300),
    "use_vpvr_anchor":   ("bool",  False, True),
    "position_size_pct": ("float", 1.0,   8.0),
    "max_open_levels":   ("int",   2,     8),
}

# Ordered list of param names — vector index is stable
_PARAM_NAMES: list[str] = list(PARAM_SPACE.keys())


# ── Constraint fix ─────────────────────────────────────────────────────────────

def _apply_constraints(p: GridParams) -> GridParams:
    """Enforce hard constraints that cannot be expressed as independent bounds."""
    p.max_open_levels = min(p.max_open_levels, p.n_levels)
    return p


# ── Random sampling ────────────────────────────────────────────────────────────

def random_params() -> GridParams:
    """Draw a uniformly random GridParams from PARAM_SPACE."""
    vals: dict = {}
    for name, (ptype, low, high) in PARAM_SPACE.items():
        if ptype == "int":
            vals[name] = random.randint(int(low), int(high))
        elif ptype == "float":
            vals[name] = random.uniform(float(low), float(high))
        else:  # bool
            vals[name] = random.choice([False, True])
    return _apply_constraints(GridParams(**vals))


# ── Optuna trial sampling ──────────────────────────────────────────────────────

def sample_optuna_params(trial) -> GridParams:
    """
    Sample GridParams from an Optuna trial.
    Called inside the objective function for each trial.
    """
    import optuna  # local import — not all callers need optuna
    vals: dict = {}
    for name, (ptype, low, high) in PARAM_SPACE.items():
        if ptype == "int":
            vals[name] = trial.suggest_int(name, int(low), int(high))
        elif ptype == "float":
            vals[name] = trial.suggest_float(name, float(low), float(high))
        else:  # bool
            vals[name] = trial.suggest_categorical(name, [False, True])
    return _apply_constraints(GridParams(**vals))


# ── Vector encode / decode ─────────────────────────────────────────────────────

def params_to_vector(p: GridParams) -> np.ndarray:
    """
    Encode GridParams as a float64 vector in [0, 1]^d.
    Booleans → 0.0 (False) or 1.0 (True).
    """
    v = np.empty(len(_PARAM_NAMES), dtype=np.float64)
    for i, name in enumerate(_PARAM_NAMES):
        ptype, low, high = PARAM_SPACE[name]
        val = getattr(p, name)
        if ptype == "bool":
            v[i] = 1.0 if val else 0.0
        else:
            span = float(high) - float(low)
            v[i] = (float(val) - float(low)) / span if span > 0 else 0.0
    return v


def vector_to_params(v: np.ndarray) -> GridParams:
    """
    Decode a [0, 1]^d vector back to GridParams.
    Applies constraints after decoding.
    """
    vals: dict = {}
    for i, name in enumerate(_PARAM_NAMES):
        ptype, low, high = PARAM_SPACE[name]
        raw = float(np.clip(v[i], 0.0, 1.0))
        if ptype == "int":
            decoded = int(round(float(low) + raw * (float(high) - float(low))))
            decoded = int(np.clip(decoded, int(low), int(high)))
            vals[name] = decoded
        elif ptype == "float":
            decoded = float(low) + raw * (float(high) - float(low))
            decoded = float(np.clip(decoded, float(low), float(high)))
            vals[name] = decoded
        else:  # bool
            vals[name] = raw > 0.5
    return _apply_constraints(GridParams(**vals))


# ── Parameter hashing ──────────────────────────────────────────────────────────

def params_hash(p: GridParams) -> str:
    """
    SHA-256 of a canonical JSON representation of rounded GridParams.
    Floats rounded to 6 decimal places to absorb floating-point noise.
    Used for deduplication in the strategy database.
    """
    canonical: dict = {}
    for name, (ptype, _, _) in PARAM_SPACE.items():
        val = getattr(p, name)
        if ptype == "float":
            canonical[name] = round(float(val), 6)
        elif ptype == "int":
            canonical[name] = int(val)
        else:
            canonical[name] = bool(val)
    payload = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
