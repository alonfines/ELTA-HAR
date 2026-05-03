import numpy as np


def time_warp(x: np.ndarray, sigma: float = 0.15, n_knots: int = 4) -> np.ndarray:
    """
    Nonlinearly stretch / compress the time axis.

    Physical meaning: the same gesture performed at varying speeds is still the
    same gesture. Interior knot positions are randomly displaced; endpoints are
    fixed so sequence length stays T.
    """
    T         = x.shape[0]
    new_t     = np.linspace(0, T - 1, T)
    knot_new  = np.linspace(0, T - 1, n_knots + 2)
    knot_orig = np.linspace(0, T - 1, n_knots + 2)
    knot_orig[1:-1] += np.random.normal(0, sigma * T / (n_knots + 1), n_knots)
    knot_orig        = np.clip(np.sort(knot_orig), 0, T - 1)
    knot_orig[0], knot_orig[-1] = 0.0, float(T - 1)
    orig_t = np.interp(new_t, knot_new, knot_orig)
    src_t  = np.arange(T, dtype=float)
    return np.stack(
        [np.interp(orig_t, src_t, x[:, d]) for d in range(x.shape[1])], axis=-1
    ).astype(np.float32)


def amplitude_warp(x: np.ndarray, sigma: float = 0.10, n_knots: int = 4) -> np.ndarray:
    """
    Multiply signal by a smooth random amplitude curve.

    Physical meaning: different people apply different force magnitudes for the
    same gesture. Interior knots drawn from N(1, sigma); endpoints fixed at 1.
    """
    T      = x.shape[0]
    knot_t = np.linspace(0, T - 1, n_knots + 2)
    scale  = np.random.normal(1.0, sigma, n_knots + 2)
    scale[0] = scale[-1] = 1.0
    curve  = np.interp(np.arange(T), knot_t, scale)
    return (x * curve[:, np.newaxis]).astype(np.float32)
