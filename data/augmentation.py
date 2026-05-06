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


def landmark_rotation(seq: np.ndarray, angle_range: float = 5.0) -> np.ndarray:
    """Rotate 2D landmark coordinates around their centroid by a random angle.

    Args:
        seq: shape (T, D) where D is even (landmark pairs: x, y)
        angle_range: degrees to rotate (will sample from [-angle_range, +angle_range])

    Returns:
        shape (T, D) — rotated landmarks
    """
    angle_deg = np.random.uniform(-angle_range, angle_range)
    angle_rad = np.radians(angle_deg)

    num_landmarks = seq.shape[1] // 2
    coords = seq.reshape(seq.shape[0], num_landmarks, 2)

    centroid = coords.mean(axis=(0, 1), keepdims=True)
    centered = coords - centroid

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    rotated = centered @ rot_matrix.T
    rotated = rotated + centroid

    return rotated.reshape(seq.shape).astype(np.float32)


def landmark_scaling(seq: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Scale 2D landmark coordinates around their centroid by a random factor.

    Args:
        seq: shape (T, D) where D is even (landmark pairs: x, y)
        scale_range: tuple (min_scale, max_scale), will sample uniformly from this range

    Returns:
        shape (T, D) — scaled landmarks
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])

    num_landmarks = seq.shape[1] // 2
    coords = seq.reshape(seq.shape[0], num_landmarks, 2)

    centroid = coords.mean(axis=(0, 1), keepdims=True)
    centered = coords - centroid

    scaled = centered * scale
    scaled = scaled + centroid

    return scaled.reshape(seq.shape).astype(np.float32)
