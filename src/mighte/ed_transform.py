"""ED response link, defined in canonical proportions rather than percentage points."""
from __future__ import annotations

import numpy as np
from scipy.special import logit


def boundary_epsilon(settings: dict) -> float:
    if settings["name"] != "logit":
        raise ValueError("ed_target_transform.name must be logit")
    epsilon = float(settings["boundary_epsilon"])
    if not np.isfinite(epsilon) or not 0 < epsilon < .5:
        raise ValueError("ED boundary_epsilon must be a finite proportion between 0 and 0.5")
    return epsilon


def ed_logit(proportions, epsilon: float) -> np.ndarray:
    """Clip input boundaries only; interior observations retain their exact log-odds."""
    values = np.asarray(proportions, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("ED response observations must be finite proportions in [0,1]")
    return logit(np.clip(values, epsilon, 1 - epsilon))
