"""Small shared validators for the public API."""

from numbers import Real

import numpy as np


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def finite_real(value, name, *, minimum=0, strict=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (value <= minimum if strict else value < minimum)):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return float(value)


def storage_key(value, name="key"):
    if not isinstance(value, str) or not value or "/" in value:
        raise ValueError(f"{name} must be a non-empty string without '/'")
    return value


def seed_value(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= value < 2**32):
        raise ValueError(f"{name} must be an integer in [0, 2**32)")
    return int(value)
