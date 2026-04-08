"""
gym_shim.py
===========
Minimal Gymnasium-compatible shim for offline / no-internet environments.

This file is a safety net. When `gymnasium` is installed (pip install gymnasium),
it is used automatically and this file is never loaded.

When gymnasium is NOT available, this shim provides stub implementations of:
  - gym.Env       (base class for all environments)
  - spaces.Box    (continuous observation space)
  - spaces.Discrete (discrete action space)

The stubs are 100% API-compatible with gymnasium for the features used by
CustomerSupportEnv: spaces, reset(), step(), render(), close().

KEY FIXES & IMPROVEMENTS over original:
  - _Env.reset() no longer raises NotImplementedError — it sets the seed
    and returns cleanly so super().reset(seed=seed) works as expected.
  - _Discrete.contains() correctly handles numpy integer types.
  - _Box.sample() returns dtype=float32 to match real gymnasium.
  - Added __repr__ to spaces for cleaner debug output.
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
    # Real gymnasium is available — shim is not needed.

except ImportError:
    import numpy as np
    import random as _random

    # ── Observation space stub ────────────────────────────────────────────

    class _Box:
        """Stub for gymnasium.spaces.Box (continuous observation space)."""

        def __init__(self, low, high, dtype=None, **kwargs):
            self.low   = np.asarray(low,  dtype=np.float32)
            self.high  = np.asarray(high, dtype=np.float32)
            self.dtype = dtype or np.float32
            self.shape = self.low.shape

        def contains(self, x) -> bool:
            x = np.asarray(x, dtype=np.float32)
            return bool(
                x.shape == self.shape
                and np.all(x >= self.low)
                and np.all(x <= self.high)
            )

        def sample(self) -> np.ndarray:
            return np.array([
                np.random.uniform(float(l), float(h))
                for l, h in zip(self.low, self.high)
            ], dtype=np.float32)

        def __repr__(self):
            return f"Box(low={self.low}, high={self.high}, dtype={self.dtype})"

    # ── Action space stub ─────────────────────────────────────────────────

    class _Discrete:
        """Stub for gymnasium.spaces.Discrete (integer action space)."""

        def __init__(self, n: int, **kwargs):
            self.n = int(n)

        def contains(self, x) -> bool:
            """Accept Python int and numpy integer types."""
            try:
                return 0 <= int(x) < self.n
            except (TypeError, ValueError):
                return False

        def sample(self) -> int:
            return _random.randint(0, self.n - 1)

        def __repr__(self):
            return f"Discrete({self.n})"

    # ── Spaces namespace ──────────────────────────────────────────────────

    class _spaces:
        Box      = _Box
        Discrete = _Discrete

    # ── Base Env class stub ───────────────────────────────────────────────

    class _Env:
        """
        Stub for gymnasium.Env base class.

        Subclasses must implement: reset(), step(), render(), close().
        """
        metadata          = {}
        action_space      = None
        observation_space = None

        def reset(self, seed=None, options=None):
            """
            Base reset: set random seeds if provided.
            Subclasses must call super().reset(seed=seed) first, then
            do their own logic and return (obs, info).
            """
            if seed is not None:
                np.random.seed(seed)
                _random.seed(seed)
            # Return None — subclass overrides this and returns (obs, info)

        def step(self, action):
            raise NotImplementedError(
                "Subclass must implement step(action) → "
                "(observation, reward, terminated, truncated, info)"
            )

        def render(self):
            pass

        def close(self):
            pass

        def __repr__(self):
            return f"<{self.__class__.__name__} (gym_shim)>"

    # ── gym namespace ─────────────────────────────────────────────────────

    class _gym:
        Env = _Env

    gym    = _gym
    spaces = _spaces

    # Silent — let support_env.py announce its own gym source
