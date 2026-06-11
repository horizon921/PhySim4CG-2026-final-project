"""Lightweight soft-body proxy used by the coupling layer.

The real soft-body solver can replace this class as long as it exposes the
same NumPy-facing surface interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _fibonacci_sphere(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    i = np.arange(n, dtype=np.float32)
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1).astype(np.float32)


@dataclass
class KinematicSoftBody:
    """Small mass-lumped soft object for coupling tests and demo scenes."""

    center: tuple[float, float, float] = (0.5, 0.55, 0.5)
    radius: float = 0.12
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 1.0
    surface_samples: int = 96
    damping: float = 0.03
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None

    _center0: np.ndarray = field(init=False, repr=False)
    _velocity0: np.ndarray = field(init=False, repr=False)
    _center: np.ndarray = field(init=False, repr=False)
    _velocity: np.ndarray = field(init=False, repr=False)
    _pending_force: np.ndarray = field(init=False, repr=False)
    _dirs: np.ndarray = field(init=False, repr=False)
    last_force: np.ndarray = field(init=False)

    def __post_init__(self):
        self._center0 = np.asarray(self.center, dtype=np.float32)
        self._velocity0 = np.asarray(self.velocity, dtype=np.float32)
        self._center = self._center0.copy()
        self._velocity = self._velocity0.copy()
        self._pending_force = np.zeros(3, dtype=np.float32)
        self.last_force = np.zeros(3, dtype=np.float32)
        self._dirs = _fibonacci_sphere(self.surface_samples)

    @property
    def position(self) -> np.ndarray:
        return self._center.copy()

    @property
    def linear_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def reset(self):
        self._center = self._center0.copy()
        self._velocity = self._velocity0.copy()
        self._pending_force.fill(0.0)
        self.last_force.fill(0.0)

    def surface_points_np(self) -> np.ndarray:
        return self._center[None, :] + self.radius * self._dirs

    def surface_velocities_np(self) -> np.ndarray:
        n = len(self._dirs)
        return np.repeat(self._velocity[None, :], n, axis=0).astype(np.float32)

    def sdf_np(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        return (np.linalg.norm(points - self._center[None, :], axis=1) - self.radius).astype(np.float32)

    def apply_forces_np(self, forces: np.ndarray):
        if forces.size == 0:
            self.last_force.fill(0.0)
            return
        total = np.asarray(forces, dtype=np.float32).reshape(-1, 3).sum(axis=0)
        self._pending_force += total
        self.last_force = total.astype(np.float32)

    def step(self, dt: float):
        inv_mass = 1.0 / max(float(self.mass), 1e-6)
        self._velocity += dt * inv_mass * self._pending_force
        self._velocity *= max(0.0, 1.0 - self.damping * dt)
        self._center += dt * self._velocity
        self._pending_force.fill(0.0)
        self._project_bounds()

    def _project_bounds(self):
        if self.bounds_min is None or self.bounds_max is None:
            return
        lo = np.asarray(self.bounds_min, dtype=np.float32) + self.radius
        hi = np.asarray(self.bounds_max, dtype=np.float32) - self.radius
        old = self._center.copy()
        self._center = np.minimum(np.maximum(self._center, lo), hi)
        hit = np.abs(self._center - old) > 1e-7
        self._velocity[hit] *= -0.25
