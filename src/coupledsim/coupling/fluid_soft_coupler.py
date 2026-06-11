"""Two-way helper between the FLIP solver and soft-body surface interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .boundary import Shape
from .dynamic_boundary import soft_body_solid_fields


def _trilerp_scalar(field: np.ndarray, points: np.ndarray, dx: float,
                   offset: tuple[float, float, float]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float32)

    shape = np.array(field.shape, dtype=np.int32)
    g = points / dx - np.asarray(offset, dtype=np.float32)[None, :]
    g = np.minimum(np.maximum(g, 0.0), (shape - 1).astype(np.float32))
    base = np.floor(g).astype(np.int32)
    base = np.minimum(base, shape[None, :] - 2)
    base = np.maximum(base, 0)
    f = g - base.astype(np.float32)

    i, j, k = base[:, 0], base[:, 1], base[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    c000 = field[i, j, k]
    c100 = field[i + 1, j, k]
    c010 = field[i, j + 1, k]
    c110 = field[i + 1, j + 1, k]
    c001 = field[i, j, k + 1]
    c101 = field[i + 1, j, k + 1]
    c011 = field[i, j + 1, k + 1]
    c111 = field[i + 1, j + 1, k + 1]
    return (
        c000 * (1 - fx) * (1 - fy) * (1 - fz)
        + c100 * fx * (1 - fy) * (1 - fz)
        + c010 * (1 - fx) * fy * (1 - fz)
        + c110 * fx * fy * (1 - fz)
        + c001 * (1 - fx) * (1 - fy) * fz
        + c101 * fx * (1 - fy) * fz
        + c011 * (1 - fx) * fy * fz
        + c111 * fx * fy * fz
    ).astype(np.float32)


def sample_mac_velocity_np(points: np.ndarray, u: np.ndarray, v: np.ndarray,
                           w: np.ndarray, dx: float) -> np.ndarray:
    return np.stack([
        _trilerp_scalar(u, points, dx, (0.0, 0.5, 0.5)),
        _trilerp_scalar(v, points, dx, (0.5, 0.0, 0.5)),
        _trilerp_scalar(w, points, dx, (0.5, 0.5, 0.0)),
    ], axis=1).astype(np.float32)


@dataclass
class FluidSoftCoupler:
    boundary_radius: float | None = None
    velocity_radius: float | None = None
    drag_coeff: float = 8.0
    max_point_force: float | None = 4.0

    def rasterize_soft_boundary(self, solver, soft_bodies: Iterable[object],
                                base_shapes: Iterable[Shape] | None = None):
        phi, u_solid, v_solid, w_solid = soft_body_solid_fields(
            solver.nx,
            solver.ny,
            solver.nz,
            solver.dx,
            soft_bodies,
            base_shapes=base_shapes,
            with_walls=True,
            boundary_radius=self.boundary_radius,
            velocity_radius=self.velocity_radius,
        )
        solver.set_solid_fields(phi, u_solid, v_solid, w_solid)
        return phi, u_solid, v_solid, w_solid

    def apply_fluid_forces(self, solver, soft_bodies: Iterable[object]):
        u = solver.u.to_numpy()
        v = solver.v.to_numpy()
        w = solver.w.to_numpy()
        for body in soft_bodies:
            points = np.asarray(body.surface_points_np(), dtype=np.float32).reshape(-1, 3)
            body_vel = np.asarray(body.surface_velocities_np(), dtype=np.float32).reshape(-1, 3)
            if len(points) == 0:
                body.apply_forces_np(np.zeros((0, 3), dtype=np.float32))
                continue
            fluid_vel = sample_mac_velocity_np(points, u, v, w, solver.dx)
            forces = self.drag_coeff * (fluid_vel - body_vel) / max(len(points), 1)
            if self.max_point_force is not None:
                norm = np.linalg.norm(forces, axis=1)
                mask = norm > self.max_point_force
                forces[mask] *= (self.max_point_force / norm[mask])[:, None]
            body.apply_forces_np(forces.astype(np.float32))
