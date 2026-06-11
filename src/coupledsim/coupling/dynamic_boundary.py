"""Rasterize moving soft-body surfaces into FLIP solid boundary fields."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .boundary import Shape, static_solid_phi


def _cell_center_points(nx: int, ny: int, nz: int, dx: float) -> np.ndarray:
    xs = (np.arange(nx, dtype=np.float32) + 0.5) * dx
    ys = (np.arange(ny, dtype=np.float32) + 0.5) * dx
    zs = (np.arange(nz, dtype=np.float32) + 0.5) * dx
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _face_points(nx: int, ny: int, nz: int, dx: float, axis: int) -> np.ndarray:
    if axis == 0:
        xs = np.arange(nx + 1, dtype=np.float32) * dx
        ys = (np.arange(ny, dtype=np.float32) + 0.5) * dx
        zs = (np.arange(nz, dtype=np.float32) + 0.5) * dx
    elif axis == 1:
        xs = (np.arange(nx, dtype=np.float32) + 0.5) * dx
        ys = np.arange(ny + 1, dtype=np.float32) * dx
        zs = (np.arange(nz, dtype=np.float32) + 0.5) * dx
    else:
        xs = (np.arange(nx, dtype=np.float32) + 0.5) * dx
        ys = (np.arange(ny, dtype=np.float32) + 0.5) * dx
        zs = np.arange(nz + 1, dtype=np.float32) * dx
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _collect_surface(soft_bodies: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    points = []
    velocities = []
    for body in soft_bodies:
        pts = np.asarray(body.surface_points_np(), dtype=np.float32).reshape(-1, 3)
        vel = np.asarray(body.surface_velocities_np(), dtype=np.float32).reshape(-1, 3)
        if len(pts) != len(vel):
            raise ValueError("soft body surface points and velocities must have the same length")
        if len(pts):
            points.append(pts)
            velocities.append(vel)
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0), np.concatenate(velocities, axis=0)


def point_cloud_sdf(query: np.ndarray, surface_points: np.ndarray, radius: float,
                    chunk: int = 32768) -> np.ndarray:
    """Approximate a solid SDF from surface samples swept by small spheres."""
    query = np.asarray(query, dtype=np.float32).reshape(-1, 3)
    surface_points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    if len(surface_points) == 0:
        return np.full(len(query), 1e9, dtype=np.float32)

    out = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        d = q[:, None, :] - surface_points[None, :, :]
        d2 = np.sum(d * d, axis=2)
        out[start:start + chunk] = np.sqrt(np.min(d2, axis=1)) - radius
    return out


def nearest_surface_velocity(query: np.ndarray, surface_points: np.ndarray,
                             surface_velocities: np.ndarray, active_radius: float,
                             component: int, chunk: int = 32768) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1, 3)
    surface_points = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    surface_velocities = np.asarray(surface_velocities, dtype=np.float32).reshape(-1, 3)
    values = np.zeros(len(query), dtype=np.float32)
    if len(surface_points) == 0:
        return values

    active2 = active_radius * active_radius
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        d = q[:, None, :] - surface_points[None, :, :]
        d2 = np.sum(d * d, axis=2)
        idx = np.argmin(d2, axis=1)
        nearest2 = d2[np.arange(len(q)), idx]
        local = np.zeros(len(q), dtype=np.float32)
        mask = nearest2 <= active2
        local[mask] = surface_velocities[idx[mask], component]
        values[start:start + chunk] = local
    return values


def soft_body_solid_phi(nx: int, ny: int, nz: int, dx: float,
                        soft_bodies: Iterable[object],
                        base_shapes: Iterable[Shape] | None = None,
                        with_walls: bool = True,
                        boundary_radius: float | None = None) -> np.ndarray:
    soft_bodies = list(soft_bodies)
    base_shapes = list(base_shapes or [])
    phi = static_solid_phi(nx, ny, nz, dx, base_shapes, with_walls=with_walls)
    queries = _cell_center_points(nx, ny, nz, dx)
    for body in soft_bodies:
        if hasattr(body, "sdf_np"):
            phi = np.minimum(phi.ravel(), np.asarray(body.sdf_np(queries), dtype=np.float32)).reshape(nx, ny, nz)

    pts, _ = _collect_surface(soft_bodies)
    if len(pts) == 0:
        return phi

    radius = float(boundary_radius if boundary_radius is not None else 0.75 * dx)
    soft_phi = point_cloud_sdf(queries, pts, radius)
    return np.minimum(phi.ravel(), soft_phi).reshape(nx, ny, nz).astype(np.float32)


def soft_body_solid_fields(nx: int, ny: int, nz: int, dx: float,
                           soft_bodies: Iterable[object],
                           base_shapes: Iterable[Shape] | None = None,
                           with_walls: bool = True,
                           boundary_radius: float | None = None,
                           velocity_radius: float | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    soft_bodies = list(soft_bodies)
    radius = float(boundary_radius if boundary_radius is not None else 0.75 * dx)
    active_radius = float(velocity_radius if velocity_radius is not None else radius + 1.5 * dx)
    phi = soft_body_solid_phi(nx, ny, nz, dx, soft_bodies, base_shapes, with_walls, radius)
    pts, vel = _collect_surface(soft_bodies)

    u = nearest_surface_velocity(_face_points(nx, ny, nz, dx, 0), pts, vel, active_radius, 0)
    v = nearest_surface_velocity(_face_points(nx, ny, nz, dx, 1), pts, vel, active_radius, 1)
    w = nearest_surface_velocity(_face_points(nx, ny, nz, dx, 2), pts, vel, active_radius, 2)
    return (
        phi,
        u.reshape(nx + 1, ny, nz).astype(np.float32),
        v.reshape(nx, ny + 1, nz).astype(np.float32),
        w.reshape(nx, ny, nz + 1).astype(np.float32),
    )
