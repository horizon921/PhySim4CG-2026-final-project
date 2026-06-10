"""XPBD soft body primitives and fluid coupling helpers.

The soft body is represented as a small lattice of particles connected by
distance constraints.  It is deliberately CPU/NumPy based: the lattice is tiny
compared with the FLIP particle set, and keeping it outside Taichi makes it easy
to test, tune, and later replace with a tetrahedral FEM implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


Vec3 = Tuple[float, float, float]


@dataclass
class XPBDConfig:
    """Parameters for a real-time jelly-like soft body."""

    spacing: float = 0.055
    radius: float = 0.038
    density: float = 0.35
    stiffness: float = 0.75
    damping: float = 2.5
    drag: float = 13.0
    lift: float = 0.0
    fluid_coupling: float = 1.0
    solver_iters: int = 8
    max_speed: float = 5.0


class XPBDSoftBody:
    """Small distance-constraint lattice soft body.

    The body stores positions/velocities in world coordinates.  Constraints use
    the XPBD compliance form, which keeps the solver stable even with fairly
    large frame time steps.
    """

    def __init__(self, positions: np.ndarray, constraints: Sequence[Tuple[int, int]],
                 cfg: XPBDConfig | None = None):
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n, 3)")
        self.cfg = cfg or XPBDConfig()
        self.x = positions.astype(np.float32).copy()
        self.prev_x = self.x.copy()
        self.v = np.zeros_like(self.x)
        self.inv_mass = np.full(len(self.x), 1.0 / max(self.cfg.density, 1e-6), np.float32)
        self.constraints: List[Tuple[int, int, float]] = []
        for a, b in constraints:
            rest = float(np.linalg.norm(self.x[a] - self.x[b]))
            if rest > 1e-8:
                self.constraints.append((int(a), int(b), rest))
        self._lambdas = np.zeros(len(self.constraints), np.float32)
        self._force = np.zeros_like(self.x)

    @classmethod
    def make_box(cls, center: Vec3 = (0.5, 0.64, 0.5), dims: Tuple[int, int, int] = (4, 4, 4),
                 spacing: float = 0.055, cfg: XPBDConfig | None = None) -> "XPBDSoftBody":
        """Create a cubical jelly lattice with structural and diagonal links."""
        nx, ny, nz = dims
        origin = np.array(center, np.float32) - 0.5 * spacing * np.array([nx - 1, ny - 1, nz - 1], np.float32)
        pts = []
        index = {}
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    index[(i, j, k)] = len(pts)
                    pts.append(origin + spacing * np.array([i, j, k], np.float32))

        cons = set()
        offsets = []
        for di in range(0, 3):
            for dj in range(0, 3):
                for dk in range(0, 3):
                    if di == dj == dk == 0:
                        continue
                    if max(di, dj, dk) <= 2 and (di, dj, dk) != (0, 0, 0):
                        offsets.append((di, dj, dk))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    a = index[(i, j, k)]
                    for di, dj, dk in offsets:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if ni < nx and nj < ny and nk < nz:
                            b = index[(ni, nj, nk)]
                            cons.add((min(a, b), max(a, b)))
        return cls(np.asarray(pts, np.float32), sorted(cons), cfg)

    def reset(self, other: "XPBDSoftBody") -> None:
        self.x[...] = other.x
        self.prev_x[...] = other.prev_x
        self.v.fill(0.0)
        self._force.fill(0.0)

    @property
    def center(self) -> np.ndarray:
        return self.x.mean(axis=0)

    def positions_np(self) -> np.ndarray:
        return self.x.copy()

    def velocities_np(self) -> np.ndarray:
        return self.v.copy()

    def add_force(self, force: np.ndarray) -> None:
        self._force += force.astype(np.float32)

    def apply_fluid_drag(self, fluid_velocity: np.ndarray) -> None:
        """Apply sampled fluid velocity as drag/lift forces at lattice nodes."""
        rel = fluid_velocity.astype(np.float32) - self.v
        self._force += self.cfg.fluid_coupling * self.cfg.drag * rel
        if self.cfg.lift != 0.0:
            self._force[:, 1] += self.cfg.lift

    def step(self, dt: float, bounds: Vec3, static_phi: np.ndarray | None = None,
             dx: float | None = None, gravity: Vec3 = (0.0, -9.8, 0.0)) -> None:
        self.prev_x[...] = self.x
        acc = np.array(gravity, np.float32)[None, :] + self._force * self.inv_mass[:, None]
        self.v += dt * acc
        self.v *= np.exp(-self.cfg.damping * dt)
        speed = np.linalg.norm(self.v, axis=1)
        fast = speed > self.cfg.max_speed
        if np.any(fast):
            self.v[fast] *= (self.cfg.max_speed / speed[fast])[:, None]
        self.x += dt * self.v

        self._lambdas.fill(0.0)
        compliance = max(0.0, (1.0 - self.cfg.stiffness)) * 1e-3
        alpha = compliance / max(dt * dt, 1e-12)
        for _ in range(max(1, self.cfg.solver_iters)):
            self._solve_distances(alpha)
            self._project_bounds(bounds)
            if static_phi is not None and dx is not None:
                self._project_static_solid(static_phi, dx)

        self.v = (self.x - self.prev_x) / max(dt, 1e-12)
        self._force.fill(0.0)

    def _solve_distances(self, alpha: float) -> None:
        for ci, (a, b, rest) in enumerate(self.constraints):
            d = self.x[a] - self.x[b]
            length = float(np.linalg.norm(d))
            if length < 1e-8:
                continue
            n = d / length
            c = length - rest
            w = self.inv_mass[a] + self.inv_mass[b]
            dlambda = (-c - alpha * self._lambdas[ci]) / (w + alpha)
            self._lambdas[ci] += dlambda
            self.x[a] += self.inv_mass[a] * dlambda * n
            self.x[b] -= self.inv_mass[b] * dlambda * n

    def _project_bounds(self, bounds: Vec3) -> None:
        r = self.cfg.radius
        hi = np.array(bounds, np.float32) - r
        lo = np.full(3, r, np.float32)
        self.x[:] = np.minimum(np.maximum(self.x, lo), hi)

    def _project_static_solid(self, phi: np.ndarray, dx: float) -> None:
        nx, ny, nz = phi.shape
        for p in range(len(self.x)):
            i, j, k = np.clip((self.x[p] / dx).astype(np.int32), [0, 0, 0], [nx - 1, ny - 1, nz - 1])
            if phi[i, j, k] >= self.cfg.radius * 0.35:
                continue
            grad = np.zeros(3, np.float32)
            for axis in range(3):
                lo = [i, j, k]
                hi = [i, j, k]
                lo[axis] = max(0, lo[axis] - 1)
                hi[axis] = min(phi.shape[axis] - 1, hi[axis] + 1)
                grad[axis] = phi[tuple(hi)] - phi[tuple(lo)]
            n = grad / max(float(np.linalg.norm(grad)), 1e-8)
            self.x[p] += (self.cfg.radius * 0.35 - phi[i, j, k]) * n


def rasterize_soft_bodies(base_phi: np.ndarray, bodies: Iterable[XPBDSoftBody],
                          dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize soft bodies into FLIP solid fields.

    Returns ``solid_phi, u_solid, v_solid, w_solid``.  SDF samples are built as
    the union of moving spheres centered at XPBD particles.
    """
    phi = base_phi.copy()
    nx, ny, nz = phi.shape
    u = np.zeros((nx + 1, ny, nz), np.float32)
    v = np.zeros((nx, ny + 1, nz), np.float32)
    w = np.zeros((nx, ny, nz + 1), np.float32)
    uw = np.zeros_like(u)
    vw = np.zeros_like(v)
    ww = np.zeros_like(w)

    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    zs = (np.arange(nz) + 0.5) * dx

    for body in bodies:
        r = body.cfg.radius
        band = r + 1.75 * dx
        for p, vel in zip(body.x, body.v):
            i0 = max(0, int(np.floor((p[0] - band) / dx)))
            i1 = min(nx - 1, int(np.ceil((p[0] + band) / dx)))
            j0 = max(0, int(np.floor((p[1] - band) / dx)))
            j1 = min(ny - 1, int(np.ceil((p[1] + band) / dx)))
            k0 = max(0, int(np.floor((p[2] - band) / dx)))
            k1 = min(nz - 1, int(np.ceil((p[2] + band) / dx)))
            X, Y, Z = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], zs[k0:k1 + 1], indexing="ij")
            d = np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2) - r
            phi[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = np.minimum(
                phi[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1], d.astype(np.float32)
            )
            _splat_face_velocity(u, uw, p, vel[0], r, dx, offset=(0.0, 0.5, 0.5))
            _splat_face_velocity(v, vw, p, vel[1], r, dx, offset=(0.5, 0.0, 0.5))
            _splat_face_velocity(w, ww, p, vel[2], r, dx, offset=(0.5, 0.5, 0.0))

    np.divide(u, uw, out=u, where=uw > 1e-8)
    np.divide(v, vw, out=v, where=vw > 1e-8)
    np.divide(w, ww, out=w, where=ww > 1e-8)
    return phi, u, v, w


def _splat_face_velocity(field: np.ndarray, weight: np.ndarray, center: np.ndarray,
                         value: float, radius: float, dx: float, offset: Vec3) -> None:
    sx, sy, sz = field.shape
    band = radius + dx
    lo = np.maximum(np.floor(center / dx - np.array(offset) - band / dx).astype(int), 0)
    hi = np.minimum(np.ceil(center / dx - np.array(offset) + band / dx).astype(int),
                    np.array([sx - 1, sy - 1, sz - 1]))
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):
                pos = (np.array([i, j, k], np.float32) + np.array(offset, np.float32)) * dx
                d = float(np.linalg.norm(pos - center))
                if d <= band:
                    wgt = max(0.0, 1.0 - d / band)
                    field[i, j, k] += wgt * value
                    weight[i, j, k] += wgt
