"""场景 / 关卡组装层（三维）。

把流体求解器、固体形状、初始水块、喷口（发射器）、吸入口（sink）组合成一个
可推进、可重置的场景。后续阶段的软体 / 刚体也将通过这一层接入同一个流体网格。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..config import FluidConfig
from ..coupling import Shape, static_solid_phi
from ..fluid import FlipSolver
from ..softbody import XPBDSoftBody, rasterize_soft_bodies

Region = Tuple[float, float, float, float, float, float]  # (x0, y0, z0, x1, y1, z1)


@dataclass
class Emitter:
    """喷口：每帧在长方体区域内发射带初速度的粒子。"""
    region: Region
    velocity: Tuple[float, float, float]
    count: int
    enabled: bool = True


@dataclass
class FluidScene:
    name: str
    cfg: FluidConfig
    shapes: List[Shape] = field(default_factory=list)
    init_blocks: List[Region] = field(default_factory=list)
    emitters: List[Emitter] = field(default_factory=list)
    soft_bodies: List[XPBDSoftBody] = field(default_factory=list)
    sink: Optional[Region] = None
    hint: str = ""

    def __post_init__(self):
        self.solver = FlipSolver(self.cfg)
        self._initial_soft_bodies = [
            XPBDSoftBody(body.positions_np(), [(a, b) for a, b, _rest in body.constraints], body.cfg)
            for body in self.soft_bodies
        ]
        self._build_solid()
        self.frame = 0
        self.reset()

    @property
    def lx(self) -> float:
        return self.cfg.res_x * self.cfg.dx

    @property
    def ly(self) -> float:
        return self.cfg.res_y * self.cfg.dx

    @property
    def lz(self) -> float:
        return self.cfg.res_z * self.cfg.dx

    def _build_solid(self):
        self.static_phi = static_solid_phi(self.cfg.res_x, self.cfg.res_y, self.cfg.res_z, self.cfg.dx,
                                           shapes=self.shapes, with_walls=True)
        self._update_dynamic_solid()

    def _update_dynamic_solid(self):
        if self.soft_bodies:
            phi, u, v, w = rasterize_soft_bodies(self.static_phi, self.soft_bodies, self.cfg.dx)
            self.solver.set_solid_phi(phi)
            self.solver.u_solid.from_numpy(u)
            self.solver.v_solid.from_numpy(v)
            self.solver.w_solid.from_numpy(w)
        else:
            self.solver.set_solid_phi(self.static_phi)

    def _sample_fluid_for_soft_bodies(self) -> List[np.ndarray]:
        samples: List[np.ndarray] = []
        if not self.soft_bodies:
            return samples
        u = self.solver.u.to_numpy()
        v = self.solver.v.to_numpy()
        w = self.solver.w.to_numpy()
        for body in self.soft_bodies:
            vel = np.zeros_like(body.x)
            for i, p in enumerate(body.x):
                vel[i] = _sample_mac_velocity_np(u, v, w, p, self.cfg.dx)
            samples.append(vel)
        return samples

    def reset(self):
        for body, initial in zip(self.soft_bodies, self._initial_soft_bodies):
            body.reset(initial)
        self._update_dynamic_solid()
        self.solver.n_particles[None] = 0
        for region in self.init_blocks:
            self.solver.add_particle_block(*region)
        self.frame = 0

    def step(self) -> int:
        self._update_dynamic_solid()
        for e in self.emitters:
            if e.enabled and self.solver.n_particles[None] < self.cfg.max_particles:
                self.solver.emit_block(*e.region, e.velocity[0], e.velocity[1], e.velocity[2], e.count)
        nsub = self.solver.step()
        if self.soft_bodies:
            fluid_vel = self._sample_fluid_for_soft_bodies()
            sdt = self.cfg.dt / max(nsub, 1)
            bounds = (self.lx, self.ly, self.lz)
            gravity = (self.cfg.gravity_x, self.cfg.gravity_y, self.cfg.gravity_z)
            for _ in range(nsub):
                for body, vel in zip(self.soft_bodies, fluid_vel):
                    body.apply_fluid_drag(vel)
                    body.step(sdt, bounds=bounds, static_phi=self.static_phi, dx=self.cfg.dx,
                              gravity=gravity)
            self._update_dynamic_solid()
        if self.sink is not None:
            self.solver.compact(self.sink)
        elif self.solver.n_particles[None] > 0.98 * self.cfg.max_particles:
            self.solver.compact(None)
        self.frame += 1
        return nsub

    @property
    def n_particles(self) -> int:
        return self.solver.n_particles[None]


def _trilerp_np(arr: np.ndarray, gx: float, gy: float, gz: float) -> float:
    sx, sy, sz = arr.shape
    gx = float(np.clip(gx, 0.0, sx - 1.0))
    gy = float(np.clip(gy, 0.0, sy - 1.0))
    gz = float(np.clip(gz, 0.0, sz - 1.0))
    i = min(int(gx), sx - 2)
    j = min(int(gy), sy - 2)
    k = min(int(gz), sz - 2)
    fx, fy, fz = gx - i, gy - j, gz - k
    c00 = arr[i, j, k] * (1 - fx) + arr[i + 1, j, k] * fx
    c10 = arr[i, j + 1, k] * (1 - fx) + arr[i + 1, j + 1, k] * fx
    c01 = arr[i, j, k + 1] * (1 - fx) + arr[i + 1, j, k + 1] * fx
    c11 = arr[i, j + 1, k + 1] * (1 - fx) + arr[i + 1, j + 1, k + 1] * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return float(c0 * (1 - fz) + c1 * fz)


def _sample_mac_velocity_np(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                            x: np.ndarray, dx: float) -> np.ndarray:
    inv_dx = 1.0 / dx
    return np.array([
        _trilerp_np(u, x[0] * inv_dx, x[1] * inv_dx - 0.5, x[2] * inv_dx - 0.5),
        _trilerp_np(v, x[0] * inv_dx - 0.5, x[1] * inv_dx, x[2] * inv_dx - 0.5),
        _trilerp_np(w, x[0] * inv_dx - 0.5, x[1] * inv_dx - 0.5, x[2] * inv_dx),
    ], dtype=np.float32)
