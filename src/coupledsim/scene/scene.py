"""场景 / 关卡组装层。

把流体求解器、固体形状、初始水块、喷口（发射器）、吸入口（sink）组合成一个
可推进、可重置的场景。后续阶段的软体 / 刚体也将通过这一层接入同一个流体网格。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..config import FluidConfig
from ..coupling import Shape, static_solid_phi
from ..fluid import FlipSolver

Region = Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class Emitter:
    """喷口：每帧在矩形区域内发射带初速度的粒子。"""
    region: Region
    velocity: Tuple[float, float]
    count: int
    enabled: bool = True


@dataclass
class FluidScene:
    name: str
    cfg: FluidConfig
    shapes: List[Shape] = field(default_factory=list)
    init_blocks: List[Region] = field(default_factory=list)
    emitters: List[Emitter] = field(default_factory=list)
    sink: Optional[Region] = None
    hint: str = ""

    def __post_init__(self):
        self.solver = FlipSolver(self.cfg)
        self._build_solid()
        self.frame = 0
        self.reset()

    @property
    def lx(self) -> float:
        return self.cfg.res_x * self.cfg.dx

    @property
    def ly(self) -> float:
        return self.cfg.res_y * self.cfg.dx

    def _build_solid(self):
        phi = static_solid_phi(self.cfg.res_x, self.cfg.res_y, self.cfg.dx,
                               shapes=self.shapes, with_walls=True)
        self.solver.set_solid_phi(phi)

    def reset(self):
        self.solver.n_particles[None] = 0
        for region in self.init_blocks:
            self.solver.add_particle_block(*region)
        self.frame = 0

    def step(self) -> int:
        for e in self.emitters:
            if e.enabled and self.solver.n_particles[None] < self.cfg.max_particles:
                self.solver.emit_block(*e.region, e.velocity[0], e.velocity[1], e.count)
        nsub = self.solver.step()
        # 有吸入口时每帧压缩；否则接近上限时清理一次出界粒子
        if self.sink is not None:
            self.solver.compact(self.sink)
        elif self.solver.n_particles[None] > 0.98 * self.cfg.max_particles:
            self.solver.compact(None)
        self.frame += 1
        return nsub

    @property
    def n_particles(self) -> int:
        return self.solver.n_particles[None]
