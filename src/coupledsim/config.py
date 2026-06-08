"""全局仿真配置。

这里集中管理网格、时间步、流体参数等，方便各物理模块（流体 / 软体 / 刚体）
共享同一套场景坐标与时间步约定。坐标系约定见 README 与 docs/framework.md。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class TransferMode(IntEnum):
    """粒子 <-> 网格 速度传输方式。

    PIC  : 完全用网格速度覆盖粒子速度，数值耗散大、最稳定。
    FLIP : 只把网格速度的“增量”加回粒子，保留细节但易抖动。
    APIC : 仿射 PIC，粒子携带仿射矩阵 C，低耗散且稳定。
    """

    PIC = 0
    FLIP = 1
    APIC = 2


@dataclass
class FluidConfig:
    """流体（FLIP/PIC/APIC）模块参数。"""

    # --- 网格 ---
    res_x: int = 128
    res_y: int = 128
    domain_x: float = 1.0          # 世界坐标下水箱宽度
    # domain_y 由 res 与 dx 推导，保证正方形单元

    # --- 时间步 ---
    dt: float = 1.0 / 60.0         # 每帧总时间
    substeps: int = 2              # 每帧子步数（提高稳定性）
    cfl: float = 1.0               # CFL 上限（自适应子步时用）
    adaptive_substeps: bool = True # 是否根据最大速度自动增加子步

    # --- 物理参数 ---
    gravity_x: float = 0.0
    gravity_y: float = -9.8
    rho: float = 1.0               # 流体密度
    flip_ratio: float = 0.97       # FLIP/PIC 混合比例（仅 FLIP 模式用）
    transfer: TransferMode = TransferMode.APIC
    vel_damping: float = 0.0       # 额外整体速度阻尼（1/s），耦合稳定性用

    # --- 求解 ---
    use_cg: bool = True            # True: Jacobi-PCG（推荐，收敛快）; False: 红黑 Gauss-Seidel
    pressure_iters: int = 80       # Gauss-Seidel 迭代数（use_cg=False 时）
    cg_max_iters: int = 120        # PCG 最大迭代数
    cg_tol: float = 1e-4           # PCG 相对残差阈值（视觉足够；做精度对比可调到 1e-6）
    extrapolate_iters: int = 4     # 速度外插迭代数

    # --- 粒子 ---
    particles_per_cell: int = 4    # 初始每个流体单元的粒子数（2x2 抖动）
    max_particles: int = 400_000   # 粒子缓冲上限（含发射余量）
    seed: int = 1234

    @property
    def dx(self) -> float:
        return self.domain_x / self.res_x

    @property
    def domain_y(self) -> float:
        return self.dx * self.res_y

    def grid_velocity_clamp(self) -> float:
        """允许的最大网格速度（数值安全，避免爆炸）。"""
        return 0.45 * self.dx / (self.dt / max(self.substeps, 1))


@dataclass
class SceneConfig:
    """场景 / 关卡级配置，组合各物理模块。"""

    name: str = "tank"
    fluid: FluidConfig = field(default_factory=FluidConfig)
    # 软体 / 刚体配置后续阶段补充
