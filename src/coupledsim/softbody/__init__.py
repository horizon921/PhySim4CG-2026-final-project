"""软体模块（XPBD）。

软体通过两条路径与流体耦合：
    - 流体 -> 软体：在软体节点处采样流体速度，施加拖拽 / 浮力近似。
    - 软体 -> 流体：把软体节点栅格化为运动球 SDF，写入 solid_phi 与固体面速度。
"""

from .xpbd import XPBDConfig, XPBDSoftBody, rasterize_soft_bodies

__all__ = ["XPBDConfig", "XPBDSoftBody", "rasterize_soft_bodies"]
