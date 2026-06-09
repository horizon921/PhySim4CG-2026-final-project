"""固体边界 / 障碍物的统一表示（三维）。

这是“流固耦合”的关键接口：所有固体（静态障碍、可动刚体、软体）最终都被
栅格化为流体网格上的若干量：

    solid_phi[i,j,k]   单元中心到最近固体表面的有符号距离（< 0 表示在固体内部）
    u_solid[i,j,k]     x 面上的固体法向速度
    v_solid[i,j,k]     y 面上的固体法向速度
    w_solid[i,j,k]     z 面上的固体法向速度

流体求解器只依赖这些量，因此后续接入软体 / 刚体时，只要它们能写出自己的
``solid_phi`` 与表面速度即可，无需改流体内核。这里先实现“静态解析障碍物”，
为稳定的 FLIP 水箱提供边界，并作为耦合接口的参考实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# --------------------------------------------------------------------------- #
# 解析形状：返回世界坐标网格上的有符号距离（负数在内部）
# --------------------------------------------------------------------------- #
class Shape:
    def sdf(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Box(Shape):
    cx: float
    cy: float
    cz: float
    hx: float          # 半宽 (x)
    hy: float          # 半高 (y)
    hz: float          # 半深 (z)

    def sdf(self, X, Y, Z):
        dx = np.abs(X - self.cx) - self.hx
        dy = np.abs(Y - self.cy) - self.hy
        dz = np.abs(Z - self.cz) - self.hz
        outside = np.sqrt(np.maximum(dx, 0.0) ** 2
                          + np.maximum(dy, 0.0) ** 2
                          + np.maximum(dz, 0.0) ** 2)
        inside = np.minimum(np.maximum(np.maximum(dx, dy), dz), 0.0)
        return outside + inside


@dataclass
class Sphere(Shape):
    cx: float
    cy: float
    cz: float
    r: float

    def sdf(self, X, Y, Z):
        return np.sqrt((X - self.cx) ** 2 + (Y - self.cy) ** 2 + (Z - self.cz) ** 2) - self.r


@dataclass
class DomainWalls(Shape):
    """水箱六壁：在区域内部为正，区域外为负（即墙体视为固体）。"""

    lx: float
    ly: float
    lz: float

    def sdf(self, X, Y, Z):
        return np.minimum.reduce([X, self.lx - X, Y, self.ly - Y, Z, self.lz - Z])


def union_sdf(shapes: List[Shape], X, Y, Z) -> np.ndarray:
    if not shapes:
        return np.full_like(X, 1e9)
    phi = shapes[0].sdf(X, Y, Z)
    for s in shapes[1:]:
        phi = np.minimum(phi, s.sdf(X, Y, Z))
    return phi


def static_solid_phi(nx: int, ny: int, nz: int, dx: float, shapes: List[Shape],
                     with_walls: bool = True, lx: float | None = None,
                     ly: float | None = None, lz: float | None = None) -> np.ndarray:
    """在单元中心上采样固体有符号距离场，返回 (nx, ny, nz) 的 numpy 数组。"""
    lx = lx if lx is not None else nx * dx
    ly = ly if ly is not None else ny * dx
    lz = lz if lz is not None else nz * dx
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    zs = (np.arange(nz) + 0.5) * dx
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    all_shapes = list(shapes)
    if with_walls:
        all_shapes.append(DomainWalls(lx, ly, lz))
    phi = union_sdf(all_shapes, X, Y, Z)
    return phi.astype(np.float32)
