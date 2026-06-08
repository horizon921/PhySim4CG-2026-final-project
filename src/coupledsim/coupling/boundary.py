"""固体边界 / 障碍物的统一表示。

这是“流固耦合”的关键接口：所有固体（静态障碍、可动刚体、软体）最终都被
栅格化为流体网格上的三个量：

    solid_phi[i, j]   单元中心到最近固体表面的有符号距离（< 0 表示在固体内部）
    u_solid[i, j]     竖直面（x 速度）上的固体法向速度
    v_solid[i, j]     水平面（y 速度）上的固体法向速度

流体求解器只依赖这三个量，因此后续接入软体 / 刚体时，只要它们能写出自己的
``solid_phi`` 与表面速度即可，无需改流体内核。这里先实现“静态解析障碍物”，
为第一阶段（稳定的 FLIP 水箱）提供边界，并作为耦合接口的参考实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# --------------------------------------------------------------------------- #
# 解析形状：返回世界坐标网格上的有符号距离（负数在内部）
# --------------------------------------------------------------------------- #
class Shape:
    def sdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Box(Shape):
    cx: float
    cy: float
    hx: float          # 半宽
    hy: float          # 半高

    def sdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        dx = np.abs(X - self.cx) - self.hx
        dy = np.abs(Y - self.cy) - self.hy
        outside = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2)
        inside = np.minimum(np.maximum(dx, dy), 0.0)
        return outside + inside


@dataclass
class Circle(Shape):
    cx: float
    cy: float
    r: float

    def sdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.sqrt((X - self.cx) ** 2 + (Y - self.cy) ** 2) - self.r


@dataclass
class DomainWalls(Shape):
    """水箱四壁：在区域内部为正，区域外为负（即墙体视为固体）。"""

    lx: float
    ly: float

    def sdf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.minimum.reduce([X, self.lx - X, Y, self.ly - Y])


def union_sdf(shapes: List[Shape], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if not shapes:
        return np.full_like(X, 1e9)
    phi = shapes[0].sdf(X, Y)
    for s in shapes[1:]:
        phi = np.minimum(phi, s.sdf(X, Y))
    return phi


def static_solid_phi(nx: int, ny: int, dx: float, shapes: List[Shape],
                     with_walls: bool = True, lx: float | None = None,
                     ly: float | None = None) -> np.ndarray:
    """在单元中心上采样固体有符号距离场，返回 (nx, ny) 的 numpy 数组。"""
    lx = lx if lx is not None else nx * dx
    ly = ly if ly is not None else ny * dx
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    all_shapes = list(shapes)
    if with_walls:
        all_shapes.append(DomainWalls(lx, ly))
    phi = union_sdf(all_shapes, X, Y)
    return phi.astype(np.float32)
