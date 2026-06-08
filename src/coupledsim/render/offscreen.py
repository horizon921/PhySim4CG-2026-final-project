"""离屏渲染：把场景栅格化为 numpy 图像并保存 PNG。

无需窗口 / 显示器，可用于：
    - 生成报告插图、对比图、录屏帧序列
    - headless 环境下肉眼检查仿真是否正确
"""

import os

import numpy as np
import taichi as ti


def render_frame(scene, width: int = 512, vref: float = 3.0) -> np.ndarray:
    """返回 (W, H, 3) float32 图像（[0,1]），符合 ti.tools.imwrite 约定（x 右，y 上）。"""
    cfg = scene.cfg
    lx, ly = scene.lx, scene.ly
    aspect = ly / lx
    W = width
    H = int(round(width * aspect))

    img = np.empty((W, H, 3), dtype=np.float32)
    img[:] = np.array([0.03, 0.08, 0.13], dtype=np.float32)

    # 障碍物 / 墙：solid_phi < 0 的单元染灰
    phi = scene.solver.solid_phi.to_numpy()          # (nx, ny)
    nx, ny = phi.shape
    xi = np.clip((np.arange(W) / W * nx).astype(int), 0, nx - 1)
    yj = np.clip((np.arange(H) / H * ny).astype(int), 0, ny - 1)
    solid_mask = phi[np.ix_(xi, yj)] < 0.0           # (W, H)
    img[solid_mask] = np.array([0.27, 0.30, 0.36], dtype=np.float32)

    # 吸入口区域（红色淡染）
    if scene.sink is not None:
        x0, y0, x1, y1 = scene.sink
        sx0, sx1 = int(x0 / lx * W), int(x1 / lx * W)
        sy0, sy1 = int(y0 / ly * H), int(y1 / ly * H)
        img[sx0:sx1, sy0:sy1] = np.array([0.35, 0.12, 0.10], dtype=np.float32)

    # 粒子按速度着色，splat 到邻域
    pos = scene.solver.particle_positions_np()
    if len(pos) > 0:
        vel = scene.solver.particle_velocities_np()
        sp = np.linalg.norm(vel, axis=1)
        t = np.clip(sp / vref, 0.0, 1.0)
        col = np.stack([0.12 + 0.75 * t ** 1.6,
                        0.34 + 0.60 * t,
                        0.60 + 0.40 * (1.0 - (1.0 - t) ** 2)], axis=1).astype(np.float32)
        px = np.clip((pos[:, 0] / lx * W).astype(int), 0, W - 1)
        py = np.clip((pos[:, 1] / ly * H).astype(int), 0, H - 1)
        rad = max(1, int(round(0.45 * W / nx)))
        for ox in range(-rad, rad + 1):
            for oy in range(-rad, rad + 1):
                if ox * ox + oy * oy > rad * rad + 1:
                    continue
                ix = np.clip(px + ox, 0, W - 1)
                iy = np.clip(py + oy, 0, H - 1)
                img[ix, iy] = col
    return img


def save_png(img: np.ndarray, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ti.tools.imwrite(img, path)
