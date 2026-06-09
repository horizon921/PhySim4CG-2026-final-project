"""离屏渲染：把三维场景正交投影成 2D 图像并保存 PNG。

无需窗口 / 显示器，可用于：
    - 生成报告插图、对比图、录屏帧序列
    - headless 环境下肉眼检查三维仿真是否正确

相机为正交投影，由方位角 azim / 仰角 elev 指定视向；粒子按速度着色、按深度
做明暗（近亮远暗），用画家算法（远 -> 近）叠绘。障碍体素染灰一并绘制。
"""

import os

import numpy as np
import taichi as ti


def _camera_basis(azim_deg: float, elev_deg: float):
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    forward = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward


def render_frame(scene, width: int = 600, azim: float = 35.0, elev: float = 22.0,
                 vref: float = 3.0) -> np.ndarray:
    """返回 (W, W, 3) float32 图像（[0,1]），ti.tools.imwrite 约定 (x 右, y 上)。"""
    W = width
    H = width
    img = np.empty((W, H, 3), dtype=np.float32)
    img[:] = np.array([0.03, 0.06, 0.11], dtype=np.float32)

    lx, ly, lz = scene.lx, scene.ly, scene.lz
    center = np.array([lx * 0.5, ly * 0.5, lz * 0.5])
    diag = np.sqrt(lx * lx + ly * ly + lz * lz)
    scale = 0.78 * W / diag
    right, up, forward = _camera_basis(azim, elev)

    pts_list, col_list = [], []

    # 障碍体素（solid_phi<0 的内部单元；域壁因中心处 phi>0 不会被选中）
    phi = scene.solver.solid_phi.to_numpy()
    nx, ny, nz = phi.shape
    dx = scene.cfg.dx
    si, sj, sk = np.where(phi < 0.0)
    if len(si) > 0:
        ob = np.stack([(si + 0.5) * dx, (sj + 0.5) * dx, (sk + 0.5) * dx], axis=1)
        pts_list.append(ob)
        col_list.append(np.tile(np.array([0.30, 0.33, 0.40], np.float32), (len(ob), 1)))

    # 流体粒子
    pos = scene.solver.particle_positions_np()
    if len(pos) > 0:
        vel = scene.solver.particle_velocities_np()
        sp = np.linalg.norm(vel, axis=1)
        t = np.clip(sp / vref, 0.0, 1.0)
        col = np.stack([0.12 + 0.75 * t ** 1.6,
                        0.34 + 0.60 * t,
                        0.62 + 0.38 * (1.0 - (1.0 - t) ** 2)], axis=1).astype(np.float32)
        pts_list.append(pos)
        col_list.append(col)

    if not pts_list:
        return img

    P = np.concatenate(pts_list, axis=0)
    Col = np.concatenate(col_list, axis=0)

    rel = P - center
    sx = rel @ right
    sy = rel @ up
    depth = rel @ forward

    # 画家算法：远 -> 近
    order = np.argsort(-depth)
    sx, sy, depth, Col = sx[order], sy[order], depth[order], Col[order]

    # 深度明暗（近亮远暗）
    dmin, dmax = depth.min(), depth.max()
    shade = 0.55 + 0.45 * (1.0 - (depth - dmin) / max(dmax - dmin, 1e-6))
    Col = Col * shade[:, None]

    px = np.clip((W * 0.5 + sx * scale).astype(int), 0, W - 1)
    py = np.clip((H * 0.5 + sy * scale).astype(int), 0, H - 1)

    rad = max(1, int(round(0.42 * scale * dx)))
    for ox in range(-rad, rad + 1):
        for oy in range(-rad, rad + 1):
            if ox * ox + oy * oy > rad * rad + 1:
                continue
            ix = np.clip(px + ox, 0, W - 1)
            iy = np.clip(py + oy, 0, H - 1)
            img[ix, iy] = Col

    # 画出水箱底面与背面边框，提供空间参考
    _draw_box_wire(img, center, scale, right, up, forward, lx, ly, lz, W, H)
    return img


def _draw_box_wire(img, center, scale, right, up, forward, lx, ly, lz, W, H):
    corners = np.array([[x, y, z] for x in (0, lx) for y in (0, ly) for z in (0, lz)], np.float32)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    rel = corners - center
    sx = rel @ right
    sy = rel @ up
    cx = (W * 0.5 + sx * scale)
    cy = (H * 0.5 + sy * scale)
    line_col = np.array([0.20, 0.24, 0.30], np.float32)
    for a, b in edges:
        n = int(max(abs(cx[a] - cx[b]), abs(cy[a] - cy[b])) + 1)
        for t in np.linspace(0, 1, n):
            ix = int(np.clip(cx[a] * (1 - t) + cx[b] * t, 0, W - 1))
            iy = int(np.clip(cy[a] * (1 - t) + cy[b] * t, 0, H - 1))
            img[ix, iy] = line_col


def save_png(img: np.ndarray, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ti.tools.imwrite(img, path)
