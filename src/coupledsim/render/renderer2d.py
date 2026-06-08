"""二维可视化（基于 ti.GUI，CPU/GPU 均可，macOS 友好）。

把世界坐标 [0,Lx]x[0,Ly] 映射到 GUI 归一化坐标 [0,1]^2：
    - 粒子按速度大小着色（深蓝 -> 青 -> 白）
    - 障碍物（盒/圆）填充为灰色
    - 吸入口区域以红色轮廓标出
    - 左上角绘制 HUD（模式、粒子数、FPS、操作提示）
"""

import numpy as np
import taichi as ti

from ..coupling import Box, Circle


def _pack_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1)
    r = (rgb[:, 0] * 255).astype(np.uint32)
    g = (rgb[:, 1] * 255).astype(np.uint32)
    b = (rgb[:, 2] * 255).astype(np.uint32)
    return (r << 16) | (g << 8) | b


class Renderer2D:
    def __init__(self, scene, window_size: int = 720, title: str | None = None):
        self.scene = scene
        self.cfg = scene.cfg
        self.lx = scene.lx
        self.ly = scene.ly
        aspect = self.ly / self.lx
        self.W = window_size
        self.H = int(round(window_size * aspect))
        self.gui = ti.GUI(title or f"coupledsim · {scene.name}",
                          res=(self.W, self.H), background_color=0x081420)
        self._prep_obstacles()
        # 粒子像素半径：约占一个网格单元的一半
        self.particle_radius_px = max(1.3, 0.42 * self.W / self.cfg.res_x)

    # ---- 预处理障碍几何（归一化坐标） ----
    def _prep_obstacles(self):
        ta, tb, tc = [], [], []
        self.draw_circles = []
        for s in self.scene.shapes:
            if isinstance(s, Box):
                x0 = (s.cx - s.hx) / self.lx
                x1 = (s.cx + s.hx) / self.lx
                y0 = (s.cy - s.hy) / self.ly
                y1 = (s.cy + s.hy) / self.ly
                ta += [[x0, y0], [x0, y0]]
                tb += [[x1, y0], [x1, y1]]
                tc += [[x1, y1], [x0, y1]]
            elif isinstance(s, Circle):
                self.draw_circles.append(((s.cx / self.lx, s.cy / self.ly),
                                          s.r / self.lx))
        self.tri_a = np.array(ta, dtype=np.float32) if ta else None
        self.tri_b = np.array(tb, dtype=np.float32) if ta else None
        self.tri_c = np.array(tc, dtype=np.float32) if ta else None

    def _to_norm(self, pos: np.ndarray) -> np.ndarray:
        out = np.empty_like(pos)
        out[:, 0] = pos[:, 0] / self.lx
        out[:, 1] = pos[:, 1] / self.ly
        return out

    def _speed_colors(self, speeds: np.ndarray, vref: float = 3.0) -> np.ndarray:
        t = np.clip(speeds / vref, 0.0, 1.0)
        r = 0.12 + 0.75 * t ** 1.6
        g = 0.34 + 0.60 * t
        b = 0.60 + 0.40 * (1.0 - (1.0 - t) ** 2)
        return _pack_rgb(np.stack([r, g, b], axis=1))

    def draw(self, info_lines=None):
        gui = self.gui
        obstacle_color = 0x46505E

        # 障碍物（先画，粒子覆盖其上）
        if self.tri_a is not None:
            gui.triangles(self.tri_a, self.tri_b, self.tri_c, color=obstacle_color)
        for center, r in self.draw_circles:
            gui.circle(center, color=obstacle_color, radius=r * self.W)

        # 吸入口
        if self.scene.sink is not None:
            x0, y0, x1, y1 = self.scene.sink
            gui.rect((x0 / self.lx, y0 / self.ly), (x1 / self.lx, y1 / self.ly),
                     radius=2, color=0xE0533D)

        # 粒子
        pos = self.scene.solver.particle_positions_np()
        if len(pos) > 0:
            vel = self.scene.solver.particle_velocities_np()
            sp = np.linalg.norm(vel, axis=1)
            colors = self._speed_colors(sp)
            gui.circles(self._to_norm(pos), radius=self.particle_radius_px, color=colors)

        # HUD
        if info_lines:
            y = 0.97
            for line in info_lines:
                gui.text(line, pos=(0.012, y), font_size=18, color=0xE8F0F8)
                y -= 0.035
        return gui.show()

    @property
    def running(self) -> bool:
        return self.gui.running
