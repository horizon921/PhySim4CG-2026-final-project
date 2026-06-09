"""交互式三维查看器（CPU 友好）。

GGUI（ti.ui.Window）需要 GPU 后端（macOS 上为 vulkan），而本项目流体计算在
CPU/metal 上运行、且 GGUI 的 GPU 后端在本机会卡住。因此交互窗口采用更稳妥的
方案：用 ``ti.GUI``（CPU 软件光栅）+ ``set_image`` 显示离屏投影出来的三维点云，
鼠标左键拖动旋转相机。所有渲染逻辑复用 ``offscreen.render_frame``，已 headless 验证。
"""

import taichi as ti

from .offscreen import render_frame


class Viewer3D:
    def __init__(self, scene, window_size: int = 720, title: str | None = None,
                 show_gui: bool = True, azim: float = 35.0, elev: float = 22.0):
        self.scene = scene
        self.W = window_size
        self.azim = azim
        self.elev = elev
        self.gui = ti.GUI(title or f"coupledsim · {scene.name}",
                          res=(self.W, self.W), show_gui=show_gui)

    def rotate(self, dazim: float, delev: float):
        self.azim += dazim
        self.elev = max(-85.0, min(85.0, self.elev + delev))

    def draw(self, info_lines=None, save_path=None):
        img = render_frame(self.scene, width=self.W, azim=self.azim, elev=self.elev)
        self.gui.set_image(img)
        if info_lines:
            y = 0.975
            for line in info_lines:
                self.gui.text(line, pos=(0.012, y), font_size=18, color=0xF0F4FA)
                y -= 0.035
        return self.gui.show(save_path)

    @property
    def running(self) -> bool:
        return self.gui.running
