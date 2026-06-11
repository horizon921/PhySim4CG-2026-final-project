"""交互式三维查看器（CPU 友好）。

GGUI（ti.ui.Window）需要 GPU 后端（macOS 上为 vulkan），而本项目流体计算在
CPU/metal 上运行、且 GGUI 的 GPU 后端在本机会卡住。因此交互窗口采用更稳妥的
方案：用 ``ti.GUI``（CPU 软件光栅）+ ``set_image`` 显示离屏投影出来的三维点云，
鼠标左键拖动旋转相机。所有渲染逻辑复用 ``offscreen.render_frame``，已 headless 验证。
"""

import numpy as np
import taichi as ti

from .offscreen import (
    render_frame,
    render_map_frame,
    _camera_basis,
    _draw_game_overlay,
    _map_bounds,
    _project_point,
)


class Viewer3D:
    def __init__(self, scene, window_size: int = 720, title: str | None = None,
                 show_gui: bool = True, azim: float = 35.0, elev: float = 22.0,
                 max_fluid_particles: int | None = None, render_scale: float = 1.0,
                 color_by_velocity: bool = False, draw_solid_voxels: bool = False,
                 hud_detail: str = "full", view_mode: str = "3d"):
        self.scene = scene
        self.W = window_size
        self.azim = azim
        self.elev = elev
        self.max_fluid_particles = max_fluid_particles
        self.render_scale = max(0.4, min(1.0, float(render_scale)))
        self.color_by_velocity = color_by_velocity
        self.draw_solid_voxels = draw_solid_voxels
        self.hud_detail = hud_detail
        self.view_mode = view_mode
        self._scale_cache: tuple[int, np.ndarray] | None = None
        self.gui = ti.GUI(title or f"coupledsim · {scene.name}",
                          res=(self.W, self.W), show_gui=show_gui)

    def rotate(self, dazim: float, delev: float):
        self.azim += dazim
        self.elev = max(-85.0, min(85.0, self.elev + delev))

    def draw(self, info_lines=None, save_path=None):
        render_w = max(160, int(round(self.W * self.render_scale)))
        if self.view_mode == "map":
            img = render_map_frame(self.scene, width=render_w,
                                   max_fluid_particles=self.max_fluid_particles,
                                   draw_ui_overlay=False)
        else:
            img = render_frame(self.scene, width=render_w, azim=self.azim, elev=self.elev,
                               max_fluid_particles=self.max_fluid_particles,
                               draw_ui_overlay=False,
                               color_by_velocity=self.color_by_velocity,
                               draw_solid_voxels=self.draw_solid_voxels)
        if render_w != self.W:
            if self._scale_cache is None or self._scale_cache[0] != render_w:
                self._scale_cache = (render_w, np.linspace(0, render_w - 1, self.W).astype(np.int32))
            idx = self._scale_cache[1]
            img = img[idx][:, idx]
        _draw_game_overlay(img, self.scene, self.W, self.W)
        self._paint_end_state_panel(img)
        self.gui.set_image(img)
        self._draw_world_labels()
        self._draw_game_ui_labels()
        if info_lines:
            y = 0.700 if hasattr(self.scene, "game_status") else 0.975
            max_chars = max(56, self.W // 10)
            for line in info_lines:
                if len(line) > max_chars:
                    line = line[:max_chars - 3] + "..."
                self._label(line, 0.012, y, size=12, color=0xEAF2FF)
                y -= 0.026
        self._draw_end_state_message()
        return self.gui.show(save_path)

    def _text(self, text: str, x: float, y: float, size: int = 15, color: int = 0xEAF2FF):
        self.gui.text(text, pos=(x + 0.002, y - 0.002), font_size=size, color=0x050B12)
        self.gui.text(text, pos=(x, y), font_size=size, color=color)

    def _label(self, text: str, x: float, y: float, size: int = 15, color: int = 0xEAF2FF):
        self.gui.text(text, pos=(x, y), font_size=size, color=color)

    def _scene_label_pos(self, point) -> tuple[float, float]:
        scene = self.scene
        if self.view_mode == "map":
            x0, y0, x1, y1 = _map_bounds(self.W, self.W)
            p = np.asarray(point, dtype=np.float32)
            x = x0 + p[0] / max(scene.lx, 1e-6) * (x1 - x0)
            y = y0 + p[2] / max(scene.lz, 1e-6) * (y1 - y0)
            return float(np.clip(x / self.W, 0.01, 0.94)), float(np.clip(y / self.W, 0.06, 0.80))
        lx, ly, lz = scene.lx, scene.ly, scene.lz
        center = np.array([lx * 0.5, ly * 0.5, lz * 0.5])
        diag = np.sqrt(lx * lx + ly * ly + lz * lz)
        scale = 0.74 * self.W / max(diag, 1e-6)
        right, up, _ = _camera_basis(self.azim, self.elev)
        x, y = _project_point(point, center, scale, right, up, self.W, self.W)
        return float(np.clip(x / self.W, 0.01, 0.94)), float(np.clip(y / self.W, 0.06, 0.80))

    def _draw_world_labels(self):
        scene = self.scene
        if not hasattr(scene, "game_status"):
            return

        targets = getattr(scene, "targets", ())
        current = getattr(scene, "current_target", 0)
        for idx, target in enumerate(targets):
            if idx != current:
                continue
            x0, y0, z0, x1, y1, z1 = target.region
            x, y = self._scene_label_pos(((x0 + x1) * 0.5, y1 + 0.045, (z0 + z1) * 0.5))
            self._label(f"NEXT {target.name.upper()}", x, y, size=11, color=0xBFFFCA)

        if self.hud_detail == "full":
            for hazard in getattr(scene, "hazards", ()):
                x0, y0, z0, x1, y1, z1 = hazard.region
                x, y = self._scene_label_pos(((x0 + x1) * 0.5, y1 + 0.035, (z0 + z1) * 0.5))
                self._label(f"DANGER {hazard.name.upper()}", x, y, size=10, color=0xFF9188)


    def _paint_end_state_panel(self, img):
        scene = self.scene
        if not hasattr(scene, "game_status") or getattr(scene, "game_status", "playing") == "playing":
            return
        W, H = img.shape[:2]
        x0, x1 = int(W * 0.30), int(W * 0.70)
        y0, y1 = int(H * 0.45), int(H * 0.60)
        panel = np.array([0.025, 0.045, 0.075], dtype=np.float32)
        border = np.array([0.62, 0.72, 0.86], dtype=np.float32)
        img[x0:x1, y0:y1] = img[x0:x1, y0:y1] * 0.18 + panel * 0.82
        img[x0:x1, y0:y0 + 2] = border
        img[x0:x1, y1 - 2:y1] = border
        img[x0:x0 + 2, y0:y1] = border
        img[x1 - 2:x1, y0:y1] = border

    def _draw_game_ui_labels(self):
        scene = self.scene
        if not hasattr(scene, "game_status"):
            return

        active = getattr(scene, "active_target", None)
        target_name = "DONE" if active is None else active.name.upper()
        progress = getattr(scene, "target_progress", 0.0) * 100.0
        water = getattr(scene, "remaining_water", None)
        water_text = "INF" if water is None else str(water)
        jet_on = getattr(scene, "player_jet_enabled", None)
        jet_state = "ON" if jet_on else "OFF"
        event = getattr(scene, "last_event", "")
        status_raw = getattr(scene, "game_status", "playing")
        status = "PAUSED" if getattr(scene, "_ui_paused", False) and status_raw == "playing" else status_raw.upper()
        stat_x = max(0.52, (max(16, self.W // 36) + max(170, int(self.W * 0.42)) + 8) / self.W)
        stat_size = 10 if self.W < 400 else 12

        self._label(f"{scene.name.upper()}  {status}", 0.025, 0.955, size=15, color=0xFFFFFF)
        self._label(f"WATER {water_text}", stat_x, 0.866, size=stat_size, color=0xBFD9FF)
        self._label(f"GOAL {target_name} {progress:.0f}%", stat_x, 0.836, size=stat_size, color=0xC7FFD2)
        self._label(f"JET {jet_state}", 0.845, 0.852, size=stat_size,
                    color=0xBFD9FF if jet_on else 0xA7AFBA)
        if event and event != "ready":
            self._label(f"EVENT  {event.upper()}", 0.025, 0.720, size=12, color=0xFFD1B8)

        instruction = self._current_instruction(target_name)
        self._label(instruction, 0.025, 0.742, size=12, color=0xFFFFFF)

        if self.hud_detail == "full":
            legend = [("FLUID", 0.050), ("SOFT", 0.140), ("GOAL", 0.230),
                      ("DANGER", 0.320), ("JET", 0.430)]
            for label, x in legend:
                self._label(label, x, 0.026, size=10, color=0xDDE8F7)

    def _current_instruction(self, target_name: str) -> str:
        scene = self.scene
        status = getattr(scene, "game_status", "playing")
        if status == "won":
            return "Goal complete. Press R to replay."
        if status == "lost":
            return "Try again: keep the orange jelly out of red zones."
        if getattr(scene, "_ui_paused", False):
            return "Goal: push ORANGE JELLY into GREEN zones in order. Press Space."
        if scene.name == "soft_plug" and target_name == "ALIGN":
            return "1/2 ALIGN: keep the jelly inside the green box until 100%."
        if scene.name == "soft_plug" and target_name == "SEAL":
            return "2/2 SEAL: steer the yellow jet so the jelly moves right."
        return f"Target {target_name}: hold the jelly in the active green box."

    def _draw_end_state_message(self):
        scene = self.scene
        if not hasattr(scene, "game_status"):
            return
        status = getattr(scene, "game_status", "playing")
        if status == "playing":
            return
        if status == "won":
            self._label("WON", 0.430, 0.545, size=30, color=0x7EF6FF)
            self._label("Press R to play again", 0.350, 0.492, size=14, color=0xEAF2FF)
        elif status == "lost":
            self._label("LOST", 0.405, 0.545, size=30, color=0xFF6B5E)
            self._label("Press R to restart", 0.365, 0.492, size=14, color=0xEAF2FF)

    @property
    def running(self) -> bool:
        return self.gui.running
