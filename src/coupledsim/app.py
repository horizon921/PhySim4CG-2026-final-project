"""交互式入口（三维）。

用法：
    python -m coupledsim.app                      # 3D dam break（默认）
    python -m coupledsim.app --scene jet
    python -m coupledsim.app --transfer flip --res 48
    python -m coupledsim.app --headless 400 --save outputs/dambreak  # 离屏出图

交互：
    鼠标左键拖动 = 旋转相机
    空格 = 暂停/继续   R = 重置   P/F/A = 切换 PIC/FLIP/APIC
    G = 开/关重力      X = 横向晃动一下   H = 帮助   Esc = 退出
"""

import argparse
import time

import taichi as ti

from .config import TransferMode
from .scene import build_scene, BUILDERS

ARCH_MAP = {"cpu": ti.cpu, "gpu": ti.gpu, "metal": ti.metal,
            "vulkan": ti.vulkan, "cuda": ti.cuda}
TRANSFER_MAP = {"pic": TransferMode.PIC, "flip": TransferMode.FLIP, "apic": TransferMode.APIC}
GAME_SCENES = {"soft_plug", "soft_slalom", "soft_rescue"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="coupledsim 三维流体交互演示")
    p.add_argument("--scene", default="dambreak", choices=list(BUILDERS))
    p.add_argument("--transfer", default="apic", choices=list(TRANSFER_MAP))
    p.add_argument("--res", type=int, default=None,
                   help="网格分辨率（每轴）。CPU 上 32≈流畅, 48≈较慢但更精细")
    p.add_argument("--arch", default="cpu", choices=list(ARCH_MAP))
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--autoplay", action="store_true",
                   help="启动后直接运行仿真；默认先暂停以避免窗口首帧未响应")
    p.add_argument("--render-particles", type=int, default=None,
                   help="GUI 中最多绘制多少流体粒子；0 表示全部绘制")
    p.add_argument("--render-scale", type=float, default=None,
                   help="GUI 场景内部渲染比例；低于 1 可提升帧率，文字 HUD 不受影响")
    p.add_argument("--color-by-velocity", action="store_true",
                   help="Color fluid particles by velocity in the GUI; prettier but slower.")
    p.add_argument("--quality", default="fast", choices=("fast", "pretty"),
                   help="GUI quality preset. fast is smoother; pretty draws solid voxels.")
    p.add_argument("--hud", default=None, choices=("compact", "full"),
                   help="HUD detail level. compact is smoother; full shows extra labels.")
    p.add_argument("--view", default=None, choices=("map", "3d"),
                   help="Game view mode. map is clearer and smoother; 3d is better for demos.")
    p.add_argument("--headless", type=int, default=0,
                   help="不开窗口，运行指定帧数后退出（用于出图/计时）")
    p.add_argument("--save", default="", help="headless 模式下保存帧的前缀路径")
    p.add_argument("--save-every", type=int, default=40, help="headless 每隔多少帧存一帧")
    return p.parse_args(argv)


def run_headless(scene, steps, save_prefix, save_every):
    from .render import render_frame, save_png
    t0 = time.time()
    saved = 0
    azim = 35.0
    for f in range(steps):
        scene.step()
        if save_prefix and (f % save_every == 0 or f == steps - 1):
            img = render_frame(scene, width=640, azim=azim + 0.25 * f, elev=22.0)
            save_png(img, f"{save_prefix}_{f:04d}.png")
            saved += 1
    dt = time.time() - t0
    print(f"[headless] scene={scene.name} transfer={scene.cfg.transfer.name} "
          f"steps={steps} N={scene.n_particles} "
          f"time={dt:.2f}s ({steps / max(dt, 1e-9):.1f} steps/s) saved={saved}")


def _hud_lines(scene, nsub: int, fps: float, paused: bool, show_help: bool) -> list[str]:
    if hasattr(scene, "game_status"):
        lines = []
        status = getattr(scene, "game_status", "playing")
        active = getattr(scene, "active_target", None)
        target_name = "DONE" if active is None else active.name.upper()
        jet_on = getattr(scene, "player_jet_enabled", True)
        if status == "playing" and not paused and not jet_on:
            lines.append("Water is OFF. Press O to turn the jet back on.")
        if paused and scene.frame == 0:
            lines.append("First start compiles kernels and may take 1-2 minutes.")
        if show_help:
            lines.append(f"Target {target_name}: move the orange jelly into the green box.")
            lines.append("Controls: Space start/pause | Drag orbit | H hide HUD | Esc quit")
            lines.append("Jet: W/S/A/D aim | Q/E speed | O toggle | U pulse | R reset")
        return lines

    lines = [
        f"{scene.name} | {scene.cfg.transfer.name} | {scene.n_particles} particles "
        f"| sub {nsub} | {fps:4.1f} FPS" + (" | PAUSED" if paused else "")
    ]
    if show_help:
        if hasattr(scene, "steer_player_jet"):
            lines.append("Jet: I/K up-down | J/L depth | Q/E speed | O toggle | U pulse")
        if paused and scene.frame == 0:
            lines.append("Press Space to compile and start. First frame may take 1-2 minutes.")
        lines.append("Space pause/start | R reset | P/F/A transfer | G gravity | Drag orbit | H HUD | Esc quit")
    return lines


def main(argv=None):
    args = parse_args(argv)
    ti.init(arch=ARCH_MAP[args.arch], default_fp=ti.f32)
    is_game = args.scene in GAME_SCENES
    res = args.res if args.res is not None else (8 if is_game else 40)
    window = args.window if args.window is not None else (380 if is_game else 480)
    render_particles = args.render_particles if args.render_particles is not None else (800 if is_game else 1400)
    render_scale = args.render_scale if args.render_scale is not None else (0.55 if is_game else 0.65)
    hud_detail = args.hud if args.hud is not None else ("compact" if is_game and args.quality == "fast" else "full")
    view_mode = args.view if args.view is not None else ("map" if is_game and args.quality == "fast" else "3d")
    scene = build_scene(args.scene, res=res, transfer=TRANSFER_MAP[args.transfer])

    if args.headless > 0:
        run_headless(scene, args.headless, args.save, args.save_every)
        return

    from .render import Viewer3D
    render_cap = None if render_particles <= 0 else render_particles
    viewer = Viewer3D(scene, window_size=window, max_fluid_particles=render_cap,
                      render_scale=render_scale, color_by_velocity=args.color_by_velocity,
                      draw_solid_voxels=(args.quality == "pretty"),
                      hud_detail=hud_detail, view_mode=view_mode)
    gui = viewer.gui

    paused = not args.autoplay
    show_help = not hasattr(scene, "game_status")
    gravity_on = True
    last_cur = None
    nsub = scene.cfg.substeps
    fps_t = time.time()
    fps = 0.0

    while viewer.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key == "r":
                scene.reset()
            elif e.key == "w" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(dy=0.25)
            elif e.key == "s" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(dy=-0.25)
            elif e.key == "d" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(dz=0.25)
            elif e.key == "a" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(dz=-0.25)
            elif e.key == "p" and not hasattr(scene, "game_status"):
                scene.cfg.transfer = TransferMode.PIC
            elif e.key == "f" and not hasattr(scene, "game_status"):
                scene.cfg.transfer = TransferMode.FLIP
            elif e.key == "a" and not hasattr(scene, "game_status"):
                scene.cfg.transfer = TransferMode.APIC
            elif e.key == "g" and not hasattr(scene, "game_status"):
                gravity_on = not gravity_on
                scene.cfg.gravity_y = -9.8 if gravity_on else 0.0
            elif e.key == "x":
                cx, cy, cz = scene.lx * 0.5, scene.ly * 0.5, scene.lz * 0.5
                scene.solver.apply_drag_force(cx, cy, cz, 3.0, 0.0, 0.0, scene.lx, 1.0)
            elif e.key == "i" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(dy=0.25)
            elif e.key == "k" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(dy=-0.25)
            elif e.key == "j" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(dz=-0.25)
            elif e.key == "l" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(dz=0.25)
            elif e.key == "q" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(scale=0.9)
            elif e.key == "e" and hasattr(scene, "steer_player_jet"):
                scene.steer_player_jet(scale=1.1)
            elif e.key == "o" and hasattr(scene, "toggle_player_jet"):
                scene.toggle_player_jet()
            elif e.key == "u" and hasattr(scene, "pulse_player_jet"):
                scene.pulse_player_jet()
            elif e.key == "h":
                show_help = not show_help

        # 鼠标左键拖动旋转相机
        cur = gui.get_cursor_pos()
        if gui.is_pressed(ti.GUI.LMB) and last_cur is not None:
            viewer.rotate((cur[0] - last_cur[0]) * 220.0, -(cur[1] - last_cur[1]) * 160.0)
        last_cur = cur

        if hasattr(scene, "steer_player_jet"):
            aim_step = 0.055
            if gui.is_pressed("w") or gui.is_pressed("i"):
                scene.steer_player_jet(dy=aim_step)
            if gui.is_pressed("s") or gui.is_pressed("k"):
                scene.steer_player_jet(dy=-aim_step)
            if gui.is_pressed("d") or gui.is_pressed("l"):
                scene.steer_player_jet(dz=aim_step)
            if gui.is_pressed("a") or gui.is_pressed("j"):
                scene.steer_player_jet(dz=-aim_step)
            if gui.is_pressed("q"):
                scene.steer_player_jet(scale=0.99)
            if gui.is_pressed("e"):
                scene.steer_player_jet(scale=1.01)

        if not paused:
            nsub = scene.step()

        now = time.time()
        dtf = now - fps_t
        fps_t = now
        if dtf > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dtf)

        setattr(scene, "_ui_paused", paused)
        viewer.draw(_hud_lines(scene, nsub, fps, paused, show_help))


if __name__ == "__main__":
    main()
