"""交互式入口。

用法：
    python -m coupledsim.app                      # dam break（默认）
    python -m coupledsim.app --scene jet
    python -m coupledsim.app --transfer flip --res 128 --arch gpu
    python -m coupledsim.app --headless 400 --save outputs/dambreak  # 离屏出图

交互：
    鼠标左键拖动 = 搅动水（施加拖拽力）
    鼠标右键    = 在光标处加水
    空格 = 暂停/继续    R = 重置    P/F/A = 切换 PIC/FLIP/APIC
    G = 开/关重力       H = 帮助    Esc = 退出
"""

import argparse
import time

import taichi as ti

from .config import TransferMode
from .scene import build_scene, BUILDERS

ARCH_MAP = {"cpu": ti.cpu, "gpu": ti.gpu, "metal": ti.metal,
            "vulkan": ti.vulkan, "cuda": ti.cuda}
TRANSFER_MAP = {"pic": TransferMode.PIC, "flip": TransferMode.FLIP, "apic": TransferMode.APIC}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="coupledsim 二维流体交互演示")
    p.add_argument("--scene", default="dambreak", choices=list(BUILDERS))
    p.add_argument("--transfer", default="apic", choices=list(TRANSFER_MAP))
    p.add_argument("--res", type=int, default=96)
    p.add_argument("--arch", default="cpu", choices=list(ARCH_MAP))
    p.add_argument("--window", type=int, default=720)
    p.add_argument("--headless", type=int, default=0,
                   help="不开窗口，运行指定帧数后退出（用于出图/计时）")
    p.add_argument("--save", default="", help="headless 模式下保存帧的前缀路径")
    p.add_argument("--save-every", type=int, default=40, help="headless 每隔多少帧存一帧")
    return p.parse_args(argv)


def run_headless(scene, steps, save_prefix, save_every):
    from .render.offscreen import render_frame, save_png
    t0 = time.time()
    saved = 0
    for f in range(steps):
        scene.step()
        if save_prefix and (f % save_every == 0 or f == steps - 1):
            img = render_frame(scene, width=640)
            save_png(img, f"{save_prefix}_{f:04d}.png")
            saved += 1
    dt = time.time() - t0
    print(f"[headless] scene={scene.name} transfer={scene.cfg.transfer.name} "
          f"steps={steps} N={scene.n_particles} "
          f"time={dt:.2f}s ({steps / max(dt, 1e-9):.1f} steps/s) saved={saved}")


def main(argv=None):
    args = parse_args(argv)
    ti.init(arch=ARCH_MAP[args.arch], default_fp=ti.f32)
    scene = build_scene(args.scene, res=args.res, transfer=TRANSFER_MAP[args.transfer])

    if args.headless > 0:
        run_headless(scene, args.headless, args.save, args.save_every)
        return

    from .render import Renderer2D
    renderer = Renderer2D(scene, window_size=args.window)
    gui = renderer.gui

    paused = False
    show_help = True
    gravity_on = True
    last_cur = None
    nsub = scene.cfg.substeps
    fps_t = time.time()
    fps = 0.0

    while renderer.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key == "r":
                scene.reset()
            elif e.key == "p":
                scene.cfg.transfer = TransferMode.PIC
            elif e.key == "f":
                scene.cfg.transfer = TransferMode.FLIP
            elif e.key == "a":
                scene.cfg.transfer = TransferMode.APIC
            elif e.key == "g":
                gravity_on = not gravity_on
                scene.cfg.gravity_y = -9.8 if gravity_on else 0.0
            elif e.key == "h":
                show_help = not show_help

        # 鼠标交互
        cur = gui.get_cursor_pos()
        wx, wy = cur[0] * scene.lx, cur[1] * scene.ly
        if gui.is_pressed(ti.GUI.LMB) and last_cur is not None:
            dx = (cur[0] - last_cur[0]) * scene.lx
            dy = (cur[1] - last_cur[1]) * scene.ly
            inv_dt = 1.0 / scene.cfg.dt
            radius = 0.09 * scene.lx
            scene.solver.apply_drag_force(wx, wy, dx * inv_dt, dy * inv_dt, radius, 0.4)
        if gui.is_pressed(ti.GUI.RMB):
            r = 0.035 * scene.lx
            scene.solver.emit_block(wx - r, wy - r, wx + r, wy + r, 0.0, 0.0, 18)
        last_cur = cur

        if not paused:
            nsub = scene.step()

        now = time.time()
        dtf = now - fps_t
        fps_t = now
        if dtf > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dtf)

        info = [f"{scene.name} | {scene.cfg.transfer.name} | N={scene.n_particles} "
                f"| sub={nsub} | {fps:4.0f} fps" + ("  [PAUSED]" if paused else "")]
        if show_help:
            info.append(scene.hint)
            info.append("[Space]暂停 [R]重置 [P/F/A]PIC/FLIP/APIC [G]重力 [H]帮助 [Esc]退出")
        renderer.draw(info)


if __name__ == "__main__":
    main()
