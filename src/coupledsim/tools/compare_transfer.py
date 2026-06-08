"""PIC / FLIP / APIC 传输方式对比出图。

相同初始条件（默认溃坝）跑到同一帧，三种模式并排，直观展示数值耗散差异：
    PIC  耗散最大，水面最“黏稠”、细节最少
    FLIP 细节最丰富，但偶有噪声/抖动
    APIC 介于两者之间，低耗散且稳定

用法：
    python -m coupledsim.tools.compare_transfer
    python -m coupledsim.tools.compare_transfer --scene dambreak --res 96 --frames 70
输出：outputs/compare_transfer.png（含各模式定量指标：粒子最大速度、平均散度）
"""

import argparse

import numpy as np
import taichi as ti


def main(argv=None):
    p = argparse.ArgumentParser(description="PIC/FLIP/APIC 对比")
    p.add_argument("--scene", default="dambreak")
    p.add_argument("--res", type=int, default=96)
    p.add_argument("--frames", type=int, default=70)
    p.add_argument("--window", type=int, default=460)
    p.add_argument("--arch", default="cpu", choices=["cpu", "gpu", "metal", "cuda"])
    p.add_argument("--out", default="outputs/compare_transfer.png")
    args = p.parse_args(argv)

    arch = {"cpu": ti.cpu, "gpu": ti.gpu, "metal": ti.metal, "cuda": ti.cuda}[args.arch]
    ti.init(arch=arch, default_fp=ti.f32)

    # 延迟导入：必须在 ti.init 之后（构建场景会分配 taichi 字段）
    from ..config import TransferMode
    from ..scene import build_scene
    from ..render import Renderer2D
    from ..fluid.flip_solver import FLUID

    modes = [TransferMode.PIC, TransferMode.FLIP, TransferMode.APIC]
    panels = []
    for mode in modes:
        scene = build_scene(args.scene, res=args.res, transfer=mode)
        renderer = Renderer2D(scene, window_size=args.window, show_gui=False,
                              title=f"compare-{mode.name}")
        for _ in range(args.frames):
            scene.step()
        # 定量指标
        vmax = float(np.linalg.norm(scene.solver.particle_velocities_np(), axis=1).max())
        scene.solver.compute_divergence()
        div = scene.solver.divergence.to_numpy()
        ct = scene.solver.cell_type.to_numpy()
        mean_div = float(np.abs(div[ct == FLUID]).mean()) if (ct == FLUID).any() else 0.0
        path = args.out.replace(".png", f"_{mode.name}.png")
        renderer.draw(info_lines=[f"{mode.name}",
                                  f"frame {args.frames}  N={scene.n_particles}",
                                  f"vmax={vmax:.2f}  mean|div|={mean_div:.1e}"],
                      save_path=path)
        panels.append(ti.tools.imread(path))
        print(f"[{mode.name:4s}] N={scene.n_particles}  vmax={vmax:.3f}  mean|div|={mean_div:.2e}")

    # 横向拼接：imread/imwrite 为 ti 约定 (W, H, 3)，沿 axis=0(宽) 拼接即左右并排
    H = min(pan.shape[1] for pan in panels)
    gap = np.full((6, H, 3), 255, dtype=np.uint8)
    pieces = []
    for k, pan in enumerate(panels):
        pieces.append(pan[:, :H])
        if k < len(panels) - 1:
            pieces.append(gap)
    combined = np.concatenate(pieces, axis=0)
    ti.tools.imwrite(combined, args.out)
    print(f"saved -> {args.out}  (W={combined.shape[0]} H={combined.shape[1]})")


if __name__ == "__main__":
    main()
