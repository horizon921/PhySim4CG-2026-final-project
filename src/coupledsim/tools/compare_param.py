"""任意流体参数的对比出图（相同初值、同一帧、横向并排）。

可对比 proposal 中提到的参数：流体粘性 viscosity、FLIP 混合比例 flip_ratio、
压力精度 cg_tol、分辨率等（凡 FluidConfig 中的字段均可）。

用法：
    # 粘性对比（无粘 / 低粘 / 高粘）
    python -m coupledsim.tools.compare_param --param viscosity --values 0,0.03,0.12
    # FLIP ratio 对比（需用 FLIP 模式）
    python -m coupledsim.tools.compare_param --param flip_ratio --values 0.0,0.95,0.99 --transfer flip
输出：outputs/compare_<param>.png
"""

import argparse

import numpy as np
import taichi as ti


def main(argv=None):
    p = argparse.ArgumentParser(description="流体参数对比")
    p.add_argument("--param", required=True, help="FluidConfig 字段名，如 viscosity / flip_ratio")
    p.add_argument("--values", required=True, help="逗号分隔的取值，如 0,0.03,0.12")
    p.add_argument("--scene", default="dambreak")
    p.add_argument("--transfer", default="apic", choices=["pic", "flip", "apic"])
    p.add_argument("--res", type=int, default=96)
    p.add_argument("--frames", type=int, default=70)
    p.add_argument("--window", type=int, default=460)
    p.add_argument("--arch", default="cpu", choices=["cpu", "gpu", "metal", "cuda"])
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    arch = {"cpu": ti.cpu, "gpu": ti.gpu, "metal": ti.metal, "cuda": ti.cuda}[args.arch]
    ti.init(arch=arch, default_fp=ti.f32)

    from ..config import TransferMode
    from ..scene import build_scene
    from ..render import Renderer2D
    from ..fluid.flip_solver import FLUID

    transfer = {"pic": TransferMode.PIC, "flip": TransferMode.FLIP,
                "apic": TransferMode.APIC}[args.transfer]
    values = [float(x) for x in args.values.split(",")]
    out = args.out or f"outputs/compare_{args.param}.png"

    panels = []
    for val in values:
        scene = build_scene(args.scene, res=args.res, transfer=transfer)
        if not hasattr(scene.cfg, args.param):
            raise ValueError(f"FluidConfig 无字段 '{args.param}'")
        setattr(scene.cfg, args.param, val)
        renderer = Renderer2D(scene, window_size=args.window, show_gui=False,
                              title=f"{args.param}={val}")
        for _ in range(args.frames):
            scene.step()
        vmax = float(np.linalg.norm(scene.solver.particle_velocities_np(), axis=1).max())
        scene.solver.compute_divergence()
        div = scene.solver.divergence.to_numpy()
        ct = scene.solver.cell_type.to_numpy()
        mean_div = float(np.abs(div[ct == FLUID]).mean()) if (ct == FLUID).any() else 0.0
        path = out.replace(".png", f"_{val}.png")
        renderer.draw(info_lines=[f"{args.param} = {val}",
                                  f"{transfer.name}  frame {args.frames}  N={scene.n_particles}",
                                  f"vmax={vmax:.2f}  mean|div|={mean_div:.1e}"],
                      save_path=path)
        panels.append(ti.tools.imread(path))
        print(f"[{args.param}={val}] N={scene.n_particles} vmax={vmax:.3f} mean|div|={mean_div:.2e}")

    H = min(pan.shape[1] for pan in panels)
    gap = np.full((6, H, 3), 255, dtype=np.uint8)
    pieces = []
    for k, pan in enumerate(panels):
        pieces.append(pan[:, :H])
        if k < len(panels) - 1:
            pieces.append(gap)
    combined = np.concatenate(pieces, axis=0)
    ti.tools.imwrite(combined, out)
    print(f"saved -> {out}  (W={combined.shape[0]} H={combined.shape[1]})")


if __name__ == "__main__":
    main()
