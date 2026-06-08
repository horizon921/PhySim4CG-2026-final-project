"""流体求解器的 headless 正确性 / 稳定性检查。

可直接运行：  uv run python tests/test_fluid_headless.py
也可用 pytest： uv run pytest
检查项：
    - 压力投影后散度显著下降（近似不可压缩）
    - 多帧推进无 NaN / Inf
    - 粒子始终在域内
    - 粒子数（无发射/吸入时）守恒
    - dam break 水柱坍塌后水面下降、整体下落（合乎重力直觉）
"""

import os
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=0)

from coupledsim.config import FluidConfig, TransferMode          # noqa: E402
from coupledsim.coupling import Box, static_solid_phi            # noqa: E402
from coupledsim.fluid import FlipSolver, FLUID                   # noqa: E402


def make_solver(res=32, transfer=TransferMode.APIC, ppc=4):
    cfg = FluidConfig(res_x=res, res_y=res, domain_x=1.0, particles_per_cell=ppc,
                      transfer=transfer, pressure_iters=60, substeps=2,
                      adaptive_substeps=False, max_particles=80_000)
    solver = FlipSolver(cfg)
    phi = static_solid_phi(res, res, cfg.dx, shapes=[], with_walls=True)
    solver.set_solid_phi(phi)
    return cfg, solver


def test_projection_reduces_divergence():
    cfg, solver = make_solver(res=32)
    # 一根偏置水柱，先跑若干帧让流动充分发展，产生真实散度
    solver.add_particle_block(0.05, 0.1, 0.45, 0.9)
    assert solver.n_particles[None] > 0
    for _ in range(25):
        solver.step()

    # 手动执行单步投影前后，测量散度下降
    sdt = cfg.dt / cfg.substeps
    solver.p2g(1)
    solver.normalize_grid()
    solver.classify_cells()
    solver.add_gravity(sdt, cfg.gravity_x, cfg.gravity_y)
    solver.apply_solid_boundaries()
    solver.compute_divergence()
    div_before = solver.compute_max_divergence()
    solver.solve_pressure(sdt)
    solver.apply_pressure(sdt)
    solver.compute_divergence()
    div_after = solver.compute_max_divergence()
    print(f"[projection] div_before={div_before:.4f}  div_after={div_after:.6f}  "
          f"ratio={div_after / max(div_before, 1e-9):.4e}")
    assert div_before > 1e-2, "测试前提：投影前应存在明显散度"
    assert div_after < div_before * 0.1, "压力投影未能显著降低散度"


def run_stability(transfer, res=40, frames=120):
    cfg, solver = make_solver(res=res, transfer=transfer)
    # dam break: 左侧一根水柱
    solver.add_particle_block(0.05, 0.05, 0.4, 0.9)
    n0 = solver.n_particles[None]
    y_top0 = solver.particle_positions_np()[:, 1].max()

    lx = cfg.res_x * cfg.dx
    ly = cfg.res_y * cfg.dx
    for f in range(frames):
        solver.step()
        pos = solver.particle_positions_np()
        assert np.isfinite(pos).all(), f"frame {f}: 粒子位置出现 NaN/Inf"
        vel = solver.particle_velocities_np()
        assert np.isfinite(vel).all(), f"frame {f}: 粒子速度出现 NaN/Inf"
        assert pos[:, 0].min() >= -1e-3 and pos[:, 0].max() <= lx + 1e-3, \
            f"frame {f}: 粒子越过 x 边界"
        assert pos[:, 1].min() >= -1e-3 and pos[:, 1].max() <= ly + 1e-3, \
            f"frame {f}: 粒子越过 y 边界"

    n1 = solver.n_particles[None]
    assert n1 == n0, f"无发射/吸入时粒子数应守恒：{n0} -> {n1}"
    pos = solver.particle_positions_np()
    y_top1 = pos[:, 1].max()
    x_spread1 = pos[:, 0].max() - pos[:, 0].min()
    print(f"[stability {transfer.name}] n={n1}  y_top {y_top0:.3f}->{y_top1:.3f}  "
          f"x_spread_end={x_spread1:.3f}  vmax={np.linalg.norm(vel, axis=1).max():.3f}")
    # dam break：水柱坍塌后顶部应下降，且向右铺开
    assert y_top1 < y_top0 + 1e-2, "水面未下降，疑似不稳定上冲"
    assert x_spread1 > 0.5, "水未向右铺开，dam break 行为异常"
    return solver


def test_stability_apic():
    run_stability(TransferMode.APIC)


def test_stability_flip():
    run_stability(TransferMode.FLIP)


def test_stability_pic():
    run_stability(TransferMode.PIC)


def test_obstacle_blocks_fluid():
    """放一个障碍盒，水块落下后不应有粒子停留在障碍内部。"""
    res = 40
    cfg = FluidConfig(res_x=res, res_y=res, domain_x=1.0, transfer=TransferMode.APIC,
                      adaptive_substeps=False, pressure_iters=60, max_particles=80_000)
    solver = FlipSolver(cfg)
    box = Box(cx=0.5, cy=0.35, hx=0.12, hy=0.12)
    phi = static_solid_phi(res, res, cfg.dx, shapes=[box], with_walls=True)
    solver.set_solid_phi(phi)
    solver.add_particle_block(0.05, 0.55, 0.95, 0.9)
    for _ in range(150):
        solver.step()
    pos = solver.particle_positions_np()
    # 检查没有粒子落在障碍内部（留 0.5 单元容差）
    inside = (np.abs(pos[:, 0] - box.cx) < box.hx - 0.5 * cfg.dx) & \
             (np.abs(pos[:, 1] - box.cy) < box.hy - 0.5 * cfg.dx)
    n_inside = int(inside.sum())
    print(f"[obstacle] particles_inside_box={n_inside} / {len(pos)}")
    assert n_inside <= 0.01 * len(pos), "过多粒子穿入障碍内部"


def test_renderer_headless_draws():
    """实际交互渲染器（ti.GUI, show_gui=False）的绘制路径不应报错并能出图。"""
    from coupledsim.scene import build_scene
    from coupledsim.render import Renderer2D
    scene = build_scene("obstacle", res=48, transfer=TransferMode.APIC)
    r = Renderer2D(scene, window_size=400, show_gui=False)
    for _ in range(20):
        scene.step()
    r.draw(info_lines=[f"N={scene.n_particles}", "headless render test"])
    # 离屏渲染器同样应可用
    from coupledsim.render.offscreen import render_frame
    img = render_frame(scene, width=256)
    assert img.shape[0] == 256 and img.ndim == 3
    assert np.isfinite(img).all()
    print(f"[render] Renderer2D + offscreen 均正常，img={img.shape}")


if __name__ == "__main__":
    print("=== projection ===")
    test_projection_reduces_divergence()
    print("=== stability APIC ===")
    test_stability_apic()
    print("=== stability FLIP ===")
    test_stability_flip()
    print("=== stability PIC ===")
    test_stability_pic()
    print("=== obstacle ===")
    test_obstacle_blocks_fluid()
    print("=== renderer ===")
    test_renderer_headless_draws()
    print("\nALL HEADLESS CHECKS PASSED ✅")
