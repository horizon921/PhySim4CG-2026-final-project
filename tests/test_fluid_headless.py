"""三维流体求解器的 headless 正确性 / 稳定性检查。

可直接运行：  uv run python tests/test_fluid_headless.py
也可用 pytest： uv run pytest
检查项：
    - 压力投影后散度显著下降（近似不可压缩）
    - 多帧推进无 NaN / Inf，粒子守界，粒子数守恒
    - dam break 水柱坍塌后质心下降、水平铺开（合乎重力直觉）
    - 障碍（盒+球）内部不残留粒子
    - 粘度单调降低动能且不失稳
    - 三维离屏 / 交互渲染路径可用
"""

import os
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=0)

from coupledsim.config import FluidConfig, TransferMode          # noqa: E402
from coupledsim.coupling import Box, Sphere, static_solid_phi    # noqa: E402
from coupledsim.fluid import FlipSolver, FLUID                   # noqa: E402


def make_solver(res=20, transfer=TransferMode.APIC, ppc=8, **kw):
    cfg = FluidConfig(res_x=res, res_y=res, res_z=res, domain_x=1.0, particles_per_cell=ppc,
                      transfer=transfer, substeps=2, adaptive_substeps=False,
                      max_particles=400_000, **kw)
    solver = FlipSolver(cfg)
    phi = static_solid_phi(res, res, res, cfg.dx, shapes=[], with_walls=True)
    solver.set_solid_phi(phi)
    return cfg, solver


def test_projection_reduces_divergence():
    cfg, solver = make_solver(res=20)
    solver.add_particle_block(0.05, 0.1, 0.05, 0.45, 0.9, 0.95)
    assert solver.n_particles[None] > 0
    for _ in range(15):
        solver.step()

    sdt = cfg.dt / cfg.substeps
    solver.p2g(1)
    solver.normalize_grid()
    solver.classify_cells()
    solver.add_gravity(sdt, cfg.gravity_x, cfg.gravity_y, cfg.gravity_z)
    solver.apply_solid_boundaries()
    solver.compute_divergence()
    div_before = solver.compute_max_divergence()
    solver.solve_pressure(sdt)
    solver.apply_pressure(sdt)
    solver.compute_divergence()
    div_after = solver.compute_max_divergence()
    print(f"[projection] div_before={div_before:.3f}  div_after={div_after:.6f}  "
          f"ratio={div_after / max(div_before, 1e-9):.3e}")
    assert div_before > 1e-2, "测试前提：投影前应存在明显散度"
    assert div_after < div_before * 0.05, "压力投影未能显著降低散度"


def run_stability(transfer, res=22, frames=60):
    cfg, solver = make_solver(res=res, transfer=transfer)
    solver.add_particle_block(0.04, 0.04, 0.04, 0.42, 0.9, 0.55)
    n0 = solver.n_particles[None]
    pos0 = solver.particle_positions_np()
    com_y0 = pos0[:, 1].mean()

    lx, ly, lz = cfg.res_x * cfg.dx, cfg.res_y * cfg.dx, cfg.res_z * cfg.dx
    for f in range(frames):
        solver.step()
        pos = solver.particle_positions_np()
        vel = solver.particle_velocities_np()
        assert np.isfinite(pos).all() and np.isfinite(vel).all(), f"frame {f}: NaN/Inf"
        assert pos[:, 0].min() >= -1e-3 and pos[:, 0].max() <= lx + 1e-3
        assert pos[:, 1].min() >= -1e-3 and pos[:, 1].max() <= ly + 1e-3
        assert pos[:, 2].min() >= -1e-3 and pos[:, 2].max() <= lz + 1e-3

    assert solver.n_particles[None] == n0, "无发射/吸入时粒子数应守恒"
    pos = solver.particle_positions_np()
    com_y1 = pos[:, 1].mean()
    x_spread = pos[:, 0].max() - pos[:, 0].min()
    print(f"[stability {transfer.name}] n={n0}  com_y {com_y0:.3f}->{com_y1:.3f}  "
          f"x_spread={x_spread:.3f}  vmax={np.linalg.norm(vel, axis=1).max():.3f}")
    assert com_y1 < com_y0 - 0.03, "质心未下降，疑似不稳定"
    assert x_spread > 0.55, "水未水平铺开，dam break 行为异常"


def test_stability_apic():
    run_stability(TransferMode.APIC)


def test_stability_flip():
    run_stability(TransferMode.FLIP)


def test_stability_pic():
    run_stability(TransferMode.PIC)


def test_obstacle_blocks_fluid():
    res = 24
    cfg = FluidConfig(res_x=res, res_y=res, res_z=res, domain_x=1.0, transfer=TransferMode.APIC,
                      adaptive_substeps=False, substeps=2, max_particles=400_000)
    solver = FlipSolver(cfg)
    box = Box(cx=0.5, cy=0.34, cz=0.5, hx=0.14, hy=0.07, hz=0.14)
    sphere = Sphere(cx=0.28, cy=0.2, cz=0.5, r=0.09)
    phi = static_solid_phi(res, res, res, cfg.dx, shapes=[box, sphere], with_walls=True)
    solver.set_solid_phi(phi)
    solver.add_particle_block(0.05, 0.55, 0.05, 0.95, 0.95, 0.95)
    for _ in range(90):
        solver.step()
    pos = solver.particle_positions_np()
    m = 0.5 * cfg.dx
    in_box = ((np.abs(pos[:, 0] - box.cx) < box.hx - m)
              & (np.abs(pos[:, 1] - box.cy) < box.hy - m)
              & (np.abs(pos[:, 2] - box.cz) < box.hz - m))
    in_sph = ((pos - np.array([sphere.cx, sphere.cy, sphere.cz])) ** 2).sum(1) < (sphere.r - m) ** 2
    n_in = int(in_box.sum() + in_sph.sum())
    print(f"[obstacle] inside_box={int(in_box.sum())} inside_sphere={int(in_sph.sum())} / {len(pos)}")
    assert n_in <= 0.01 * len(pos), "过多粒子穿入障碍内部"


def test_viscosity_dissipates_energy():
    def run(nu):
        cfg, solver = make_solver(res=20, transfer=TransferMode.APIC, viscosity=nu)
        solver.add_particle_block(0.05, 0.1, 0.05, 0.45, 0.9, 0.95)
        ke_mid = 0.0
        for f in range(55):
            solver.step()
            v = solver.particle_velocities_np()
            assert np.isfinite(v).all(), f"nu={nu} frame {f} NaN"
            if f == 28:
                ke_mid = 0.5 * float((v ** 2).sum())
        return ke_mid

    ke0, ke1, ke2 = run(0.0), run(0.03), run(0.12)
    print(f"[viscosity] KE@28  nu0={ke0:.1f}  nu.03={ke1:.1f}  nu.12={ke2:.1f}")
    assert ke1 < ke0 and ke2 < ke1, "粘度未单调降低动能"


def test_renderer_headless_draws():
    """三维离屏 + 交互渲染路径不应报错并能出图。"""
    from coupledsim.scene import build_scene
    from coupledsim.render import Viewer3D, render_frame
    scene = build_scene("obstacle", res=20, transfer=TransferMode.APIC)
    viewer = Viewer3D(scene, window_size=320, show_gui=False)
    for _ in range(15):
        scene.step()
    viewer.draw(info_lines=[f"N={scene.n_particles}", "headless 3D render test"])
    img = render_frame(scene, width=200, azim=40.0, elev=20.0)
    assert img.shape == (200, 200, 3)
    assert np.isfinite(img).all()
    print(f"[render] Viewer3D + offscreen 正常，img={img.shape}")


if __name__ == "__main__":
    print("=== projection ===");        test_projection_reduces_divergence()
    print("=== stability APIC ===");     test_stability_apic()
    print("=== stability FLIP ===");     test_stability_flip()
    print("=== stability PIC ===");      test_stability_pic()
    print("=== obstacle ===");           test_obstacle_blocks_fluid()
    print("=== viscosity ===");          test_viscosity_dissipates_energy()
    print("=== renderer ===");           test_renderer_headless_draws()
    print("\nALL 3D HEADLESS CHECKS PASSED ✅")
