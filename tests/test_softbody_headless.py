"""XPBD soft body and fluid coupling headless checks."""

import os
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=1)

from coupledsim.config import TransferMode  # noqa: E402
from coupledsim.scene import build_scene    # noqa: E402
from coupledsim.softbody import XPBDConfig, XPBDSoftBody, rasterize_soft_bodies  # noqa: E402


def test_xpbd_softbody_stays_finite_and_constrained():
    cfg = XPBDConfig(spacing=0.08, radius=0.045, stiffness=0.72, damping=2.0,
                     solver_iters=8, max_speed=4.0)
    body = XPBDSoftBody.make_box(center=(0.45, 0.72, 0.5), dims=(3, 3, 3),
                                 spacing=cfg.spacing, cfg=cfg)
    rest = np.array([r for _a, _b, r in body.constraints])
    for _ in range(50):
        body.step(1.0 / 120.0, bounds=(1.0, 1.0, 1.0), gravity=(0.0, -9.8, 0.0))
    pos = body.positions_np()
    assert np.isfinite(pos).all()
    assert pos[:, 1].min() >= cfg.radius - 1e-5
    lengths = np.array([np.linalg.norm(pos[a] - pos[b]) for a, b, _r in body.constraints])
    rel_err = np.abs(lengths - rest) / np.maximum(rest, 1e-8)
    assert float(np.percentile(rel_err, 90)) < 0.25


def test_softbody_rasterizes_dynamic_solid():
    cfg = XPBDConfig(spacing=0.08, radius=0.05)
    body = XPBDSoftBody.make_box(center=(0.5, 0.5, 0.5), dims=(2, 2, 2),
                                 spacing=cfg.spacing, cfg=cfg)
    base = np.full((18, 18, 18), 1.0, np.float32)
    phi, u, v, w = rasterize_soft_bodies(base, [body], 1.0 / 18.0)
    assert phi.min() < 0.0
    assert u.shape == (19, 18, 18)
    assert v.shape == (18, 19, 18)
    assert w.shape == (18, 18, 19)


def test_softbody_scene_coupling_runs_headless():
    scene = build_scene("softbody", res=18, transfer=TransferMode.APIC)
    assert scene.soft_bodies
    y0 = float(scene.soft_bodies[0].center[1])
    for _ in range(18):
        scene.step()
    pos = scene.soft_bodies[0].positions_np()
    vel = scene.soft_bodies[0].velocities_np()
    fluid = scene.solver.particle_positions_np()
    assert np.isfinite(pos).all() and np.isfinite(vel).all()
    assert np.isfinite(fluid).all()
    assert scene.solver.solid_phi.to_numpy().min() < 0.0
    assert abs(float(scene.soft_bodies[0].center[1]) - y0) > 1e-4
