import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coupledsim.coupling import FluidSoftCoupler, sample_mac_velocity_np, soft_body_solid_fields
from coupledsim.softbody import KinematicSoftBody, XPBDConfig, XPBDSoftBody, rasterize_soft_bodies


def test_soft_body_solid_fields_have_expected_shapes_and_volume():
    nx = ny = nz = 12
    dx = 1.0 / nx
    body = KinematicSoftBody(
        center=(0.5, 0.5, 0.5),
        radius=0.16,
        velocity=(1.0, 0.5, -0.25),
        surface_samples=64,
    )

    phi, u, v, w = soft_body_solid_fields(nx, ny, nz, dx, [body], with_walls=True)

    assert phi.shape == (nx, ny, nz)
    assert u.shape == (nx + 1, ny, nz)
    assert v.shape == (nx, ny + 1, nz)
    assert w.shape == (nx, ny, nz + 1)
    assert phi.min() < -0.05
    assert np.abs(u).max() > 0.9
    assert np.abs(v).max() > 0.4
    assert np.abs(w).max() > 0.2


def test_sample_mac_velocity_np_matches_constant_fields():
    points = np.array([[0.25, 0.35, 0.45], [0.8, 0.7, 0.6]], dtype=np.float32)
    u = np.full((5, 4, 4), 1.0, dtype=np.float32)
    v = np.full((4, 5, 4), 2.0, dtype=np.float32)
    w = np.full((4, 4, 5), 3.0, dtype=np.float32)

    vel = sample_mac_velocity_np(points, u, v, w, dx=0.25)

    np.testing.assert_allclose(vel, np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32))


def test_coupler_applies_drag_force_to_soft_body():
    class Field:
        def __init__(self, value):
            self.value = value

        def to_numpy(self):
            return self.value.copy()

    class SolverStub:
        dx = 0.25
        u = Field(np.full((5, 4, 4), 1.0, dtype=np.float32))
        v = Field(np.full((4, 5, 4), 2.0, dtype=np.float32))
        w = Field(np.full((4, 4, 5), 3.0, dtype=np.float32))

    body = KinematicSoftBody(center=(0.5, 0.5, 0.5), radius=0.1, surface_samples=32, mass=2.0)
    coupler = FluidSoftCoupler(drag_coeff=6.0, max_point_force=None)

    coupler.apply_fluid_forces(SolverStub(), [body])
    np.testing.assert_allclose(body.last_force, np.array([6.0, 12.0, 18.0], dtype=np.float32), rtol=1e-5)

    before = body.position
    body.step(0.1)
    assert body.position[0] > before[0]
    assert body.position[1] > before[1]
    assert body.position[2] > before[2]


def test_xpbd_softbody_interface_and_rasterization():
    cfg = XPBDConfig(spacing=0.08, radius=0.05, stiffness=0.72, damping=2.0,
                     solver_iters=8, max_speed=4.0)
    body = XPBDSoftBody.make_box(center=(0.5, 0.65, 0.5), dims=(3, 3, 3),
                                 spacing=cfg.spacing, cfg=cfg)
    assert body.surface_points_np().shape == body.positions_np().shape
    assert body.surface_velocities_np().shape == body.velocities_np().shape

    body.apply_forces_np(np.tile(np.array([0.0, 1.0, 0.0], np.float32), (len(body.x), 1)))
    for _ in range(10):
        body.step(1.0 / 120.0, bounds=(1.0, 1.0, 1.0), gravity=(0.0, -9.8, 0.0))
    pos = body.positions_np()
    assert np.isfinite(pos).all()
    assert pos[:, 1].min() >= cfg.radius - 1e-5

    base = np.full((16, 16, 16), 1.0, np.float32)
    phi, u, v, w = rasterize_soft_bodies(base, [body], 1.0 / 16.0)
    assert phi.min() < 0.0
    assert u.shape == (17, 16, 16)
    assert v.shape == (16, 17, 16)
    assert w.shape == (16, 16, 17)


def test_soft_plug_game_rules_render_and_controls():
    import taichi as ti

    ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=0)

    from coupledsim.render import render_frame
    from coupledsim.scene import GameZone, build_scene

    scene = build_scene("soft_plug", res=6)
    assert isinstance(scene.soft_bodies[0], XPBDSoftBody)
    assert len(scene.targets) == 2
    assert len(scene.hazards) == 1
    assert scene.game_status == "playing"
    assert scene.target_progress == 0.0

    v0 = np.asarray(scene.player_jet_velocity, dtype=np.float32)
    scene.steer_player_jet(dy=0.25, dz=0.25, scale=1.1)
    v1 = np.asarray(scene.player_jet_velocity, dtype=np.float32)
    assert not np.allclose(v0, v1)
    assert scene.player_jet_enabled is True
    scene.toggle_player_jet()
    assert scene.player_jet_enabled is False
    scene.toggle_player_jet()
    assert scene.player_jet_enabled is True

    scene.water_budget = 5
    scene.emitted_particles = 3
    assert scene.remaining_water == 2
    assert scene._emitter_count(scene.emitters[0]) == 2
    scene.pulse_cost = 3
    scene.pulse_player_jet()
    assert scene.last_event == "not enough water"
    scene.pulse_cost = 1
    scene.pulse_player_jet()
    assert scene.last_event == "pulse"
    assert scene.pulse_cooldown == scene.pulse_cooldown_frames
    scene.pulse_player_jet()
    assert scene.last_event == "pulse cooling"

    px, py, pz = scene.primary_body_position
    local = (px - 0.1, py - 0.1, pz - 0.1, px + 0.1, py + 0.1, pz + 0.1)
    scene.targets = [
        GameZone("first", local, hold_frames=2, score=10),
        GameZone("second", local, hold_frames=2, score=20),
    ]
    scene.hazards = []
    scene.current_target = 0
    scene._sync_legacy_target()
    assert scene.target_match > 0.0
    assert "match=" in scene.status_line
    for _ in range(2):
        scene._update_game_status()
    assert scene.game_status == "playing"
    assert scene.current_target == 1
    assert scene.score == 10
    for _ in range(2):
        scene._update_game_status()
    assert scene.game_status == "won"
    assert scene.score >= 30
    assert scene.completed_targets == 2

    scene.reset()
    scene.targets = []
    scene.target = (0.0, 0.0, 0.0, 0.01, 0.01, 0.01)
    scene.hazards = [GameZone("bad", local, hold_frames=3)]
    for _ in range(3):
        scene._update_game_status()
    assert scene.game_status == "lost"
    assert scene.last_event == "plug overheated"

    img = render_frame(scene, width=96)
    assert img.shape == (96, 96, 3)
    assert np.isfinite(img).all()
    assert float(img.max()) > 0.9


def test_all_soft_game_levels_build_and_render_low_res():
    import taichi as ti

    ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=2)

    from coupledsim.render import render_frame
    from coupledsim.scene import build_scene

    for name in ("soft_plug", "soft_slalom", "soft_rescue"):
        scene = build_scene(name, res=4)
        assert scene.soft_bodies
        assert scene.targets
        assert scene.hazards
        img = render_frame(scene, width=72)
        assert scene.game_status == "playing"
        assert np.isfinite(scene.soft_bodies[0].positions_np()).all()
        assert np.isfinite(scene.solver.solid_phi.to_numpy()).all()
        assert img.shape == (72, 72, 3)
        assert np.isfinite(img).all()


def test_slow_soft_game_levels_step_low_res():
    import pytest
    import taichi as ti

    if os.environ.get("COUPLEDSIM_RUN_SLOW") != "1":
        pytest.skip("set COUPLEDSIM_RUN_SLOW=1 to run Taichi multi-level step smoke test")

    ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=3)

    from coupledsim.scene import build_scene

    for name in ("soft_plug", "soft_slalom", "soft_rescue"):
        scene = build_scene(name, res=4)
        nsub = scene.step()
        assert nsub >= 1
        assert scene.game_status == "playing"
        assert np.isfinite(scene.soft_bodies[0].positions_np()).all()
        assert np.isfinite(scene.solver.solid_phi.to_numpy()).all()


def test_coupled_scene_handles_missing_game_objects():
    import taichi as ti

    ti.init(arch=ti.cpu, default_fp=ti.f32, random_seed=1)

    from coupledsim.config import FluidConfig
    from coupledsim.scene import CoupledScene, GameZone

    cfg = FluidConfig(res_x=4, res_y=4, res_z=4, domain_x=1.0,
                      adaptive_substeps=False, max_particles=128)
    scene = CoupledScene(name="empty_game", cfg=cfg, max_frames=1)

    assert scene.primary_body_position is None
    assert scene.player_jet_velocity is None
    scene.steer_player_jet(dy=1.0, dz=1.0, scale=2.0)
    scene.toggle_player_jet()
    scene.pulse_player_jet()
    assert scene.last_event == "pulse unavailable"

    scene.frame = 1
    scene._update_game_status()
    assert scene.game_status == "lost"
    assert scene.last_event == "time out"

    body = KinematicSoftBody(center=(1.5, 0.5, 0.5), radius=0.1, surface_samples=8)
    scene = CoupledScene(name="escaped_game", cfg=cfg, soft_bodies=[body],
                         targets=[GameZone("home", (0.0, 0.0, 0.0, 0.2, 0.2, 0.2))])
    scene._update_game_status()
    assert scene.game_status == "lost"
    assert scene.last_event == "plug escaped"
