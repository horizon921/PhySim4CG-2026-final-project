"""Preset scenes and playable levels."""

from ..config import FluidConfig, TransferMode
from ..coupling import Box, FluidSoftCoupler, Sphere
from ..softbody import XPBDConfig, XPBDSoftBody
from .coupled_scene import CoupledScene, GameZone
from .scene import Emitter, FluidScene


def _base_cfg(res: int, transfer: TransferMode, **kw) -> FluidConfig:
    cfg = FluidConfig(res_x=res, res_y=res, res_z=res, domain_x=1.0, transfer=transfer)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _game_cfg(res: int, transfer: TransferMode, **kw) -> FluidConfig:
    defaults = dict(
        particles_per_cell=4,
        substeps=1,
        cg_max_iters=45,
        cg_tol=5e-4,
        extrapolate_iters=1,
    )
    defaults.update(kw)
    return _base_cfg(res, transfer, **defaults)


def build_dambreak(res=48, transfer=TransferMode.APIC) -> FluidScene:
    cfg = _base_cfg(res, transfer)
    return FluidScene(
        name="dambreak",
        cfg=cfg,
        init_blocks=[(0.04, 0.04, 0.04, 0.40, 0.92, 0.55)],
        hint="3D dam break. Drag camera / right click inject / R reset / Space pause.",
    )


def build_tank(res=48, transfer=TransferMode.APIC) -> FluidScene:
    cfg = _base_cfg(res, transfer)
    return FluidScene(
        name="tank",
        cfg=cfg,
        init_blocks=[(0.03, 0.03, 0.03, 0.97, 0.42, 0.97)],
        hint="Tank. Drag camera / right click inject / R reset.",
    )


def build_obstacle(res=48, transfer=TransferMode.APIC) -> FluidScene:
    cfg = _base_cfg(res, transfer)
    shapes = [
        Box(cx=0.5, cy=0.32, cz=0.5, hx=0.14, hy=0.07, hz=0.14),
        Sphere(cx=0.28, cy=0.18, cz=0.5, r=0.09),
    ]
    return FluidScene(
        name="obstacle",
        cfg=cfg,
        shapes=shapes,
        init_blocks=[(0.05, 0.55, 0.05, 0.95, 0.95, 0.95)],
        hint="Obstacle flow. Drag camera / R reset.",
    )


def build_jet(res=48, transfer=TransferMode.APIC) -> FluidScene:
    cfg = _base_cfg(res, transfer, max_particles=1_500_000)
    dx = cfg.dx
    return FluidScene(
        name="jet",
        cfg=cfg,
        init_blocks=[(0.03, 0.03, 0.03, 0.97, 0.12, 0.97)],
        emitters=[Emitter(region=(2 * dx, 0.55, 0.42, 5 * dx, 0.68, 0.58),
                          velocity=(4.5, 0.0, 0.0), count=40)],
        hint="Side jet. Drag camera / right click inject / R reset.",
    )


def build_softbody(res=40, transfer=TransferMode.APIC) -> FluidScene:
    cfg = _base_cfg(res, transfer, viscosity=0.015, vel_damping=0.08,
                    particles_per_cell=6, max_particles=80_000)
    dx = cfg.dx
    soft_cfg = XPBDConfig(
        spacing=0.075, radius=0.052, density=0.42, stiffness=0.68,
        damping=3.2, drag=18.0, lift=0.8, fluid_coupling=1.0,
        solver_iters=10, max_speed=3.5,
    )
    jelly = XPBDSoftBody.make_box(center=(0.55, 0.48, 0.5), dims=(3, 4, 3),
                                  spacing=soft_cfg.spacing, cfg=soft_cfg)
    return FluidScene(
        name="softbody",
        cfg=cfg,
        shapes=[Box(cx=0.5, cy=0.19, cz=0.5, hx=0.13, hy=0.035, hz=0.20)],
        soft_bodies=[jelly],
        init_blocks=[(0.03, 0.03, 0.03, 0.42, 0.38, 0.97)],
        emitters=[Emitter(region=(2 * dx, 0.36, 0.35, 5 * dx, 0.50, 0.65),
                          velocity=(3.2, 0.25, 0.0), count=22)],
        hint="XPBD softbody demo: water pushes the jelly and the jelly blocks water.",
    )


def build_soft_plug(res=40, transfer=TransferMode.APIC) -> CoupledScene:
    cfg = _game_cfg(res, transfer, max_particles=20_000, vel_damping=0.035,
                    gravity_y=-1.5)
    dx = cfg.dx
    soft_cfg = XPBDConfig(
        spacing=0.072, radius=0.050, density=0.24, stiffness=0.70,
        damping=3.4, drag=19.0, lift=2.8, solver_iters=8, max_speed=3.8,
    )
    body = XPBDSoftBody.make_box(center=(0.50, 0.56, 0.50), dims=(3, 4, 3),
                                 spacing=soft_cfg.spacing, cfg=soft_cfg)
    return CoupledScene(
        name="soft_plug",
        cfg=cfg,
        soft_bodies=[body],
        shapes=[
            Box(cx=0.56, cy=0.22, cz=0.23, hx=0.08, hy=0.06, hz=0.11),
            Box(cx=0.56, cy=0.22, cz=0.77, hx=0.08, hy=0.06, hz=0.11),
        ],
        init_blocks=[(0.03, 0.03, 0.05, 0.34, 0.58, 0.95)],
        emitters=[Emitter(region=(2 * dx, 0.42, 0.42, 5 * dx, 0.58, 0.58),
                          velocity=(3.0, 1.2, 0.0), count=20)],
        coupler=FluidSoftCoupler(drag_coeff=10.0, max_point_force=3.0),
        targets=[
            GameZone("align", (0.48, 0.20, 0.32, 0.68, 0.72, 0.68), hold_frames=24, score=150),
            GameZone("seal", (0.64, 0.06, 0.28, 0.98, 0.68, 0.72), hold_frames=28, score=350),
        ],
        hazards=[GameZone("drain", (0.44, 0.02, 0.34, 0.78, 0.07, 0.66), hold_frames=600)],
        max_frames=4800,
        water_budget=24000,
        hint="soft_plug: push the XPBD jelly through checkpoints and avoid the red drain.",
        objective="Hold ALIGN to 100%, then push to SEAL.",
        tutorial=(
            "Green boxes are goals. Red box is danger.",
            "Use O to pause water if the jelly overshoots.",
            "This intro level is forgiving; learn the controls first.",
        ),
        jet_assist=0.080,
    )


def build_soft_slalom(res=40, transfer=TransferMode.APIC) -> CoupledScene:
    cfg = _game_cfg(res, transfer, max_particles=32_000, vel_damping=0.025, viscosity=0.01)
    dx = cfg.dx
    soft_cfg = XPBDConfig(
        spacing=0.065, radius=0.046, density=0.38, stiffness=0.62,
        damping=3.0, drag=18.0, lift=0.7, solver_iters=9, max_speed=3.5,
    )
    body = XPBDSoftBody.make_box(center=(0.34, 0.50, 0.34), dims=(3, 3, 3),
                                 spacing=soft_cfg.spacing, cfg=soft_cfg)
    return CoupledScene(
        name="soft_slalom",
        cfg=cfg,
        soft_bodies=[body],
        shapes=[
            Box(cx=0.48, cy=0.22, cz=0.52, hx=0.06, hy=0.12, hz=0.18),
            Sphere(cx=0.66, cy=0.34, cz=0.34, r=0.08),
            Sphere(cx=0.66, cy=0.34, cz=0.72, r=0.08),
        ],
        init_blocks=[(0.03, 0.03, 0.03, 0.30, 0.46, 0.92)],
        emitters=[
            Emitter(region=(2 * dx, 0.34, 0.24, 5 * dx, 0.48, 0.42),
                    velocity=(3.4, 0.15, 0.8), count=16),
            Emitter(region=(2 * dx, 0.44, 0.70, 5 * dx, 0.56, 0.88),
                    velocity=(2.4, -0.10, -1.0), count=8),
        ],
        coupler=FluidSoftCoupler(drag_coeff=11.0, max_point_force=3.0),
        targets=[
            GameZone("left gate", (0.42, 0.30, 0.20, 0.58, 0.62, 0.46), hold_frames=26, score=130),
            GameZone("right gate", (0.60, 0.28, 0.60, 0.78, 0.60, 0.86), hold_frames=32, score=180),
            GameZone("finish", (0.78, 0.28, 0.36, 0.96, 0.60, 0.64), hold_frames=46, score=320),
        ],
        hazards=[GameZone("side drain", (0.42, 0.02, 0.46, 0.78, 0.20, 0.58), hold_frames=45)],
        max_frames=2200,
        water_budget=16000,
        pulse_cost=500,
        hint="soft_slalom: weave the XPBD jelly through three checkpoints around obstacles.",
        objective="Guide the jelly through LEFT GATE, RIGHT GATE, then FINISH.",
        tutorial=(
            "Use J/L to steer around obstacles.",
            "Use U pulse only when the jelly is stuck.",
        ),
    )


def build_soft_rescue(res=40, transfer=TransferMode.APIC) -> CoupledScene:
    cfg = _game_cfg(res, transfer, max_particles=32_000, vel_damping=0.035, viscosity=0.02)
    dx = cfg.dx
    soft_cfg = XPBDConfig(
        spacing=0.070, radius=0.050, density=0.58, stiffness=0.78,
        damping=3.4, drag=20.0, lift=1.0, solver_iters=10, max_speed=3.0,
    )
    body = XPBDSoftBody.make_box(center=(0.40, 0.30, 0.50), dims=(3, 4, 3),
                                 spacing=soft_cfg.spacing, cfg=soft_cfg)
    return CoupledScene(
        name="soft_rescue",
        cfg=cfg,
        soft_bodies=[body],
        shapes=[
            Box(cx=0.54, cy=0.17, cz=0.50, hx=0.10, hy=0.04, hz=0.24),
            Box(cx=0.73, cy=0.38, cz=0.26, hx=0.05, hy=0.16, hz=0.10),
            Box(cx=0.73, cy=0.38, cz=0.74, hx=0.05, hy=0.16, hz=0.10),
        ],
        init_blocks=[(0.03, 0.03, 0.03, 0.34, 0.36, 0.97)],
        emitters=[Emitter(region=(2 * dx, 0.26, 0.42, 5 * dx, 0.42, 0.58),
                          velocity=(3.0, 1.0, 0.0), count=14)],
        coupler=FluidSoftCoupler(drag_coeff=12.0, max_point_force=3.2),
        targets=[
            GameZone("lift", (0.38, 0.44, 0.36, 0.58, 0.70, 0.64), hold_frames=34, score=180),
            GameZone("top seal", (0.76, 0.48, 0.38, 0.96, 0.76, 0.62), hold_frames=54, score=420),
        ],
        hazards=[
            GameZone("bottom drain", (0.30, 0.02, 0.28, 0.76, 0.22, 0.72), hold_frames=38),
            GameZone("overflow", (0.02, 0.78, 0.02, 0.98, 0.98, 0.98), hold_frames=80),
        ],
        max_frames=2400,
        water_budget=13500,
        pulse_cost=650,
        pulse_cooldown_frames=60,
        hint="soft_rescue: conserve water, lift the heavy XPBD jelly, then seal the upper outlet.",
        objective="Lift the heavy jelly into LIFT, then seal TOP SEAL.",
        tutorial=(
            "Keep some upward jet velocity to lift the heavy jelly.",
            "Save water and avoid both bottom drain and overflow.",
        ),
    )


BUILDERS = {
    "dambreak": build_dambreak,
    "tank": build_tank,
    "obstacle": build_obstacle,
    "jet": build_jet,
    "softbody": build_softbody,
    "soft_plug": build_soft_plug,
    "soft_slalom": build_soft_slalom,
    "soft_rescue": build_soft_rescue,
}


def build_scene(name: str, res=48, transfer=TransferMode.APIC):
    if name not in BUILDERS:
        raise ValueError(f"Unknown scene '{name}', choices: {list(BUILDERS)}")
    return BUILDERS[name](res=res, transfer=transfer)
