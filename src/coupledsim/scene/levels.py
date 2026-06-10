"""预定义关卡 / 演示场景（三维）。

每个 build_* 返回一个 FluidScene。CLI（app.py）通过名字选择。
注意：必须在 ti.init() 之后调用（构建时会分配 taichi 字段）。
区域为 (x0,y0,z0,x1,y1,z1)，世界坐标，默认域为单位立方体 [0,1]^3。
"""

from ..config import FluidConfig, TransferMode
from ..coupling import Box, Sphere
from ..softbody import XPBDConfig, XPBDSoftBody
from .scene import Emitter, FluidScene


def _base_cfg(res: int, transfer: TransferMode, **kw) -> FluidConfig:
    cfg = FluidConfig(res_x=res, res_y=res, res_z=res, domain_x=1.0, transfer=transfer)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def build_dambreak(res=48, transfer=TransferMode.APIC) -> FluidScene:
    """三维溃坝：角落一根水柱在重力下坍塌、铺开、拍打远壁。"""
    cfg = _base_cfg(res, transfer)
    return FluidScene(
        name="dambreak",
        cfg=cfg,
        init_blocks=[(0.04, 0.04, 0.04, 0.40, 0.92, 0.55)],
        hint="3D Dam break：水柱坍塌。左键搅动 / 右键加水 / R 重置 / 空格暂停",
    )


def build_tank(res=48, transfer=TransferMode.APIC) -> FluidScene:
    """平静水箱：下部充满水，适合观察自由表面与浮沉。"""
    cfg = _base_cfg(res, transfer)
    return FluidScene(
        name="tank",
        cfg=cfg,
        init_blocks=[(0.03, 0.03, 0.03, 0.97, 0.42, 0.97)],
        hint="水箱：左键搅动水面 / 右键加水 / R 重置",
    )


def build_obstacle(res=48, transfer=TransferMode.APIC) -> FluidScene:
    """带障碍：上方水块落到一个方块和一个球上，分流、绕流。"""
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
        hint="障碍绕流：水从上方落下，被方块/球分流。左键搅动 / R 重置",
    )


def build_jet(res=48, transfer=TransferMode.APIC) -> FluidScene:
    """喷口：一侧壁面喷口向内射水，逐渐充满水箱。"""
    cfg = _base_cfg(res, transfer, max_particles=1_500_000)
    dx = cfg.dx
    return FluidScene(
        name="jet",
        cfg=cfg,
        init_blocks=[(0.03, 0.03, 0.03, 0.97, 0.12, 0.97)],
        emitters=[Emitter(region=(2 * dx, 0.55, 0.42, 5 * dx, 0.68, 0.58),
                          velocity=(4.5, 0.0, 0.0), count=40)],
        hint="喷口：侧壁持续射水。左键搅动 / 右键加水 / R 重置",
    )


def build_softbody(res=40, transfer=TransferMode.APIC) -> FluidScene:
    """软体果冻：水流推动可变形软体，软体反过来作为运动障碍挡水。"""
    cfg = _base_cfg(res, transfer, viscosity=0.015, vel_damping=0.08,
                    particles_per_cell=6, max_particles=1_200_000)
    dx = cfg.dx
    soft_cfg = XPBDConfig(
        spacing=0.075,
        radius=0.052,
        density=0.42,
        stiffness=0.68,
        damping=3.2,
        drag=18.0,
        lift=0.8,
        fluid_coupling=1.0,
        solver_iters=10,
        max_speed=3.5,
    )
    jelly = XPBDSoftBody.make_box(center=(0.55, 0.48, 0.5), dims=(3, 4, 3),
                                  spacing=soft_cfg.spacing, cfg=soft_cfg)
    shapes = [
        Box(cx=0.5, cy=0.19, cz=0.5, hx=0.13, hy=0.035, hz=0.20),
    ]
    return FluidScene(
        name="softbody",
        cfg=cfg,
        shapes=shapes,
        soft_bodies=[jelly],
        init_blocks=[(0.03, 0.03, 0.03, 0.42, 0.38, 0.97)],
        emitters=[Emitter(region=(2 * dx, 0.36, 0.35, 5 * dx, 0.50, 0.65),
                          velocity=(3.2, 0.25, 0.0), count=22)],
        hint="软体耦合：水流推动果冻，果冻作为运动障碍改变水流。R 重置 / 拖动旋转",
    )


BUILDERS = {
    "dambreak": build_dambreak,
    "tank": build_tank,
    "obstacle": build_obstacle,
    "jet": build_jet,
    "softbody": build_softbody,
}


def build_scene(name: str, res=48, transfer=TransferMode.APIC) -> FluidScene:
    if name not in BUILDERS:
        raise ValueError(f"未知场景 '{name}'，可选：{list(BUILDERS)}")
    return BUILDERS[name](res=res, transfer=transfer)
