"""预定义关卡 / 演示场景。

每个 build_* 返回一个 FluidScene。CLI（app.py）通过名字选择。
注意：必须在 ti.init() 之后调用（构建时会分配 taichi 字段）。
"""

from ..config import FluidConfig, TransferMode
from ..coupling import Box, Circle
from .scene import Emitter, FluidScene


def _base_cfg(res: int, transfer: TransferMode, **kw) -> FluidConfig:
    cfg = FluidConfig(res_x=res, res_y=res, domain_x=1.0, transfer=transfer)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def build_dambreak(res=128, transfer=TransferMode.APIC) -> FluidScene:
    """经典溃坝：左侧一根水柱在重力下坍塌、铺开、拍打右壁。"""
    cfg = _base_cfg(res, transfer)
    return FluidScene(
        name="dambreak",
        cfg=cfg,
        init_blocks=[(0.04, 0.04, 0.36, 0.92)],
        hint="Dam break：左侧水柱坍塌。左键搅动 / 右键加水 / R 重置 / 空格暂停",
    )


def build_tank(res=128, transfer=TransferMode.APIC) -> FluidScene:
    """平静水箱：下半部充满水，适合观察自由表面与浮沉。"""
    cfg = _base_cfg(res, transfer)
    return FluidScene(
        name="tank",
        cfg=cfg,
        init_blocks=[(0.03, 0.03, 0.97, 0.45)],
        hint="水箱：左键搅动水面 / 右键加水 / R 重置",
    )


def build_obstacle(res=128, transfer=TransferMode.APIC) -> FluidScene:
    """带障碍：水块落到一个方块和一个圆柱上，分流、绕流。"""
    cfg = _base_cfg(res, transfer)
    shapes = [
        Box(cx=0.5, cy=0.30, hx=0.13, hy=0.07),
        Circle(cx=0.26, cy=0.16, r=0.07),
    ]
    return FluidScene(
        name="obstacle",
        cfg=cfg,
        shapes=shapes,
        init_blocks=[(0.05, 0.55, 0.95, 0.95)],
        hint="障碍绕流：水从上方落下，被方块/圆柱分流。左键搅动 / R 重置",
    )


def build_jet(res=128, transfer=TransferMode.APIC) -> FluidScene:
    """喷口：左壁喷口向右射水，逐渐充满水箱（流固耦合演示的前身）。"""
    cfg = _base_cfg(res, transfer, max_particles=200_000)
    dx = cfg.dx
    return FluidScene(
        name="jet",
        cfg=cfg,
        init_blocks=[(0.03, 0.03, 0.97, 0.12)],
        emitters=[Emitter(region=(2 * dx, 0.58, 5 * dx, 0.70),
                          velocity=(4.5, 0.0), count=24)],
        hint="喷口：左壁持续射水。左键搅动 / 右键加水 / R 重置",
    )


BUILDERS = {
    "dambreak": build_dambreak,
    "tank": build_tank,
    "obstacle": build_obstacle,
    "jet": build_jet,
}


def build_scene(name: str, res=128, transfer=TransferMode.APIC) -> FluidScene:
    if name not in BUILDERS:
        raise ValueError(f"未知场景 '{name}'，可选：{list(BUILDERS)}")
    return BUILDERS[name](res=res, transfer=transfer)
