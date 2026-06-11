"""coupledsim —— 基于多物理耦合的软体水流解谜系统。

模块划分：
    fluid    : FLIP/PIC/APIC 流体（本阶段核心，已实现）
    coupling : 固体边界 / 流固耦合接口
    softbody : XPBD 软体与动态边界
    scene    : 场景与关卡组装
    render   : 2D / 3D 可视化与离屏出图
"""

__version__ = "0.1.0"
