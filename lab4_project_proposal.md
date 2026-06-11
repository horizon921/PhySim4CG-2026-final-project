# Lab4 选题说明：三维流体驱动的 XPBD 软体解谜系统

## 一句话概括

我们实现了一个 **3D FLIP/PIC/APIC 流体 + XPBD 软体 + SDF 关卡障碍** 的交互式
物理仿真 Demo。玩家不直接拖动软体，而是通过喷口控制水流，把橙色 XPBD 果冻推入
绿色目标区，并避开红色危险区。

对应 Lab4 推荐方向：

- 异构多物理仿真系统
- 物理仿真驱动的游戏项目

## 最终项目定位

项目核心不是“做一个有水的画面”，而是把 **水流推动软体、软体反过来改变水流**
包装成一个可玩的关卡闭环：

```text
控制喷口 -> 形成水流 -> 推动 XPBD 软体 -> 软体写回动态边界 -> 完成目标区判定
```

最终 Demo 没有启用可动刚体机关。关卡中的障碍物采用 SDF 静态边界表示，重点展示
流体、软体和关卡逻辑之间的稳定耦合。

## 核心模块

### 1. 三维流体：FLIP / PIC / APIC

- 三维交错 MAC 网格 + 粒子自由表面。
- PCG 压力投影，保持近似不可压缩。
- 支持 PIC、FLIP、APIC 三种速度传输模式。
- 支持喷口、吸入口、重力、水箱边界和 SDF 障碍。
- 支持粘性、速度阻尼和 headless 对比出图。

### 2. 软体：XPBD 果冻体

- 使用三维距离约束晶格表示软体。
- 支持重力、阻尼、边界碰撞和流体速度拖拽。
- 软体节点可栅格化为运动 SDF 边界。
- 可以表现被水流推动、挤压、漂浮、封堵和改变流路的现象。

### 3. 双向耦合接口

流体求解器只依赖统一的固体边界字段：

```text
solid_phi                    单元中心 SDF
u_solid / v_solid / w_solid  交错面固体速度
```

耦合方向：

```text
流体 -> 软体：采样流体速度，对 XPBD 节点施加拖拽/推动
软体 -> 流体：软体栅格化为动态 SDF 与边界速度，参与压力投影和粒子投影
```

稳定性处理包括子步推进、耦合力限幅、接触阻尼、速度裁剪和低分辨率 headless 检查。

### 4. 关卡化 Demo

当前有三个软体水流小游戏关卡：

```text
soft_plug    入门封堵关：学习 checkpoint 停留、喷口控制和危险区
soft_slalom  绕障通行关：多 checkpoint + SDF 障碍绕行
soft_rescue  重物救援关：更重软体 + 水量预算 + 垂直托举
```

关卡元素：

```text
蓝色粒子       流体
橙色点/线框    XPBD 软体
绿色框         目标区 / checkpoint
红色框         危险区 / drain
灰色体素       SDF 静态障碍
黄色箭头       可控喷口方向
```

## 展示重点

- 3D 流体在喷口和边界条件下形成可控水流。
- XPBD 软体被水流推动，同时作为动态边界反过来改变流体。
- SDF 静态障碍、目标区、危险区和水量预算构成可玩的解谜规则。
- 2D 俯视图便于游玩，3D 视图展示真实流体-软体同场景耦合。
- headless 测试和参数对比图支撑工程稳定性。

## 推荐运行

```powershell
python -m coupledsim.app --scene soft_plug
python -m coupledsim.app --scene soft_slalom
python -m coupledsim.app --scene soft_rescue
python -m pytest tests/test_dynamic_coupling.py -q
```

更完整的运行方式见 [`game_run_guide.md`](game_run_guide.md)，玩法目标见
[`docs/game_objectives.md`](docs/game_objectives.md)。
