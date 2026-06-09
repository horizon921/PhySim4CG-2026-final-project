# coupledsim — 基于多物理耦合的软体水流解谜系统

Lab4 / Final Project。目标是实现一个 **流体 + 软体 + 刚体** 的 **三维** 多物理耦合
交互系统（详见 [`lab4_project_proposal.md`](lab4_project_proposal.md)）。

当前进度（第一阶段已完成 + 第二阶段进行中）：

- ✅ **统一工程框架**：配置、场景、渲染、耦合接口分层，各物理模块可独立测试。
- ✅ **流体模块（核心，三维）**：**FLIP / PIC / APIC**，三维交错 MAC 网格，
  Jacobi 预条件共轭梯度（PCG）压力投影，速度外插，**粘性扩散**，
  喷口 / 吸入口 / 障碍边界。
- ✅ **固体边界接口**：静态解析障碍（盒 / 球 / 水箱壁）栅格化为有符号距离场，
  作为后续软体 / 刚体接入流体的统一通道。
- 🚧 软体（FEM/XPBD）、刚体机关、双向耦合：占位，后续阶段实现。

## 环境

使用 [uv](https://docs.astral.sh/uv/) 管理环境，[taichi](https://www.taichi-lang.org/) 作为计算后端。

```bash
uv venv --python 3.11
uv pip install -e .            # 或：uv pip install taichi numpy
```

> **后端**：默认 `arch=ti.cpu`，在本机最稳。GPU（metal/vulkan）后端在本机会卡住
> （压力 PCG 的逐迭代标量读回导致同步停顿），故暂不作默认。
> 交互窗口用 `ti.GUI`（CPU 软件光栅）显示三维点云投影，无需 vulkan/GGUI。

## 运行

交互式三维流体演示（鼠标左键拖动旋转相机）：

```bash
uv run python -m coupledsim.app                 # 默认 3D dam break 水箱
uv run python -m coupledsim.app --scene jet
uv run python -m coupledsim.app --scene obstacle
uv run python -m coupledsim.app --transfer flip --res 48
```

> `--res` 控制每轴网格分辨率（三维开销 ∝ res³）：CPU 上 32 较流畅、48 更精细但更慢。

对比出图（报告 / 展示用，无需窗口）：

```bash
# PIC / FLIP / APIC 数值耗散对比
uv run python -m coupledsim.tools.compare_transfer --res 32 --frames 50
# 任意参数对比（粘性 / FLIP ratio / 压力精度 ...）
uv run python -m coupledsim.tools.compare_param --param viscosity --values 0,0.03,0.12
uv run python -m coupledsim.tools.compare_param --param flip_ratio --values 0.0,0.95,0.99 --transfer flip
```

headless 正确性 / 稳定性检查（无需窗口，CI 友好）：

```bash
uv run python tests/test_fluid_headless.py
# 或
uv run pytest
```

测试覆盖：压力投影散度下降、PIC/FLIP/APIC 稳定性、障碍阻挡、粘性耗散、三维渲染路径。

## 代码结构

```
src/coupledsim/
  config.py            # 全局 / 流体配置（dataclass）
  fluid/flip_solver.py # 三维 FLIP/PIC/APIC 求解器（核心）
  coupling/boundary.py # 固体边界（解析形状 -> 有符号距离场）= 耦合接口
  softbody/            # 软体（占位，后续阶段）
  rigid/               # 刚体 / 机关（占位，后续阶段）
  scene/               # 场景 / 关卡组装
  render/              # 三维离屏渲染 + ti.GUI 软件查看器
  tools/               # 对比出图（传输模式 / 参数）
  app.py               # 交互式入口
tests/                 # headless 检查
docs/framework.md      # 框架与算法说明
```

算法与坐标约定详见 [`docs/framework.md`](docs/framework.md)。
