# coupledsim — 基于多物理耦合的软体水流解谜系统

Lab4 / Final Project。目标是实现一个 **流体 + 软体 + 刚体** 的二维多物理耦合
交互系统（详见 [`lab4_project_proposal.md`](lab4_project_proposal.md)）。

当前进度（第一阶段已完成 + 第二阶段进行中）：

- ✅ **统一工程框架**：配置、场景、渲染、耦合接口分层，各物理模块可独立测试。
- ✅ **流体模块（核心）**：二维 **FLIP / PIC / APIC**，交错 MAC 网格，
  Jacobi 预条件共轭梯度（PCG）压力投影，速度外插，**粘性扩散**，
  喷口 / 吸入口 / 障碍边界。
- ✅ **固体边界接口**：静态解析障碍（盒 / 圆 / 水箱壁）栅格化为有符号距离场，
  作为后续软体 / 刚体接入流体的统一通道。
- 🚧 软体（FEM/XPBD）、刚体机关、双向耦合：占位，后续阶段实现。

## 环境

使用 [uv](https://docs.astral.sh/uv/) 管理环境，[taichi](https://www.taichi-lang.org/) 作为计算后端。

```bash
uv venv --python 3.11
uv pip install taichi numpy
```

> 在 macOS 上，无 GPU 的离线计算用 `arch=ti.cpu`；交互窗口（`ti.GUI`）在 CPU 上也可运行。

## 运行

交互式流体演示（鼠标交互见窗口提示）：

```bash
uv run python -m coupledsim.app                # 默认 dam break 水箱
uv run python -m coupledsim.app --scene jet     # 喷口推水
uv run python -m coupledsim.app --scene obstacle
uv run python -m coupledsim.app --transfer flip --res 128
```

对比出图（报告 / 展示用，无需窗口）：

```bash
# PIC / FLIP / APIC 数值耗散对比
uv run python -m coupledsim.tools.compare_transfer --res 96 --frames 70
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

测试覆盖：压力投影散度下降、PIC/FLIP/APIC 稳定性、障碍阻挡、粘性耗散、渲染路径。

## 代码结构

```
src/coupledsim/
  config.py            # 全局 / 流体配置（dataclass）
  fluid/flip_solver.py # FLIP/PIC/APIC 求解器（核心）
  coupling/boundary.py # 固体边界（解析形状 -> 有符号距离场）= 耦合接口
  softbody/            # 软体（占位，后续阶段）
  rigid/               # 刚体 / 机关（占位，后续阶段）
  scene/               # 场景 / 关卡组装
  render/              # 二维可视化（ti.GUI）
  app.py               # 交互式入口
tests/                 # headless 检查
docs/framework.md      # 框架与算法说明
```

算法与坐标约定详见 [`docs/framework.md`](docs/framework.md)。
