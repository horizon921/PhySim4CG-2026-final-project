# Lab4 小游戏运行指南

这份文档面向组内同学和助教，用来快速配置环境、运行软体-流体小游戏关卡，并了解可以调哪些参数。

命令中的场景名、参数名、文件路径保持英文，方便直接复制运行。

## 1. 进入项目根目录

所有命令都需要在项目根目录下运行，也就是包含这些文件/目录的位置：

```text
pyproject.toml
src/
tests/
```

示例：

```powershell
cd path\to\PhySim4CG-2026-final-project
```

不要直接复制别人的本机绝对路径；换成你自己电脑上的项目路径。

## 2. 环境配置

推荐使用 Python `3.11`。

### 方式 A：使用已有 `.venv`

如果项目里已经有 `.venv/`：

```powershell
.\.venv\Scripts\Activate.ps1
python -m coupledsim.app --scene soft_plug
```

也可以不激活环境，直接调用 `.venv` 里的 Python：

```powershell
.\.venv\Scripts\python.exe -m coupledsim.app --scene soft_plug
```

### 方式 B：新建普通 venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

### 方式 C：使用 uv

```powershell
uv venv --python 3.11
uv pip install -e .
uv run python -m coupledsim.app --scene soft_plug
```

## 3. 场景列表

基础演示场景：

```text
dambreak    三维溃坝流体
tank        静水水箱
obstacle    带静态障碍物的流体
jet         侧向喷流演示
softbody    同学实现的 XPBD 软体-流体基础耦合 demo
```

软体小游戏关卡：

```text
soft_plug    入门关卡：把果冻体推过两个 checkpoint
soft_slalom  绕障碍关卡：多 checkpoint + 障碍绕行
soft_rescue  救援关卡：重软体 + 水量预算 + 托举封口
```

运行任意场景：

```powershell
python -m coupledsim.app --scene soft_plug
```

如果没有激活虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m coupledsim.app --scene soft_plug
```

## 4. 推荐第一次运行

建议先用低分辨率确认能跑：

```powershell
python -m coupledsim.app --scene soft_plug
```

窗口默认会先暂停显示。看到画面后按 `Space` 开始仿真；第一次开始时 Taichi 需要编译 kernel，可能等待 1-2 分钟。之后同一次运行里的后续帧会快很多。

小游戏默认使用 `--view map --hud compact`：这是更适合实时游玩的俯视关卡视图。它保留流体、软体、目标区、危险区和喷口方向，但比 3D 点云视图更流畅、更容易看懂。

确认能跑后，用可玩配置依次试玩：

```powershell
python -m coupledsim.app --scene soft_plug
python -m coupledsim.app --scene soft_slalom
python -m coupledsim.app --scene soft_rescue
```

如果机器性能足够，或者需要录制更清楚的演示画面，可以提高到：

```powershell
python -m coupledsim.app --scene soft_plug --res 16 --window 540 --quality pretty --color-by-velocity --view 3d --hud full
```

## 5. 操作方式

```text
W / S        按住控制玩家喷口向上 / 向下
A / D        按住控制玩家喷口的深度方向
I / K        兼容键：喷口向上 / 向下
J / L        兼容键：喷口深度方向
Q / E        减小 / 增大喷口速度
O            开关玩家喷口
U            对软体触发一次脉冲推动
R            重置场景
Space        暂停 / 继续
P / F / A    切换 PIC / FLIP / APIC 传输模式
G            开关重力
X            给流体施加一次中心推动
鼠标拖动      旋转视角
Esc          退出
```

## 6. 游戏规则

画面元素：

```text
蓝色粒子       流体
橙色点/线框    XPBD 软体果冻
亮绿色框       当前 checkpoint
暗绿色框       后续 checkpoint
红色框         危险区
灰色体素       固体边界 / 障碍物
```

游戏 UI：

```text
顶部状态栏       当前关卡、粒子数、FPS、暂停状态、游戏状态
蓝色进度条       剩余水量预算，旁边显示 WATER 数值
绿色进度条       当前 checkpoint 完成进度，旁边显示 GOAL 名称和百分比
右上蓝色块       玩家喷口状态，标注 JET ON / JET OFF
左侧竖条         游戏状态：绿色=进行中，青色=胜利，红色=失败
底部色块图例     蓝色=FLUID，橙色=SOFT，绿色=GOAL，红色=DANGER
黄色箭头         JET，表示当前水流喷口方向
H                展开 / 隐藏完整操作提示；默认只显示开始提示，避免遮挡关卡
```

窗口中的文字 HUD 使用英文短句，这是因为当前 `ti.GUI` 对中文字体支持不好，中文容易显示成方块。中文说明以本文档为准。

HUD 信息：

```text
status     当前状态：playing / won / lost
score      当前分数
target     当前 checkpoint 和完成进度
water      剩余水量
pulse_cd   脉冲冷却时间
event      最近事件或失败原因
```

更完整的玩法说明见：

```text
docs/game_objectives.md
```

各关卡目标摘要：

```text
soft_plug    入门封堵关
             先让软体在 align 内停留到 100%，再往右推入 seal。
             目标是学习 checkpoint 停留判定和喷口开关。
             需要避免软体长时间掉进底部红色 drain。

soft_slalom  绕障通行关
             控制软体绕过障碍物，依次通过 left gate、right gate、finish。
             目标是展示多 checkpoint、障碍物和方向控制。
             需要避免软体被冲入中下方 side drain。

soft_rescue  重物救援关
             软体更重，水量更紧。
             先用水流把软体托举到 lift，再推到 top seal。
             目标是展示垂直托举、水量管理和更难的软体控制。
```

`soft_plug` 新手流程：

```text
1. 按 Space 开始。第一次会编译，等画面重新动起来。
2. 先不要猛调方向，软体初始就在 ALIGN 附近。
3. 观察 GOAL ALIGN 进度，等它到 100%。
4. 目标切到 SEAL 后，关卡会轻微辅助喷口朝目标推进。
5. 用 W/S/A/D 微调黄色 JET 箭头，让水流把软体往右推。
6. 如果软体冲过头或水流太乱，按 O 暂停喷水，再按 O 恢复。
7. 赢了或输了都可以按 R 重开。
```

## 7. 运行时常用参数

```text
--scene       选择场景或关卡
--res         每个坐标轴的网格分辨率
--transfer    速度传输模式：pic / flip / apic
--arch        Taichi 后端：cpu / gpu / metal / vulkan / cuda
--window      窗口大小
--autoplay    启动后直接运行仿真；不加时默认先暂停
--render-particles GUI 中最多绘制多少流体粒子；0 表示全部绘制
--render-scale GUI 内部场景渲染比例；默认低于 1，文字 HUD 仍保持清晰
--color-by-velocity GUI 中按速度给流体上色；更美观但更慢，默认关闭
--quality GUI 质量预设：fast 更流畅，pretty 会绘制更多固体体素细节
--hud GUI 文字详细程度：compact 更清爽，full 显示额外标签和图例文字
--view 视角模式：map 是默认玩法俯视图，更流畅更清楚；3d 用于展示三维耦合效果
--headless    不开窗口，运行指定帧数
--save        headless 模式下的输出图片前缀
--save-every  每隔多少帧保存一张图
```

示例：

```powershell
python -m coupledsim.app --scene soft_plug --transfer apic
python -m coupledsim.app --scene soft_slalom --render-particles 1000
python -m coupledsim.app --scene soft_plug --res 16 --window 540 --render-scale 1.0 --quality pretty --hud full --view 3d
python -m coupledsim.app --scene soft_rescue --res 4 --headless 1
```

传输模式建议：

```text
APIC  默认推荐，比较稳定，视觉效果也较顺滑
FLIP  动感更强，但可能更 noisy
PIC   数值耗散更强，适合调稳定性
```

后端建议：

```text
cpu  当前最稳，推荐默认使用
gpu / cuda  可以尝试，但首帧仍会受 Taichi 编译影响，不一定比 CPU 更快
metal / vulkan  取决于本机 Taichi 支持情况
```

## 8. 代码中可以调的参数

关卡和游戏参数主要在：

```text
src/coupledsim/scene/levels.py
```

常用参数：

```text
XPBDConfig.spacing       软体节点间距
XPBDConfig.radius        软体节点碰撞/栅格化半径
XPBDConfig.density       软体质量尺度
XPBDConfig.stiffness     软体约束刚度
XPBDConfig.damping       软体阻尼
XPBDConfig.drag          流体拖拽强度
FluidSoftCoupler.drag_coeff
FluidSoftCoupler.max_point_force
GameZone.region          checkpoint / hazard 的空间范围
GameZone.hold_frames     需要在区域内停留的帧数
water_budget             总水量资源
pulse_cost               使用脉冲消耗的水量
pulse_cooldown_frames    脉冲冷却时间
max_frames               关卡时间限制
```

流体参数主要在：

```text
src/coupledsim/config.py
```

常用参数：

```text
dt
substeps
gravity_x / gravity_y / gravity_z
viscosity
vel_damping
particles_per_cell
max_particles
pressure_iters / cg_max_iters / cg_tol
```

## 9. 性能建议

三维仿真开销随分辨率增长很快。

```text
res 4-6    smoke test，确认环境和窗口能打开
res 8      推荐试玩，交互相对更顺
res 10-12  画面更清楚，但帧率会下降
res 16     展示截图/录屏用，不推荐实时游玩
res 24+    CPU 上通常不适合实时游玩
```

如果太卡，可以：

```text
1. 降低 --res，优先试 --res 8 或 --res 6
2. 降低窗口大小，例如 --window 340
3. 降低内部渲染比例，例如 --render-scale 0.5
4. 降低 GUI 粒子绘制上限，例如 --render-particles 800
5. 使用 --transfer pic 提高数值耗散，通常也更稳
6. 减小 levels.py 里的 emitter count
7. 减小 XPBDConfig.solver_iters 或软体晶格 dims
```

## 10. Headless 检查

不开窗口，只跑一帧：

```powershell
python -m coupledsim.app --scene soft_plug --res 4 --headless 1
```

保存图片：

```powershell
python -m coupledsim.app --scene soft_plug --res 8 --headless 10 --save outputs\soft_plug --save-every 5
```

默认轻量测试：

```powershell
python -m pytest tests\test_dynamic_coupling.py -q
```

软体基础测试：

```powershell
python -m pytest tests\test_softbody_headless.py -q -k "not scene_coupling"
```

慢速多关卡 Taichi step 检查：

```powershell
$env:COUPLEDSIM_RUN_SLOW="1"
python -m pytest tests\test_dynamic_coupling.py -q
Remove-Item Env:\COUPLEDSIM_RUN_SLOW
```

## 11. 推荐展示顺序

```text
1. softbody      展示同学的 XPBD 软体-流体基础耦合
2. soft_plug     展示基础游戏循环
3. soft_slalom   展示多 checkpoint 和障碍绕行
4. soft_rescue   展示资源管理和更重软体
```

## 12. 常见问题

问题：找不到 `coupledsim`。

```text
确认当前目录是项目根目录，并运行：
python -m pip install -e .
```

问题：PowerShell 不允许激活虚拟环境。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

问题：窗口能打开，但仿真很慢。

```text
小游戏场景不写参数时会自动使用较流畅的试玩配置；如果还卡，改用 --res 6 --window 340。
默认启动后是暂停状态，按 Space 后第一次仿真会触发 Taichi 编译，可能等待 1-2 分钟。
如果只想检查能不能跑，可以先用 --scene soft_plug --res 4 --headless 1。
CPU 是当前最稳的后端；GPU 后端首帧也可能不会更快。
```

问题：GPU 后端卡住或崩溃。

```powershell
python -m coupledsim.app --scene soft_plug --arch cpu
```
