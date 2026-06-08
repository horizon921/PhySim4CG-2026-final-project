"""二维 FLIP / PIC / APIC 流体求解器（交错 MAC 网格 + 粒子）。

算法总览（每个子步）：
    1. P2G   : 粒子速度（及 APIC 仿射项）散布到交错网格速度面上
    2. 归一化 : 网格速度 = 加权平均；保存投影前速度供 FLIP 使用
    3. 标记单元 : 由固体场 + 粒子占据情况标记 SOLID / FLUID / AIR
    4. 体力   : 重力加到网格速度
    5. 边界   : 固体面速度置为固体速度（静态即 0）
    6. 投影   : 求解压力 Poisson，使速度近似无散（不可压缩）
    7. 外插   : 把流体速度外插到空气区域，减少自由表面伪影
    8. G2P    : 网格速度（或增量）插值回粒子，更新 APIC 矩阵 C
    9. 平流   : RK2 用网格速度场推进粒子，并做边界 / 障碍投影

坐标约定：世界坐标 [0, Lx] x [0, Ly]，单元大小 dx；
    u (x 速度) 存于竖直面，形状 (nx+1, ny)，位置 (i*dx, (j+0.5)*dx)
    v (y 速度) 存于水平面，形状 (nx, ny+1)，位置 ((i+0.5)*dx, j*dx)
    压力 / 单元类型存于单元中心，形状 (nx, ny)
"""

# 注意：本文件含 @ti.kernel，不能使用 `from __future__ import annotations`，
# 否则内核参数注解会被 PEP 563 转成字符串，taichi 无法识别其类型。

import numpy as np
import taichi as ti

from ..config import FluidConfig, TransferMode

# 单元类型
AIR = 0
FLUID = 1
SOLID = 2


@ti.data_oriented
class FlipSolver:
    def __init__(self, cfg: FluidConfig):
        self.cfg = cfg
        nx, ny = cfg.res_x, cfg.res_y
        self.nx, self.ny = nx, ny
        self.dx = cfg.dx
        self.inv_dx = 1.0 / cfg.dx
        self.rho = cfg.rho

        # --- MAC 网格速度场 ---
        self.u = ti.field(ti.f32, shape=(nx + 1, ny))
        self.v = ti.field(ti.f32, shape=(nx, ny + 1))
        self.u_old = ti.field(ti.f32, shape=(nx + 1, ny))
        self.v_old = ti.field(ti.f32, shape=(nx, ny + 1))
        self.u_w = ti.field(ti.f32, shape=(nx + 1, ny))   # P2G 权重累加
        self.v_w = ti.field(ti.f32, shape=(nx, ny + 1))

        # 外插用临时场与有效标记
        self.u_tmp = ti.field(ti.f32, shape=(nx + 1, ny))
        self.v_tmp = ti.field(ti.f32, shape=(nx, ny + 1))
        self.valid_u = ti.field(ti.i32, shape=(nx + 1, ny))
        self.valid_v = ti.field(ti.i32, shape=(nx, ny + 1))
        self.valid_u_tmp = ti.field(ti.i32, shape=(nx + 1, ny))
        self.valid_v_tmp = ti.field(ti.i32, shape=(nx, ny + 1))

        # --- 单元中心量 ---
        self.pressure = ti.field(ti.f32, shape=(nx, ny))
        self.divergence = ti.field(ti.f32, shape=(nx, ny))
        self.cell_type = ti.field(ti.i32, shape=(nx, ny))

        # 压力 PCG 工作场
        self.cg_r = ti.field(ti.f32, shape=(nx, ny))     # 残差
        self.cg_d = ti.field(ti.f32, shape=(nx, ny))     # 搜索方向
        self.cg_q = ti.field(ti.f32, shape=(nx, ny))     # A·d
        self.cg_z = ti.field(ti.f32, shape=(nx, ny))     # 预条件后残差
        self.cg_diag = ti.field(ti.f32, shape=(nx, ny))  # 对角元（Jacobi 预条件）

        # --- 固体边界（耦合接口） ---
        self.solid_phi = ti.field(ti.f32, shape=(nx, ny))   # < 0 在固体内
        self.u_solid = ti.field(ti.f32, shape=(nx + 1, ny))
        self.v_solid = ti.field(ti.f32, shape=(nx, ny + 1))

        # --- 粒子 ---
        self.max_particles = cfg.max_particles
        self.px = ti.Vector.field(2, ti.f32, shape=self.max_particles)
        self.pv = ti.Vector.field(2, ti.f32, shape=self.max_particles)
        self.C = ti.Matrix.field(2, 2, ti.f32, shape=self.max_particles)
        # 压缩 / 发射用副本
        self.px2 = ti.Vector.field(2, ti.f32, shape=self.max_particles)
        self.pv2 = ti.Vector.field(2, ti.f32, shape=self.max_particles)
        self.C2 = ti.Matrix.field(2, 2, ti.f32, shape=self.max_particles)
        self.n_particles = ti.field(ti.i32, shape=())
        self.n_tmp = ti.field(ti.i32, shape=())

        # 诊断量
        self.max_vel = ti.field(ti.f32, shape=())
        self.max_div = ti.field(ti.f32, shape=())

        self.solid_phi.fill(1e9)
        self.n_particles[None] = 0

    # ===================================================================== #
    # 初始化 / 固体场
    # ===================================================================== #
    def set_solid_phi(self, phi_np: np.ndarray):
        """从 numpy (nx, ny) 设置固体有符号距离场（单元中心）。"""
        assert phi_np.shape == (self.nx, self.ny)
        self.solid_phi.from_numpy(phi_np.astype(np.float32))
        self.u_solid.fill(0.0)
        self.v_solid.fill(0.0)

    def add_particle_block(self, x0, y0, x1, y1, jitter=True):
        """在世界坐标矩形 [x0,x1]x[y0,y1] 内按 particles_per_cell 填充粒子。

        使用 numpy 生成（确定性，受 cfg.seed 控制），再追加到粒子缓冲尾部。
        会跳过落在固体内部的粒子。
        """
        cfg = self.cfg
        dx = self.dx
        ppc = cfg.particles_per_cell
        # 每个单元放置 sub x sub 个粒子（尽量接近 ppc）
        sub = max(1, int(round(np.sqrt(ppc))))
        i0, i1 = int(np.floor(x0 / dx)), int(np.ceil(x1 / dx))
        j0, j1 = int(np.floor(y0 / dx)), int(np.ceil(y1 / dx))
        rng = np.random.default_rng(cfg.seed + self.n_particles[None])
        pts = []
        phi = self.solid_phi.to_numpy()
        for i in range(i0, i1):
            for j in range(j0, j1):
                for si in range(sub):
                    for sj in range(sub):
                        if jitter:
                            ox, oy = rng.random(), rng.random()
                        else:
                            ox, oy = 0.5, 0.5
                        x = (i + (si + ox) / sub) * dx
                        y = (j + (sj + oy) / sub) * dx
                        if not (x0 <= x <= x1 and y0 <= y <= y1):
                            continue
                        ci = min(max(int(x / dx), 0), self.nx - 1)
                        cj = min(max(int(y / dx), 0), self.ny - 1)
                        if phi[ci, cj] < 0:   # 固体内部，跳过
                            continue
                        pts.append((x, y))
        if not pts:
            return 0
        pts = np.array(pts, dtype=np.float32)
        n0 = self.n_particles[None]
        n_add = min(len(pts), self.max_particles - n0)
        if n_add <= 0:
            return 0
        self._upload_particles(pts[:n_add], n0)
        self.n_particles[None] = n0 + n_add
        return n_add

    def _upload_particles(self, pts: np.ndarray, offset: int):
        """把 numpy 位置写入粒子缓冲 [offset, offset+len)，速度/仿射清零。"""
        n = len(pts)
        full = self.px.to_numpy()
        full[offset:offset + n] = pts
        self.px.from_numpy(full)
        # 速度与 C 清零（仅新加入的区间；简单起见整体保证零初值由发射控制）
        self._zero_particle_state(offset, offset + n)

    @ti.kernel
    def _zero_particle_state(self, lo: ti.i32, hi: ti.i32):
        for p in range(lo, hi):
            self.pv[p] = ti.Vector([0.0, 0.0])
            self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)

    # ===================================================================== #
    # 网格辅助函数
    # ===================================================================== #
    @ti.func
    def is_solid(self, i, j) -> ti.i32:
        res = 0
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            res = 1
        elif self.cell_type[i, j] == SOLID:
            res = 1
        return res

    @ti.func
    def is_fluid(self, i, j) -> ti.i32:
        res = 0
        if 0 <= i < self.nx and 0 <= j < self.ny:
            if self.cell_type[i, j] == FLUID:
                res = 1
        return res

    @ti.func
    def sample_u(self, x):
        gx = ti.min(ti.max(x[0] * self.inv_dx, 0.0), float(self.nx))
        gy = ti.min(ti.max(x[1] * self.inv_dx - 0.5, 0.0), float(self.ny - 1))
        i = ti.min(int(gx), self.nx - 1)
        j = ti.min(int(gy), self.ny - 2)
        fx = gx - i
        fy = gy - j
        return (self.u[i, j] * (1 - fx) * (1 - fy)
                + self.u[i + 1, j] * fx * (1 - fy)
                + self.u[i, j + 1] * (1 - fx) * fy
                + self.u[i + 1, j + 1] * fx * fy)

    @ti.func
    def sample_v(self, x):
        gx = ti.min(ti.max(x[0] * self.inv_dx - 0.5, 0.0), float(self.nx - 1))
        gy = ti.min(ti.max(x[1] * self.inv_dx, 0.0), float(self.ny))
        i = ti.min(int(gx), self.nx - 2)
        j = ti.min(int(gy), self.ny - 1)
        fx = gx - i
        fy = gy - j
        return (self.v[i, j] * (1 - fx) * (1 - fy)
                + self.v[i + 1, j] * fx * (1 - fy)
                + self.v[i, j + 1] * (1 - fx) * fy
                + self.v[i + 1, j + 1] * fx * fy)

    @ti.func
    def sample_velocity(self, x):
        return ti.Vector([self.sample_u(x), self.sample_v(x)])

    @ti.func
    def sample_phi(self, x):
        gx = ti.min(ti.max(x[0] * self.inv_dx - 0.5, 0.0), float(self.nx - 1))
        gy = ti.min(ti.max(x[1] * self.inv_dx - 0.5, 0.0), float(self.ny - 1))
        i = ti.min(int(gx), self.nx - 2)
        j = ti.min(int(gy), self.ny - 2)
        fx = gx - i
        fy = gy - j
        return (self.solid_phi[i, j] * (1 - fx) * (1 - fy)
                + self.solid_phi[i + 1, j] * fx * (1 - fy)
                + self.solid_phi[i, j + 1] * (1 - fx) * fy
                + self.solid_phi[i + 1, j + 1] * fx * fy)

    @ti.func
    def phi_gradient(self, x):
        h = 0.5 * self.dx
        gx = self.sample_phi(x + ti.Vector([h, 0.0])) - self.sample_phi(x - ti.Vector([h, 0.0]))
        gy = self.sample_phi(x + ti.Vector([0.0, h])) - self.sample_phi(x - ti.Vector([0.0, h]))
        g = ti.Vector([gx, gy])
        n = g.norm()
        if n > 1e-8:
            g = g / n
        else:
            g = ti.Vector([0.0, 1.0])
        return g

    # ===================================================================== #
    # 1. P2G —— 粒子速度散布到网格
    # ===================================================================== #
    @ti.kernel
    def p2g(self, apic: ti.i32):
        for I in ti.grouped(self.u):
            self.u[I] = 0.0
            self.u_w[I] = 0.0
        for I in ti.grouped(self.v):
            self.v[I] = 0.0
            self.v_w[I] = 0.0

        for p in range(self.n_particles[None]):
            xp = self.px[p]
            vp = self.pv[p]
            Cp = self.C[p]

            # ---- 散布到 u 网格 (component 0) ----
            gp = ti.Vector([xp[0] * self.inv_dx, xp[1] * self.inv_dx - 0.5])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            fx = gp - base.cast(ti.f32)
            wx = ti.Vector([0.5 * (1.5 - fx[0]) ** 2,
                            0.75 - (fx[0] - 1.0) ** 2,
                            0.5 * (fx[0] - 0.5) ** 2])
            wy = ti.Vector([0.5 * (1.5 - fx[1]) ** 2,
                            0.75 - (fx[1] - 1.0) ** 2,
                            0.5 * (fx[1] - 0.5) ** 2])
            for a, b in ti.static(ti.ndrange(3, 3)):
                ni, nj = base[0] + a, base[1] + b
                if 0 <= ni <= self.nx and 0 <= nj < self.ny:
                    w = wx[a] * wy[b]
                    node_pos = ti.Vector([ni * self.dx, (nj + 0.5) * self.dx])
                    val = vp[0]
                    if apic == 1:
                        val += Cp[0, 0] * (node_pos[0] - xp[0]) + Cp[0, 1] * (node_pos[1] - xp[1])
                    self.u[ni, nj] += w * val
                    self.u_w[ni, nj] += w

            # ---- 散布到 v 网格 (component 1) ----
            gp = ti.Vector([xp[0] * self.inv_dx - 0.5, xp[1] * self.inv_dx])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            fx = gp - base.cast(ti.f32)
            wx = ti.Vector([0.5 * (1.5 - fx[0]) ** 2,
                            0.75 - (fx[0] - 1.0) ** 2,
                            0.5 * (fx[0] - 0.5) ** 2])
            wy = ti.Vector([0.5 * (1.5 - fx[1]) ** 2,
                            0.75 - (fx[1] - 1.0) ** 2,
                            0.5 * (fx[1] - 0.5) ** 2])
            for a, b in ti.static(ti.ndrange(3, 3)):
                ni, nj = base[0] + a, base[1] + b
                if 0 <= ni < self.nx and 0 <= nj <= self.ny:
                    w = wx[a] * wy[b]
                    node_pos = ti.Vector([(ni + 0.5) * self.dx, nj * self.dx])
                    val = vp[1]
                    if apic == 1:
                        val += Cp[1, 0] * (node_pos[0] - xp[0]) + Cp[1, 1] * (node_pos[1] - xp[1])
                    self.v[ni, nj] += w * val
                    self.v_w[ni, nj] += w

    @ti.kernel
    def normalize_grid(self):
        for I in ti.grouped(self.u):
            if self.u_w[I] > 1e-8:
                self.u[I] /= self.u_w[I]
            else:
                self.u[I] = 0.0
            self.u_old[I] = self.u[I]
        for I in ti.grouped(self.v):
            if self.v_w[I] > 1e-8:
                self.v[I] /= self.v_w[I]
            else:
                self.v[I] = 0.0
            self.v_old[I] = self.v[I]

    # ===================================================================== #
    # 3. 单元分类
    # ===================================================================== #
    @ti.kernel
    def classify_cells(self):
        for i, j in self.cell_type:
            if self.solid_phi[i, j] < 0.0:
                self.cell_type[i, j] = SOLID
            else:
                self.cell_type[i, j] = AIR
        for p in range(self.n_particles[None]):
            xp = self.px[p]
            ci = int(xp[0] * self.inv_dx)
            cj = int(xp[1] * self.inv_dx)
            if 0 <= ci < self.nx and 0 <= cj < self.ny:
                if self.cell_type[ci, cj] != SOLID:
                    self.cell_type[ci, cj] = FLUID

    # ===================================================================== #
    # 4 & 5. 体力 + 固体边界
    # ===================================================================== #
    @ti.kernel
    def add_gravity(self, sdt: ti.f32, gx: ti.f32, gy: ti.f32):
        for I in ti.grouped(self.u):
            if self.u_w[I] > 1e-8:
                self.u[I] += gx * sdt
        for I in ti.grouped(self.v):
            if self.v_w[I] > 1e-8:
                self.v[I] += gy * sdt

    @ti.kernel
    def apply_solid_boundaries(self):
        for i, j in self.u:
            if i == 0 or i == self.nx or self.is_solid(i - 1, j) or self.is_solid(i, j):
                self.u[i, j] = self.u_solid[i, j]
        for i, j in self.v:
            if j == 0 or j == self.ny or self.is_solid(i, j - 1) or self.is_solid(i, j):
                self.v[i, j] = self.v_solid[i, j]

    # ===================================================================== #
    # 6. 压力投影（红黑 Gauss-Seidel）
    # ===================================================================== #
    @ti.kernel
    def compute_divergence(self):
        for i, j in self.divergence:
            if self.cell_type[i, j] == FLUID:
                self.divergence[i, j] = (self.u[i + 1, j] - self.u[i, j]
                                         + self.v[i, j + 1] - self.v[i, j]) * self.inv_dx
            else:
                self.divergence[i, j] = 0.0

    @ti.kernel
    def pressure_jacobi_color(self, scale: ti.f32, color: ti.i32):
        for i, j in self.pressure:
            if self.cell_type[i, j] == FLUID and (i + j) % 2 == color:
                nonsolid = 0
                sum_p = 0.0
                # 左
                if not self.is_solid(i - 1, j):
                    nonsolid += 1
                    if self.is_fluid(i - 1, j):
                        sum_p += self.pressure[i - 1, j]
                # 右
                if not self.is_solid(i + 1, j):
                    nonsolid += 1
                    if self.is_fluid(i + 1, j):
                        sum_p += self.pressure[i + 1, j]
                # 下
                if not self.is_solid(i, j - 1):
                    nonsolid += 1
                    if self.is_fluid(i, j - 1):
                        sum_p += self.pressure[i, j - 1]
                # 上
                if not self.is_solid(i, j + 1):
                    nonsolid += 1
                    if self.is_fluid(i, j + 1):
                        sum_p += self.pressure[i, j + 1]
                if nonsolid > 0:
                    self.pressure[i, j] = (sum_p - self.divergence[i, j] / scale) / nonsolid

    def solve_pressure_gs(self, sdt: float):
        self.pressure.fill(0.0)
        scale = sdt / (self.rho * self.dx * self.dx)
        for _ in range(self.cfg.pressure_iters):
            self.pressure_jacobi_color(scale, 0)
            self.pressure_jacobi_color(scale, 1)

    # ---- Jacobi 预条件共轭梯度 (PCG) ----
    @ti.func
    def _nonsolid_count(self, i, j):
        c = 0
        if not self.is_solid(i - 1, j):
            c += 1
        if not self.is_solid(i + 1, j):
            c += 1
        if not self.is_solid(i, j - 1):
            c += 1
        if not self.is_solid(i, j + 1):
            c += 1
        return c

    @ti.kernel
    def _apply_A(self, scale: ti.f32, src: ti.template(), dst: ti.template()):
        """无矩阵地计算 (A·src)，A 为压力 Poisson 算子（仅流体单元）。"""
        for i, j in self.cell_type:
            val = 0.0
            if self.cell_type[i, j] == FLUID:
                nonsolid = self._nonsolid_count(i, j)
                s = 0.0
                if self.is_fluid(i - 1, j):
                    s += src[i - 1, j]
                if self.is_fluid(i + 1, j):
                    s += src[i + 1, j]
                if self.is_fluid(i, j - 1):
                    s += src[i, j - 1]
                if self.is_fluid(i, j + 1):
                    s += src[i, j + 1]
                val = scale * (nonsolid * src[i, j] - s)
            dst[i, j] = val

    @ti.kernel
    def _cg_init(self, scale: ti.f32) -> ti.f32:
        # x = 0 假设 -> r = b - A x = b = -divergence；并构造 Jacobi 预条件对角
        for i, j in self.cell_type:
            self.cg_q[i, j] = 0.0
            if self.cell_type[i, j] == FLUID:
                nonsolid = self._nonsolid_count(i, j)
                self.cg_diag[i, j] = scale * nonsolid if nonsolid > 0 else 1.0
                self.cg_r[i, j] = -self.divergence[i, j]
            else:
                self.cg_diag[i, j] = 1.0
                self.cg_r[i, j] = 0.0
        s = 0.0
        for i, j in self.cell_type:
            if self.cell_type[i, j] == FLUID:
                z = self.cg_r[i, j] / self.cg_diag[i, j]
                self.cg_z[i, j] = z
                self.cg_d[i, j] = z
                s += self.cg_r[i, j] * z
            else:
                self.cg_z[i, j] = 0.0
                self.cg_d[i, j] = 0.0
        return s

    @ti.kernel
    def _dot(self, a: ti.template(), b: ti.template()) -> ti.f32:
        s = 0.0
        for i, j in self.cell_type:
            if self.cell_type[i, j] == FLUID:
                s += a[i, j] * b[i, j]
        return s

    @ti.kernel
    def _cg_update_xr(self, alpha: ti.f32):
        for i, j in self.cell_type:
            if self.cell_type[i, j] == FLUID:
                self.pressure[i, j] += alpha * self.cg_d[i, j]
                self.cg_r[i, j] -= alpha * self.cg_q[i, j]

    @ti.kernel
    def _cg_precond_dot(self) -> ti.f32:
        s = 0.0
        for i, j in self.cell_type:
            if self.cell_type[i, j] == FLUID:
                z = self.cg_r[i, j] / self.cg_diag[i, j]
                self.cg_z[i, j] = z
                s += self.cg_r[i, j] * z
        return s

    @ti.kernel
    def _cg_update_d(self, beta: ti.f32):
        for i, j in self.cell_type:
            if self.cell_type[i, j] == FLUID:
                self.cg_d[i, j] = self.cg_z[i, j] + beta * self.cg_d[i, j]

    # 融合内核：减少 CPU 上的内核启动 / 同步次数（压力求解是主要瓶颈）
    @ti.kernel
    def _cg_apply_A_dot(self, scale: ti.f32) -> ti.f32:
        """一遍完成 q = A·d 与 d·q，减少一次同步。"""
        s = 0.0
        for i, j in self.cell_type:
            q = 0.0
            if self.cell_type[i, j] == FLUID:
                nonsolid = self._nonsolid_count(i, j)
                ssum = 0.0
                if self.is_fluid(i - 1, j):
                    ssum += self.cg_d[i - 1, j]
                if self.is_fluid(i + 1, j):
                    ssum += self.cg_d[i + 1, j]
                if self.is_fluid(i, j - 1):
                    ssum += self.cg_d[i, j - 1]
                if self.is_fluid(i, j + 1):
                    ssum += self.cg_d[i, j + 1]
                q = scale * (nonsolid * self.cg_d[i, j] - ssum)
                s += self.cg_d[i, j] * q
            self.cg_q[i, j] = q
        return s

    @ti.kernel
    def _cg_update_precond_dot(self, alpha: ti.f32) -> ti.f32:
        """一遍完成 x+=αd, r-=αq, z=M⁻¹r, 并返回新的 r·z。"""
        s = 0.0
        for i, j in self.cell_type:
            if self.cell_type[i, j] == FLUID:
                self.pressure[i, j] += alpha * self.cg_d[i, j]
                r = self.cg_r[i, j] - alpha * self.cg_q[i, j]
                self.cg_r[i, j] = r
                z = r / self.cg_diag[i, j]
                self.cg_z[i, j] = z
                s += r * z
        return s

    def solve_pressure_cg(self, sdt: float):
        self.pressure.fill(0.0)
        scale = sdt / (self.rho * self.dx * self.dx)
        delta_new = self._cg_init(scale)
        if delta_new <= 1e-20:
            return
        delta0 = delta_new
        tol2 = self.cfg.cg_tol ** 2
        for _ in range(self.cfg.cg_max_iters):
            dq = self._cg_apply_A_dot(scale)
            if abs(dq) < 1e-30:
                break
            alpha = delta_new / dq
            delta_old = delta_new
            delta_new = self._cg_update_precond_dot(alpha)
            if delta_new <= tol2 * delta0:
                break
            beta = delta_new / delta_old
            self._cg_update_d(beta)

    def solve_pressure(self, sdt: float):
        if self.cfg.use_cg:
            self.solve_pressure_cg(sdt)
        else:
            self.solve_pressure_gs(sdt)

    @ti.kernel
    def apply_pressure(self, sdt: ti.f32):
        coef = sdt / (self.rho * self.dx)
        for i, j in self.u:
            if 0 < i < self.nx:
                if self.is_solid(i - 1, j) or self.is_solid(i, j):
                    self.u[i, j] = self.u_solid[i, j]
                elif self.is_fluid(i - 1, j) or self.is_fluid(i, j):
                    pr = self.pressure[i, j] if self.is_fluid(i, j) else 0.0
                    pl = self.pressure[i - 1, j] if self.is_fluid(i - 1, j) else 0.0
                    self.u[i, j] -= coef * (pr - pl)
        for i, j in self.v:
            if 0 < j < self.ny:
                if self.is_solid(i, j - 1) or self.is_solid(i, j):
                    self.v[i, j] = self.v_solid[i, j]
                elif self.is_fluid(i, j - 1) or self.is_fluid(i, j):
                    pt = self.pressure[i, j] if self.is_fluid(i, j) else 0.0
                    pb = self.pressure[i, j - 1] if self.is_fluid(i, j - 1) else 0.0
                    self.v[i, j] -= coef * (pt - pb)

    # ===================================================================== #
    # 7. 速度外插（把流体速度推到空气区域）
    # ===================================================================== #
    @ti.kernel
    def mark_valid(self):
        for i, j in self.u:
            valid = 0
            if 0 < i < self.nx:
                if self.is_fluid(i - 1, j) or self.is_fluid(i, j):
                    if not (self.is_solid(i - 1, j) or self.is_solid(i, j)):
                        valid = 1
            self.valid_u[i, j] = valid
        for i, j in self.v:
            valid = 0
            if 0 < j < self.ny:
                if self.is_fluid(i, j - 1) or self.is_fluid(i, j):
                    if not (self.is_solid(i, j - 1) or self.is_solid(i, j)):
                        valid = 1
            self.valid_v[i, j] = valid

    @ti.kernel
    def extrapolate_once(self):
        # u
        for I in ti.grouped(self.u):
            self.u_tmp[I] = self.u[I]
            self.valid_u_tmp[I] = self.valid_u[I]
        for i, j in self.u:
            if self.valid_u[i, j] == 0:
                s = 0.0
                c = 0
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    if (di != 0 or dj != 0):
                        ni, nj = i + di, j + dj
                        if 0 <= ni <= self.nx and 0 <= nj < self.ny:
                            if self.valid_u[ni, nj] == 1:
                                s += self.u[ni, nj]
                                c += 1
                if c > 0:
                    self.u_tmp[i, j] = s / c
                    self.valid_u_tmp[i, j] = 1
        for I in ti.grouped(self.u):
            self.u[I] = self.u_tmp[I]
            self.valid_u[I] = self.valid_u_tmp[I]
        # v
        for I in ti.grouped(self.v):
            self.v_tmp[I] = self.v[I]
            self.valid_v_tmp[I] = self.valid_v[I]
        for i, j in self.v:
            if self.valid_v[i, j] == 0:
                s = 0.0
                c = 0
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    if (di != 0 or dj != 0):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.nx and 0 <= nj <= self.ny:
                            if self.valid_v[ni, nj] == 1:
                                s += self.v[ni, nj]
                                c += 1
                if c > 0:
                    self.v_tmp[i, j] = s / c
                    self.valid_v_tmp[i, j] = 1
        for I in ti.grouped(self.v):
            self.v[I] = self.v_tmp[I]
            self.valid_v[I] = self.valid_v_tmp[I]

    def extrapolate(self):
        self.mark_valid()
        for _ in range(self.cfg.extrapolate_iters):
            self.extrapolate_once()

    # ===================================================================== #
    # 8. G2P —— 网格速度回插粒子
    # ===================================================================== #
    @ti.kernel
    def g2p(self, mode: ti.i32, flip_ratio: ti.f32):
        for p in range(self.n_particles[None]):
            xp = self.px[p]
            pic = ti.Vector([0.0, 0.0])
            flip = self.pv[p]
            Cnew = ti.Matrix.zero(ti.f32, 2, 2)

            # ---- u 网格 ----
            gp = ti.Vector([xp[0] * self.inv_dx, xp[1] * self.inv_dx - 0.5])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            fx = gp - base.cast(ti.f32)
            wx = ti.Vector([0.5 * (1.5 - fx[0]) ** 2, 0.75 - (fx[0] - 1.0) ** 2, 0.5 * (fx[0] - 0.5) ** 2])
            wy = ti.Vector([0.5 * (1.5 - fx[1]) ** 2, 0.75 - (fx[1] - 1.0) ** 2, 0.5 * (fx[1] - 0.5) ** 2])
            up = 0.0
            udelta = 0.0
            for a, b in ti.static(ti.ndrange(3, 3)):
                ni, nj = base[0] + a, base[1] + b
                if 0 <= ni <= self.nx and 0 <= nj < self.ny:
                    w = wx[a] * wy[b]
                    node_pos = ti.Vector([ni * self.dx, (nj + 0.5) * self.dx])
                    uval = self.u[ni, nj]
                    up += w * uval
                    udelta += w * (uval - self.u_old[ni, nj])
                    Cnew[0, 0] += 4.0 * self.inv_dx * w * uval * (node_pos[0] - xp[0])
                    Cnew[0, 1] += 4.0 * self.inv_dx * w * uval * (node_pos[1] - xp[1])
            pic[0] = up
            flip[0] += udelta

            # ---- v 网格 ----
            gp = ti.Vector([xp[0] * self.inv_dx - 0.5, xp[1] * self.inv_dx])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            fx = gp - base.cast(ti.f32)
            wx = ti.Vector([0.5 * (1.5 - fx[0]) ** 2, 0.75 - (fx[0] - 1.0) ** 2, 0.5 * (fx[0] - 0.5) ** 2])
            wy = ti.Vector([0.5 * (1.5 - fx[1]) ** 2, 0.75 - (fx[1] - 1.0) ** 2, 0.5 * (fx[1] - 0.5) ** 2])
            vp = 0.0
            vdelta = 0.0
            for a, b in ti.static(ti.ndrange(3, 3)):
                ni, nj = base[0] + a, base[1] + b
                if 0 <= ni < self.nx and 0 <= nj <= self.ny:
                    w = wx[a] * wy[b]
                    node_pos = ti.Vector([(ni + 0.5) * self.dx, nj * self.dx])
                    vval = self.v[ni, nj]
                    vp += w * vval
                    vdelta += w * (vval - self.v_old[ni, nj])
                    Cnew[1, 0] += 4.0 * self.inv_dx * w * vval * (node_pos[0] - xp[0])
                    Cnew[1, 1] += 4.0 * self.inv_dx * w * vval * (node_pos[1] - xp[1])
            pic[1] = vp
            flip[1] += vdelta

            if mode == TransferMode.PIC.value:
                self.pv[p] = pic
                self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            elif mode == TransferMode.FLIP.value:
                self.pv[p] = flip_ratio * flip + (1.0 - flip_ratio) * pic
                self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            else:  # APIC
                self.pv[p] = pic
                self.C[p] = Cnew

    # ===================================================================== #
    # 9. 粒子平流 + 边界投影
    # ===================================================================== #
    @ti.kernel
    def advect_particles(self, sdt: ti.f32, vclamp: ti.f32, damping: ti.f32):
        eps = 1e-4 * self.dx
        lx = self.nx * self.dx
        ly = self.ny * self.dx
        for p in range(self.n_particles[None]):
            if damping > 0.0:
                self.pv[p] *= ti.exp(-damping * sdt)
            x = self.px[p]
            # RK2 中点法
            k1 = self.sample_velocity(x)
            xmid = x + 0.5 * sdt * k1
            k2 = self.sample_velocity(xmid)
            x = x + sdt * k2

            # 限制速度防爆
            sp = self.pv[p].norm()
            if sp > vclamp:
                self.pv[p] *= vclamp / sp

            # 域边界裁剪
            x[0] = ti.min(ti.max(x[0], eps), lx - eps)
            x[1] = ti.min(ti.max(x[1], eps), ly - eps)

            # 障碍投影：若落入固体内，沿 phi 梯度推出
            phi = self.sample_phi(x)
            if phi < 0.0:
                n = self.phi_gradient(x)
                x += (-phi + eps) * n
                # 去掉指向固体内的法向速度分量
                vn = self.pv[p].dot(n)
                if vn < 0.0:
                    self.pv[p] -= vn * n
                x[0] = ti.min(ti.max(x[0], eps), lx - eps)
                x[1] = ti.min(ti.max(x[1], eps), ly - eps)

            self.px[p] = x

    # ===================================================================== #
    # 发射 / 压缩（喷口、吸入口、出界清理）
    # ===================================================================== #
    @ti.kernel
    def emit_block(self, x0: ti.f32, y0: ti.f32, x1: ti.f32, y1: ti.f32,
                   vx: ti.f32, vy: ti.f32, count: ti.i32):
        for k in range(count):
            idx = ti.atomic_add(self.n_particles[None], 1)
            if idx < self.max_particles:
                rx = ti.random()
                ry = ti.random()
                self.px[idx] = ti.Vector([x0 + rx * (x1 - x0), y0 + ry * (y1 - y0)])
                self.pv[idx] = ti.Vector([vx, vy])
                self.C[idx] = ti.Matrix.zero(ti.f32, 2, 2)
            else:
                self.n_particles[None] = self.max_particles

    @ti.kernel
    def apply_drag_force(self, cx: ti.f32, cy: ti.f32, vx: ti.f32, vy: ti.f32,
                         radius: ti.f32, strength: ti.f32):
        """对光标半径内的粒子施加朝鼠标运动方向的拖拽速度（交互搅动）。"""
        r2 = radius * radius
        for p in range(self.n_particles[None]):
            d = self.px[p] - ti.Vector([cx, cy])
            dd = d.dot(d)
            if dd < r2:
                w = 1.0 - ti.sqrt(dd) / radius
                self.pv[p] += strength * w * ti.Vector([vx, vy])

    @ti.kernel
    def _compact(self, sink_x0: ti.f32, sink_y0: ti.f32, sink_x1: ti.f32, sink_y1: ti.f32,
                 use_sink: ti.i32):
        self.n_tmp[None] = 0
        lx = self.nx * self.dx
        ly = self.ny * self.dx
        for p in range(self.n_particles[None]):
            x = self.px[p]
            alive = 1
            if x[0] < 0 or x[0] > lx or x[1] < 0 or x[1] > ly:
                alive = 0
            if use_sink == 1 and (sink_x0 <= x[0] <= sink_x1 and sink_y0 <= x[1] <= sink_y1):
                alive = 0
            if alive == 1:
                idx = ti.atomic_add(self.n_tmp[None], 1)
                self.px2[idx] = self.px[p]
                self.pv2[idx] = self.pv[p]
                self.C2[idx] = self.C[p]

    @ti.kernel
    def _copy_back(self, n: ti.i32):
        for p in range(n):
            self.px[p] = self.px2[p]
            self.pv[p] = self.pv2[p]
            self.C[p] = self.C2[p]

    def compact(self, sink=None):
        # 把存活粒子压缩到 px2，再拷回 px（不交换字段引用，避免破坏已编译内核）
        if sink is None:
            self._compact(0.0, 0.0, 0.0, 0.0, 0)
        else:
            self._compact(sink[0], sink[1], sink[2], sink[3], 1)
        n_alive = self.n_tmp[None]
        self._copy_back(n_alive)
        self.n_particles[None] = n_alive

    # ===================================================================== #
    # 诊断
    # ===================================================================== #
    @ti.kernel
    def compute_max_velocity(self) -> ti.f32:
        m = 0.0
        for p in range(self.n_particles[None]):
            ti.atomic_max(m, self.pv[p].norm())
        return m

    @ti.kernel
    def compute_max_divergence(self) -> ti.f32:
        m = 0.0
        for i, j in self.divergence:
            if self.cell_type[i, j] == FLUID:
                ti.atomic_max(m, ti.abs(self.divergence[i, j]))
        return m

    # ===================================================================== #
    # 主推进
    # ===================================================================== #
    def substep(self, sdt: float):
        cfg = self.cfg
        apic = 1 if cfg.transfer == TransferMode.APIC else 0
        self.p2g(apic)
        self.normalize_grid()
        self.classify_cells()
        self.add_gravity(sdt, cfg.gravity_x, cfg.gravity_y)
        self.apply_solid_boundaries()
        self.compute_divergence()
        self.solve_pressure(sdt)
        self.apply_pressure(sdt)
        self.apply_solid_boundaries()
        self.extrapolate()
        self.g2p(int(cfg.transfer.value), cfg.flip_ratio)
        vclamp = cfg.grid_velocity_clamp() * 4.0
        self.advect_particles(sdt, vclamp, cfg.vel_damping)

    def step(self):
        """推进一帧（cfg.dt），内部分若干子步。"""
        cfg = self.cfg
        n_sub = cfg.substeps
        if cfg.adaptive_substeps and self.n_particles[None] > 0:
            vmax = self.compute_max_velocity()
            if vmax > 1e-6:
                need = int(np.ceil(cfg.dt * vmax / (cfg.cfl * self.dx)))
                n_sub = int(min(max(cfg.substeps, need), 20))
        sdt = cfg.dt / n_sub
        for _ in range(n_sub):
            self.substep(sdt)
        return n_sub

    # 便捷访问
    def particle_positions_np(self):
        n = self.n_particles[None]
        return self.px.to_numpy()[:n]

    def particle_velocities_np(self):
        n = self.n_particles[None]
        return self.pv.to_numpy()[:n]
