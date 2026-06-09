"""三维 FLIP / PIC / APIC 流体求解器（交错 MAC 网格 + 粒子）。

算法总览（每个子步）：
    1. P2G   : 粒子速度（及 APIC 仿射项）散布到交错网格速度面上
    2. 归一化 : 网格速度 = 加权平均；保存投影前速度供 FLIP 使用
    3. 标记单元 : 由固体场 + 粒子占据情况标记 SOLID / FLUID / AIR
    4. 体力   : 重力加到网格速度
    5. 边界   : 固体面速度置为固体速度（静态即 0）
   5b. 粘性   : 显式 Jacobi 速度扩散（可选）
    6. 投影   : 求解压力 Poisson，使速度近似无散（不可压缩）
    7. 外插   : 把流体速度外插到空气区域，减少自由表面伪影
    8. G2P    : 网格速度（或增量）插值回粒子，更新 APIC 矩阵 C（3x3）
    9. 平流   : RK2 用网格速度场推进粒子，并做边界 / 障碍投影

坐标约定：世界坐标 [0,Lx]x[0,Ly]x[0,Lz]，立方体单元 dx；
    u (x 速度) 存于 x 面，形状 (nx+1, ny, nz)，位置 (i·dx, (j+0.5)·dx, (k+0.5)·dx)
    v (y 速度) 存于 y 面，形状 (nx, ny+1, nz)，位置 ((i+0.5)·dx, j·dx, (k+0.5)·dx)
    w (z 速度) 存于 z 面，形状 (nx, ny, nz+1)，位置 ((i+0.5)·dx, (j+0.5)·dx, k·dx)
    压力 / 单元类型 / 固体距离存于单元中心，形状 (nx, ny, nz)
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


@ti.func
def quad_w(f):
    """二次 B 样条权重（base=floor(gp-0.5)，f=gp-base ∈ [0.5,1.5)）。"""
    return ti.Vector([0.5 * (1.5 - f) ** 2,
                      0.75 - (f - 1.0) ** 2,
                      0.5 * (f - 0.5) ** 2])


@ti.data_oriented
class FlipSolver:
    def __init__(self, cfg: FluidConfig):
        self.cfg = cfg
        nx, ny, nz = cfg.res_x, cfg.res_y, cfg.res_z
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = cfg.dx
        self.inv_dx = 1.0 / cfg.dx
        self.rho = cfg.rho

        # --- MAC 网格速度场 ---
        self.u = ti.field(ti.f32, shape=(nx + 1, ny, nz))
        self.v = ti.field(ti.f32, shape=(nx, ny + 1, nz))
        self.w = ti.field(ti.f32, shape=(nx, ny, nz + 1))
        self.u_old = ti.field(ti.f32, shape=(nx + 1, ny, nz))
        self.v_old = ti.field(ti.f32, shape=(nx, ny + 1, nz))
        self.w_old = ti.field(ti.f32, shape=(nx, ny, nz + 1))
        self.u_w = ti.field(ti.f32, shape=(nx + 1, ny, nz))
        self.v_w = ti.field(ti.f32, shape=(nx, ny + 1, nz))
        self.w_w = ti.field(ti.f32, shape=(nx, ny, nz + 1))

        # 外插 / 粘性临时场 + 有效标记
        self.u_tmp = ti.field(ti.f32, shape=(nx + 1, ny, nz))
        self.v_tmp = ti.field(ti.f32, shape=(nx, ny + 1, nz))
        self.w_tmp = ti.field(ti.f32, shape=(nx, ny, nz + 1))
        self.valid_u = ti.field(ti.i32, shape=(nx + 1, ny, nz))
        self.valid_v = ti.field(ti.i32, shape=(nx, ny + 1, nz))
        self.valid_w = ti.field(ti.i32, shape=(nx, ny, nz + 1))
        self.valid_u_tmp = ti.field(ti.i32, shape=(nx + 1, ny, nz))
        self.valid_v_tmp = ti.field(ti.i32, shape=(nx, ny + 1, nz))
        self.valid_w_tmp = ti.field(ti.i32, shape=(nx, ny, nz + 1))

        # --- 单元中心量 ---
        self.pressure = ti.field(ti.f32, shape=(nx, ny, nz))
        self.divergence = ti.field(ti.f32, shape=(nx, ny, nz))
        self.cell_type = ti.field(ti.i32, shape=(nx, ny, nz))

        # 压力 PCG 工作场
        self.cg_r = ti.field(ti.f32, shape=(nx, ny, nz))
        self.cg_d = ti.field(ti.f32, shape=(nx, ny, nz))
        self.cg_q = ti.field(ti.f32, shape=(nx, ny, nz))
        self.cg_z = ti.field(ti.f32, shape=(nx, ny, nz))
        self.cg_diag = ti.field(ti.f32, shape=(nx, ny, nz))

        # --- 固体边界（耦合接口） ---
        self.solid_phi = ti.field(ti.f32, shape=(nx, ny, nz))   # < 0 在固体内
        self.u_solid = ti.field(ti.f32, shape=(nx + 1, ny, nz))
        self.v_solid = ti.field(ti.f32, shape=(nx, ny + 1, nz))
        self.w_solid = ti.field(ti.f32, shape=(nx, ny, nz + 1))

        # --- 粒子 ---
        self.max_particles = cfg.max_particles
        self.px = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.pv = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.C = ti.Matrix.field(3, 3, ti.f32, shape=self.max_particles)
        self.px2 = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.pv2 = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.C2 = ti.Matrix.field(3, 3, ti.f32, shape=self.max_particles)
        self.n_particles = ti.field(ti.i32, shape=())
        self.n_tmp = ti.field(ti.i32, shape=())

        # 诊断
        self.max_vel = ti.field(ti.f32, shape=())
        self.max_div = ti.field(ti.f32, shape=())

        self.solid_phi.fill(1e9)
        self.n_particles[None] = 0

    # ===================================================================== #
    # 初始化 / 固体场
    # ===================================================================== #
    def set_solid_phi(self, phi_np: np.ndarray):
        assert phi_np.shape == (self.nx, self.ny, self.nz)
        self.solid_phi.from_numpy(phi_np.astype(np.float32))
        self.u_solid.fill(0.0)
        self.v_solid.fill(0.0)
        self.w_solid.fill(0.0)

    def add_particle_block(self, x0, y0, z0, x1, y1, z1, jitter=True):
        """在世界坐标长方体 [x0,x1]x[y0,y1]x[z0,z1] 内按 particles_per_cell 填充粒子。"""
        cfg = self.cfg
        dx = self.dx
        ppc = cfg.particles_per_cell
        sub = max(1, int(round(ppc ** (1.0 / 3.0))))
        i0, i1 = int(np.floor(x0 / dx)), int(np.ceil(x1 / dx))
        j0, j1 = int(np.floor(y0 / dx)), int(np.ceil(y1 / dx))
        k0, k1 = int(np.floor(z0 / dx)), int(np.ceil(z1 / dx))
        rng = np.random.default_rng(cfg.seed + self.n_particles[None])
        phi = self.solid_phi.to_numpy()
        # 向量化生成候选点
        cells_i = np.arange(max(i0, 0), min(i1, self.nx))
        cells_j = np.arange(max(j0, 0), min(j1, self.ny))
        cells_k = np.arange(max(k0, 0), min(k1, self.nz))
        pts = []
        for i in cells_i:
            for j in cells_j:
                for k in cells_k:
                    if jitter:
                        ox = (np.arange(sub) + rng.random(sub)) / sub
                        oy = (np.arange(sub) + rng.random(sub)) / sub
                        oz = (np.arange(sub) + rng.random(sub)) / sub
                    else:
                        ox = oy = oz = (np.arange(sub) + 0.5) / sub
                    XX = (i + ox)[:, None, None] * dx
                    YY = (j + oy)[None, :, None] * dx
                    ZZ = (k + oz)[None, None, :] * dx
                    XX, YY, ZZ = np.broadcast_arrays(XX, YY, ZZ)
                    P = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
                    m = ((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
                         & (P[:, 2] >= z0) & (P[:, 2] <= z1))
                    P = P[m]
                    if len(P) == 0:
                        continue
                    ci = np.clip((P[:, 0] / dx).astype(int), 0, self.nx - 1)
                    cj = np.clip((P[:, 1] / dx).astype(int), 0, self.ny - 1)
                    ck = np.clip((P[:, 2] / dx).astype(int), 0, self.nz - 1)
                    P = P[phi[ci, cj, ck] >= 0]
                    if len(P):
                        pts.append(P)
        if not pts:
            return 0
        pts = np.concatenate(pts, axis=0).astype(np.float32)
        n0 = self.n_particles[None]
        n_add = min(len(pts), self.max_particles - n0)
        if n_add <= 0:
            return 0
        self._upload_particles(pts[:n_add], n0)
        self.n_particles[None] = n0 + n_add
        return n_add

    def _upload_particles(self, pts: np.ndarray, offset: int):
        n = len(pts)
        full = self.px.to_numpy()
        full[offset:offset + n] = pts
        self.px.from_numpy(full)
        self._zero_particle_state(offset, offset + n)

    @ti.kernel
    def _zero_particle_state(self, lo: ti.i32, hi: ti.i32):
        for p in range(lo, hi):
            self.pv[p] = ti.Vector([0.0, 0.0, 0.0])
            self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)

    # ===================================================================== #
    # 网格辅助
    # ===================================================================== #
    @ti.func
    def is_solid(self, i, j, k) -> ti.i32:
        res = 0
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz:
            res = 1
        elif self.cell_type[i, j, k] == SOLID:
            res = 1
        return res

    @ti.func
    def is_fluid(self, i, j, k) -> ti.i32:
        res = 0
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            if self.cell_type[i, j, k] == FLUID:
                res = 1
        return res

    @ti.func
    def _trilerp(self, c000, c100, c010, c110, c001, c101, c011, c111, fx, fy, fz):
        return (c000 * (1 - fx) * (1 - fy) * (1 - fz)
                + c100 * fx * (1 - fy) * (1 - fz)
                + c010 * (1 - fx) * fy * (1 - fz)
                + c110 * fx * fy * (1 - fz)
                + c001 * (1 - fx) * (1 - fy) * fz
                + c101 * fx * (1 - fy) * fz
                + c011 * (1 - fx) * fy * fz
                + c111 * fx * fy * fz)

    @ti.func
    def sample_u(self, x):
        gx = ti.min(ti.max(x[0] * self.inv_dx, 0.0), float(self.nx))
        gy = ti.min(ti.max(x[1] * self.inv_dx - 0.5, 0.0), float(self.ny - 1))
        gz = ti.min(ti.max(x[2] * self.inv_dx - 0.5, 0.0), float(self.nz - 1))
        i = ti.min(int(gx), self.nx - 1)
        j = ti.min(int(gy), self.ny - 2)
        k = ti.min(int(gz), self.nz - 2)
        fx, fy, fz = gx - i, gy - j, gz - k
        return self._trilerp(self.u[i, j, k], self.u[i + 1, j, k], self.u[i, j + 1, k],
                             self.u[i + 1, j + 1, k], self.u[i, j, k + 1], self.u[i + 1, j, k + 1],
                             self.u[i, j + 1, k + 1], self.u[i + 1, j + 1, k + 1], fx, fy, fz)

    @ti.func
    def sample_v(self, x):
        gx = ti.min(ti.max(x[0] * self.inv_dx - 0.5, 0.0), float(self.nx - 1))
        gy = ti.min(ti.max(x[1] * self.inv_dx, 0.0), float(self.ny))
        gz = ti.min(ti.max(x[2] * self.inv_dx - 0.5, 0.0), float(self.nz - 1))
        i = ti.min(int(gx), self.nx - 2)
        j = ti.min(int(gy), self.ny - 1)
        k = ti.min(int(gz), self.nz - 2)
        fx, fy, fz = gx - i, gy - j, gz - k
        return self._trilerp(self.v[i, j, k], self.v[i + 1, j, k], self.v[i, j + 1, k],
                             self.v[i + 1, j + 1, k], self.v[i, j, k + 1], self.v[i + 1, j, k + 1],
                             self.v[i, j + 1, k + 1], self.v[i + 1, j + 1, k + 1], fx, fy, fz)

    @ti.func
    def sample_w(self, x):
        gx = ti.min(ti.max(x[0] * self.inv_dx - 0.5, 0.0), float(self.nx - 1))
        gy = ti.min(ti.max(x[1] * self.inv_dx - 0.5, 0.0), float(self.ny - 1))
        gz = ti.min(ti.max(x[2] * self.inv_dx, 0.0), float(self.nz))
        i = ti.min(int(gx), self.nx - 2)
        j = ti.min(int(gy), self.ny - 2)
        k = ti.min(int(gz), self.nz - 1)
        fx, fy, fz = gx - i, gy - j, gz - k
        return self._trilerp(self.w[i, j, k], self.w[i + 1, j, k], self.w[i, j + 1, k],
                             self.w[i + 1, j + 1, k], self.w[i, j, k + 1], self.w[i + 1, j, k + 1],
                             self.w[i, j + 1, k + 1], self.w[i + 1, j + 1, k + 1], fx, fy, fz)

    @ti.func
    def sample_velocity(self, x):
        return ti.Vector([self.sample_u(x), self.sample_v(x), self.sample_w(x)])

    @ti.func
    def sample_phi(self, x):
        gx = ti.min(ti.max(x[0] * self.inv_dx - 0.5, 0.0), float(self.nx - 1))
        gy = ti.min(ti.max(x[1] * self.inv_dx - 0.5, 0.0), float(self.ny - 1))
        gz = ti.min(ti.max(x[2] * self.inv_dx - 0.5, 0.0), float(self.nz - 1))
        i = ti.min(int(gx), self.nx - 2)
        j = ti.min(int(gy), self.ny - 2)
        k = ti.min(int(gz), self.nz - 2)
        fx, fy, fz = gx - i, gy - j, gz - k
        return self._trilerp(self.solid_phi[i, j, k], self.solid_phi[i + 1, j, k],
                             self.solid_phi[i, j + 1, k], self.solid_phi[i + 1, j + 1, k],
                             self.solid_phi[i, j, k + 1], self.solid_phi[i + 1, j, k + 1],
                             self.solid_phi[i, j + 1, k + 1], self.solid_phi[i + 1, j + 1, k + 1],
                             fx, fy, fz)

    @ti.func
    def phi_gradient(self, x):
        h = 0.5 * self.dx
        gx = self.sample_phi(x + ti.Vector([h, 0.0, 0.0])) - self.sample_phi(x - ti.Vector([h, 0.0, 0.0]))
        gy = self.sample_phi(x + ti.Vector([0.0, h, 0.0])) - self.sample_phi(x - ti.Vector([0.0, h, 0.0]))
        gz = self.sample_phi(x + ti.Vector([0.0, 0.0, h])) - self.sample_phi(x - ti.Vector([0.0, 0.0, h]))
        g = ti.Vector([gx, gy, gz])
        n = g.norm()
        if n > 1e-8:
            g = g / n
        else:
            g = ti.Vector([0.0, 1.0, 0.0])
        return g

    # ===================================================================== #
    # 1. P2G
    # ===================================================================== #
    @ti.kernel
    def p2g(self, apic: ti.i32):
        for I in ti.grouped(self.u):
            self.u[I] = 0.0
            self.u_w[I] = 0.0
        for I in ti.grouped(self.v):
            self.v[I] = 0.0
            self.v_w[I] = 0.0
        for I in ti.grouped(self.w):
            self.w[I] = 0.0
            self.w_w[I] = 0.0

        for p in range(self.n_particles[None]):
            xp = self.px[p]
            vp = self.pv[p]
            Cp = self.C[p]

            # ---- u 网格 (component 0) ----
            gp = ti.Vector([xp[0] * self.inv_dx, xp[1] * self.inv_dx - 0.5, xp[2] * self.inv_dx - 0.5])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            f = gp - base.cast(ti.f32)
            wx, wy, wz = quad_w(f[0]), quad_w(f[1]), quad_w(f[2])
            for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
                ni, nj, nk = base[0] + a, base[1] + b, base[2] + c
                if 0 <= ni <= self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                    weight = wx[a] * wy[b] * wz[c]
                    npos = ti.Vector([ni * self.dx, (nj + 0.5) * self.dx, (nk + 0.5) * self.dx])
                    val = vp[0]
                    if apic == 1:
                        val += Cp[0, 0] * (npos[0] - xp[0]) + Cp[0, 1] * (npos[1] - xp[1]) + Cp[0, 2] * (npos[2] - xp[2])
                    self.u[ni, nj, nk] += weight * val
                    self.u_w[ni, nj, nk] += weight

            # ---- v 网格 (component 1) ----
            gp = ti.Vector([xp[0] * self.inv_dx - 0.5, xp[1] * self.inv_dx, xp[2] * self.inv_dx - 0.5])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            f = gp - base.cast(ti.f32)
            wx, wy, wz = quad_w(f[0]), quad_w(f[1]), quad_w(f[2])
            for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
                ni, nj, nk = base[0] + a, base[1] + b, base[2] + c
                if 0 <= ni < self.nx and 0 <= nj <= self.ny and 0 <= nk < self.nz:
                    weight = wx[a] * wy[b] * wz[c]
                    npos = ti.Vector([(ni + 0.5) * self.dx, nj * self.dx, (nk + 0.5) * self.dx])
                    val = vp[1]
                    if apic == 1:
                        val += Cp[1, 0] * (npos[0] - xp[0]) + Cp[1, 1] * (npos[1] - xp[1]) + Cp[1, 2] * (npos[2] - xp[2])
                    self.v[ni, nj, nk] += weight * val
                    self.v_w[ni, nj, nk] += weight

            # ---- w 网格 (component 2) ----
            gp = ti.Vector([xp[0] * self.inv_dx - 0.5, xp[1] * self.inv_dx - 0.5, xp[2] * self.inv_dx])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            f = gp - base.cast(ti.f32)
            wx, wy, wz = quad_w(f[0]), quad_w(f[1]), quad_w(f[2])
            for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
                ni, nj, nk = base[0] + a, base[1] + b, base[2] + c
                if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk <= self.nz:
                    weight = wx[a] * wy[b] * wz[c]
                    npos = ti.Vector([(ni + 0.5) * self.dx, (nj + 0.5) * self.dx, nk * self.dx])
                    val = vp[2]
                    if apic == 1:
                        val += Cp[2, 0] * (npos[0] - xp[0]) + Cp[2, 1] * (npos[1] - xp[1]) + Cp[2, 2] * (npos[2] - xp[2])
                    self.w[ni, nj, nk] += weight * val
                    self.w_w[ni, nj, nk] += weight

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
        for I in ti.grouped(self.w):
            if self.w_w[I] > 1e-8:
                self.w[I] /= self.w_w[I]
            else:
                self.w[I] = 0.0
            self.w_old[I] = self.w[I]

    # ===================================================================== #
    # 3. 单元分类
    # ===================================================================== #
    @ti.kernel
    def classify_cells(self):
        for I in ti.grouped(self.cell_type):
            if self.solid_phi[I] < 0.0:
                self.cell_type[I] = SOLID
            else:
                self.cell_type[I] = AIR
        for p in range(self.n_particles[None]):
            xp = self.px[p]
            ci = int(xp[0] * self.inv_dx)
            cj = int(xp[1] * self.inv_dx)
            ck = int(xp[2] * self.inv_dx)
            if 0 <= ci < self.nx and 0 <= cj < self.ny and 0 <= ck < self.nz:
                if self.cell_type[ci, cj, ck] != SOLID:
                    self.cell_type[ci, cj, ck] = FLUID

    # ===================================================================== #
    # 4 & 5. 体力 + 固体边界
    # ===================================================================== #
    @ti.kernel
    def add_gravity(self, sdt: ti.f32, gx: ti.f32, gy: ti.f32, gz: ti.f32):
        for I in ti.grouped(self.u):
            if self.u_w[I] > 1e-8:
                self.u[I] += gx * sdt
        for I in ti.grouped(self.v):
            if self.v_w[I] > 1e-8:
                self.v[I] += gy * sdt
        for I in ti.grouped(self.w):
            if self.w_w[I] > 1e-8:
                self.w[I] += gz * sdt

    @ti.kernel
    def apply_solid_boundaries(self):
        for i, j, k in self.u:
            if i == 0 or i == self.nx or self.is_solid(i - 1, j, k) or self.is_solid(i, j, k):
                self.u[i, j, k] = self.u_solid[i, j, k]
        for i, j, k in self.v:
            if j == 0 or j == self.ny or self.is_solid(i, j - 1, k) or self.is_solid(i, j, k):
                self.v[i, j, k] = self.v_solid[i, j, k]
        for i, j, k in self.w:
            if k == 0 or k == self.nz or self.is_solid(i, j, k - 1) or self.is_solid(i, j, k):
                self.w[i, j, k] = self.w_solid[i, j, k]

    # ===================================================================== #
    # 5b. 粘性扩散（显式 Jacobi，按稳定性自动细分）
    # ===================================================================== #
    @ti.kernel
    def _diffuse_u(self, coef: ti.f32):
        for I in ti.grouped(self.u):
            self.u_tmp[I] = self.u[I]
        for i, j, k in self.u:
            active = 0
            if 0 < i < self.nx:
                if not (self.is_solid(i - 1, j, k) or self.is_solid(i, j, k)):
                    if self.is_fluid(i - 1, j, k) or self.is_fluid(i, j, k):
                        active = 1
            if active == 1:
                il, ir = max(i - 1, 0), min(i + 1, self.nx)
                jd, ju = max(j - 1, 0), min(j + 1, self.ny - 1)
                kd, ku = max(k - 1, 0), min(k + 1, self.nz - 1)
                lap = (self.u[il, j, k] + self.u[ir, j, k] + self.u[i, jd, k] + self.u[i, ju, k]
                       + self.u[i, j, kd] + self.u[i, j, ku] - 6.0 * self.u[i, j, k])
                self.u_tmp[i, j, k] = self.u[i, j, k] + coef * lap
        for I in ti.grouped(self.u):
            self.u[I] = self.u_tmp[I]

    @ti.kernel
    def _diffuse_v(self, coef: ti.f32):
        for I in ti.grouped(self.v):
            self.v_tmp[I] = self.v[I]
        for i, j, k in self.v:
            active = 0
            if 0 < j < self.ny:
                if not (self.is_solid(i, j - 1, k) or self.is_solid(i, j, k)):
                    if self.is_fluid(i, j - 1, k) or self.is_fluid(i, j, k):
                        active = 1
            if active == 1:
                il, ir = max(i - 1, 0), min(i + 1, self.nx - 1)
                jd, ju = max(j - 1, 0), min(j + 1, self.ny)
                kd, ku = max(k - 1, 0), min(k + 1, self.nz - 1)
                lap = (self.v[il, j, k] + self.v[ir, j, k] + self.v[i, jd, k] + self.v[i, ju, k]
                       + self.v[i, j, kd] + self.v[i, j, ku] - 6.0 * self.v[i, j, k])
                self.v_tmp[i, j, k] = self.v[i, j, k] + coef * lap
        for I in ti.grouped(self.v):
            self.v[I] = self.v_tmp[I]

    @ti.kernel
    def _diffuse_w(self, coef: ti.f32):
        for I in ti.grouped(self.w):
            self.w_tmp[I] = self.w[I]
        for i, j, k in self.w:
            active = 0
            if 0 < k < self.nz:
                if not (self.is_solid(i, j, k - 1) or self.is_solid(i, j, k)):
                    if self.is_fluid(i, j, k - 1) or self.is_fluid(i, j, k):
                        active = 1
            if active == 1:
                il, ir = max(i - 1, 0), min(i + 1, self.nx - 1)
                jd, ju = max(j - 1, 0), min(j + 1, self.ny - 1)
                kd, ku = max(k - 1, 0), min(k + 1, self.nz)
                lap = (self.w[il, j, k] + self.w[ir, j, k] + self.w[i, jd, k] + self.w[i, ju, k]
                       + self.w[i, j, kd] + self.w[i, j, ku] - 6.0 * self.w[i, j, k])
                self.w_tmp[i, j, k] = self.w[i, j, k] + coef * lap
        for I in ti.grouped(self.w):
            self.w[I] = self.w_tmp[I]

    def apply_viscosity(self, sdt: float):
        nu = self.cfg.viscosity
        if nu <= 0.0:
            return
        max_coef = 0.16  # 3D 显式扩散稳定性更紧
        m = max(1, int(np.ceil(nu * sdt / (max_coef * self.dx * self.dx))))
        coef = nu * sdt / (m * self.dx * self.dx)
        for _ in range(m):
            self._diffuse_u(coef)
            self._diffuse_v(coef)
            self._diffuse_w(coef)

    # ===================================================================== #
    # 6. 压力投影
    # ===================================================================== #
    @ti.kernel
    def compute_divergence(self):
        for i, j, k in self.divergence:
            if self.cell_type[i, j, k] == FLUID:
                self.divergence[i, j, k] = (self.u[i + 1, j, k] - self.u[i, j, k]
                                            + self.v[i, j + 1, k] - self.v[i, j, k]
                                            + self.w[i, j, k + 1] - self.w[i, j, k]) * self.inv_dx
            else:
                self.divergence[i, j, k] = 0.0

    @ti.func
    def _nonsolid_count(self, i, j, k):
        c = 0
        if not self.is_solid(i - 1, j, k):
            c += 1
        if not self.is_solid(i + 1, j, k):
            c += 1
        if not self.is_solid(i, j - 1, k):
            c += 1
        if not self.is_solid(i, j + 1, k):
            c += 1
        if not self.is_solid(i, j, k - 1):
            c += 1
        if not self.is_solid(i, j, k + 1):
            c += 1
        return c

    @ti.kernel
    def pressure_gs_color(self, scale: ti.f32, color: ti.i32):
        for i, j, k in self.pressure:
            if self.cell_type[i, j, k] == FLUID and (i + j + k) % 2 == color:
                nonsolid = self._nonsolid_count(i, j, k)
                sum_p = 0.0
                if self.is_fluid(i - 1, j, k):
                    sum_p += self.pressure[i - 1, j, k]
                if self.is_fluid(i + 1, j, k):
                    sum_p += self.pressure[i + 1, j, k]
                if self.is_fluid(i, j - 1, k):
                    sum_p += self.pressure[i, j - 1, k]
                if self.is_fluid(i, j + 1, k):
                    sum_p += self.pressure[i, j + 1, k]
                if self.is_fluid(i, j, k - 1):
                    sum_p += self.pressure[i, j, k - 1]
                if self.is_fluid(i, j, k + 1):
                    sum_p += self.pressure[i, j, k + 1]
                if nonsolid > 0:
                    self.pressure[i, j, k] = (sum_p - self.divergence[i, j, k] / scale) / nonsolid

    def solve_pressure_gs(self, sdt: float):
        self.pressure.fill(0.0)
        scale = sdt / (self.rho * self.dx * self.dx)
        for _ in range(self.cfg.pressure_iters):
            self.pressure_gs_color(scale, 0)
            self.pressure_gs_color(scale, 1)

    # ---- Jacobi 预条件共轭梯度 (PCG) ----
    @ti.kernel
    def _cg_init(self, scale: ti.f32) -> ti.f32:
        for I in ti.grouped(self.cell_type):
            self.cg_q[I] = 0.0
            if self.cell_type[I] == FLUID:
                nonsolid = self._nonsolid_count(I[0], I[1], I[2])
                self.cg_diag[I] = scale * nonsolid if nonsolid > 0 else 1.0
                self.cg_r[I] = -self.divergence[I]
            else:
                self.cg_diag[I] = 1.0
                self.cg_r[I] = 0.0
        s = 0.0
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] == FLUID:
                z = self.cg_r[I] / self.cg_diag[I]
                self.cg_z[I] = z
                self.cg_d[I] = z
                s += self.cg_r[I] * z
            else:
                self.cg_z[I] = 0.0
                self.cg_d[I] = 0.0
        return s

    @ti.kernel
    def _cg_apply_A_dot(self, scale: ti.f32) -> ti.f32:
        s = 0.0
        for i, j, k in self.cell_type:
            q = 0.0
            if self.cell_type[i, j, k] == FLUID:
                nonsolid = self._nonsolid_count(i, j, k)
                ssum = 0.0
                if self.is_fluid(i - 1, j, k):
                    ssum += self.cg_d[i - 1, j, k]
                if self.is_fluid(i + 1, j, k):
                    ssum += self.cg_d[i + 1, j, k]
                if self.is_fluid(i, j - 1, k):
                    ssum += self.cg_d[i, j - 1, k]
                if self.is_fluid(i, j + 1, k):
                    ssum += self.cg_d[i, j + 1, k]
                if self.is_fluid(i, j, k - 1):
                    ssum += self.cg_d[i, j, k - 1]
                if self.is_fluid(i, j, k + 1):
                    ssum += self.cg_d[i, j, k + 1]
                q = scale * (nonsolid * self.cg_d[i, j, k] - ssum)
                s += self.cg_d[i, j, k] * q
            self.cg_q[i, j, k] = q
        return s

    @ti.kernel
    def _cg_update_precond_dot(self, alpha: ti.f32) -> ti.f32:
        s = 0.0
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] == FLUID:
                self.pressure[I] += alpha * self.cg_d[I]
                r = self.cg_r[I] - alpha * self.cg_q[I]
                self.cg_r[I] = r
                z = r / self.cg_diag[I]
                self.cg_z[I] = z
                s += r * z
        return s

    @ti.kernel
    def _cg_update_d(self, beta: ti.f32):
        for I in ti.grouped(self.cell_type):
            if self.cell_type[I] == FLUID:
                self.cg_d[I] = self.cg_z[I] + beta * self.cg_d[I]

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
        for i, j, k in self.u:
            if 0 < i < self.nx:
                if self.is_solid(i - 1, j, k) or self.is_solid(i, j, k):
                    self.u[i, j, k] = self.u_solid[i, j, k]
                elif self.is_fluid(i - 1, j, k) or self.is_fluid(i, j, k):
                    pr = self.pressure[i, j, k] if self.is_fluid(i, j, k) else 0.0
                    pl = self.pressure[i - 1, j, k] if self.is_fluid(i - 1, j, k) else 0.0
                    self.u[i, j, k] -= coef * (pr - pl)
        for i, j, k in self.v:
            if 0 < j < self.ny:
                if self.is_solid(i, j - 1, k) or self.is_solid(i, j, k):
                    self.v[i, j, k] = self.v_solid[i, j, k]
                elif self.is_fluid(i, j - 1, k) or self.is_fluid(i, j, k):
                    pt = self.pressure[i, j, k] if self.is_fluid(i, j, k) else 0.0
                    pb = self.pressure[i, j - 1, k] if self.is_fluid(i, j - 1, k) else 0.0
                    self.v[i, j, k] -= coef * (pt - pb)
        for i, j, k in self.w:
            if 0 < k < self.nz:
                if self.is_solid(i, j, k - 1) or self.is_solid(i, j, k):
                    self.w[i, j, k] = self.w_solid[i, j, k]
                elif self.is_fluid(i, j, k - 1) or self.is_fluid(i, j, k):
                    pf = self.pressure[i, j, k] if self.is_fluid(i, j, k) else 0.0
                    pbk = self.pressure[i, j, k - 1] if self.is_fluid(i, j, k - 1) else 0.0
                    self.w[i, j, k] -= coef * (pf - pbk)

    # ===================================================================== #
    # 7. 速度外插
    # ===================================================================== #
    @ti.kernel
    def mark_valid(self):
        for i, j, k in self.u:
            valid = 0
            if 0 < i < self.nx:
                if self.is_fluid(i - 1, j, k) or self.is_fluid(i, j, k):
                    if not (self.is_solid(i - 1, j, k) or self.is_solid(i, j, k)):
                        valid = 1
            self.valid_u[i, j, k] = valid
        for i, j, k in self.v:
            valid = 0
            if 0 < j < self.ny:
                if self.is_fluid(i, j - 1, k) or self.is_fluid(i, j, k):
                    if not (self.is_solid(i, j - 1, k) or self.is_solid(i, j, k)):
                        valid = 1
            self.valid_v[i, j, k] = valid
        for i, j, k in self.w:
            valid = 0
            if 0 < k < self.nz:
                if self.is_fluid(i, j, k - 1) or self.is_fluid(i, j, k):
                    if not (self.is_solid(i, j, k - 1) or self.is_solid(i, j, k)):
                        valid = 1
            self.valid_w[i, j, k] = valid

    @ti.kernel
    def _extrap_field(self, fld: ti.template(), fld_tmp: ti.template(),
                      valid: ti.template(), valid_tmp: ti.template(),
                      sx: ti.i32, sy: ti.i32, sz: ti.i32):
        for I in ti.grouped(fld):
            fld_tmp[I] = fld[I]
            valid_tmp[I] = valid[I]
        for i, j, k in fld:
            if valid[i, j, k] == 0:
                s = 0.0
                cnt = 0
                for di, dj, dk in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    if not (di == 0 and dj == 0 and dk == 0):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < sx and 0 <= nj < sy and 0 <= nk < sz:
                            if valid[ni, nj, nk] == 1:
                                s += fld[ni, nj, nk]
                                cnt += 1
                if cnt > 0:
                    fld_tmp[i, j, k] = s / cnt
                    valid_tmp[i, j, k] = 1
        for I in ti.grouped(fld):
            fld[I] = fld_tmp[I]
            valid[I] = valid_tmp[I]

    def extrapolate(self):
        self.mark_valid()
        for _ in range(self.cfg.extrapolate_iters):
            self._extrap_field(self.u, self.u_tmp, self.valid_u, self.valid_u_tmp,
                               self.nx + 1, self.ny, self.nz)
            self._extrap_field(self.v, self.v_tmp, self.valid_v, self.valid_v_tmp,
                               self.nx, self.ny + 1, self.nz)
            self._extrap_field(self.w, self.w_tmp, self.valid_w, self.valid_w_tmp,
                               self.nx, self.ny, self.nz + 1)

    # ===================================================================== #
    # 8. G2P
    # ===================================================================== #
    @ti.kernel
    def g2p(self, mode: ti.i32, flip_ratio: ti.f32):
        for p in range(self.n_particles[None]):
            xp = self.px[p]
            pic = ti.Vector([0.0, 0.0, 0.0])
            flip = self.pv[p]
            Cnew = ti.Matrix.zero(ti.f32, 3, 3)

            # ---- u ----
            gp = ti.Vector([xp[0] * self.inv_dx, xp[1] * self.inv_dx - 0.5, xp[2] * self.inv_dx - 0.5])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            f = gp - base.cast(ti.f32)
            wx, wy, wz = quad_w(f[0]), quad_w(f[1]), quad_w(f[2])
            up, udelta = 0.0, 0.0
            for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
                ni, nj, nk = base[0] + a, base[1] + b, base[2] + c
                if 0 <= ni <= self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                    weight = wx[a] * wy[b] * wz[c]
                    npos = ti.Vector([ni * self.dx, (nj + 0.5) * self.dx, (nk + 0.5) * self.dx])
                    uval = self.u[ni, nj, nk]
                    up += weight * uval
                    udelta += weight * (uval - self.u_old[ni, nj, nk])
                    coef = 4.0 * self.inv_dx * weight * uval
                    Cnew[0, 0] += coef * (npos[0] - xp[0])
                    Cnew[0, 1] += coef * (npos[1] - xp[1])
                    Cnew[0, 2] += coef * (npos[2] - xp[2])
            pic[0] = up
            flip[0] += udelta

            # ---- v ----
            gp = ti.Vector([xp[0] * self.inv_dx - 0.5, xp[1] * self.inv_dx, xp[2] * self.inv_dx - 0.5])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            f = gp - base.cast(ti.f32)
            wx, wy, wz = quad_w(f[0]), quad_w(f[1]), quad_w(f[2])
            vp, vdelta = 0.0, 0.0
            for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
                ni, nj, nk = base[0] + a, base[1] + b, base[2] + c
                if 0 <= ni < self.nx and 0 <= nj <= self.ny and 0 <= nk < self.nz:
                    weight = wx[a] * wy[b] * wz[c]
                    npos = ti.Vector([(ni + 0.5) * self.dx, nj * self.dx, (nk + 0.5) * self.dx])
                    vval = self.v[ni, nj, nk]
                    vp += weight * vval
                    vdelta += weight * (vval - self.v_old[ni, nj, nk])
                    coef = 4.0 * self.inv_dx * weight * vval
                    Cnew[1, 0] += coef * (npos[0] - xp[0])
                    Cnew[1, 1] += coef * (npos[1] - xp[1])
                    Cnew[1, 2] += coef * (npos[2] - xp[2])
            pic[1] = vp
            flip[1] += vdelta

            # ---- w ----
            gp = ti.Vector([xp[0] * self.inv_dx - 0.5, xp[1] * self.inv_dx - 0.5, xp[2] * self.inv_dx])
            base = ti.floor(gp - 0.5).cast(ti.i32)
            f = gp - base.cast(ti.f32)
            wx, wy, wz = quad_w(f[0]), quad_w(f[1]), quad_w(f[2])
            wp, wdelta = 0.0, 0.0
            for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
                ni, nj, nk = base[0] + a, base[1] + b, base[2] + c
                if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk <= self.nz:
                    weight = wx[a] * wy[b] * wz[c]
                    npos = ti.Vector([(ni + 0.5) * self.dx, (nj + 0.5) * self.dx, nk * self.dx])
                    wval = self.w[ni, nj, nk]
                    wp += weight * wval
                    wdelta += weight * (wval - self.w_old[ni, nj, nk])
                    coef = 4.0 * self.inv_dx * weight * wval
                    Cnew[2, 0] += coef * (npos[0] - xp[0])
                    Cnew[2, 1] += coef * (npos[1] - xp[1])
                    Cnew[2, 2] += coef * (npos[2] - xp[2])
            pic[2] = wp
            flip[2] += wdelta

            if mode == TransferMode.PIC.value:
                self.pv[p] = pic
                self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            elif mode == TransferMode.FLIP.value:
                self.pv[p] = flip_ratio * flip + (1.0 - flip_ratio) * pic
                self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
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
        lz = self.nz * self.dx
        for p in range(self.n_particles[None]):
            if damping > 0.0:
                self.pv[p] *= ti.exp(-damping * sdt)
            x = self.px[p]
            k1 = self.sample_velocity(x)
            xmid = x + 0.5 * sdt * k1
            k2 = self.sample_velocity(xmid)
            x = x + sdt * k2

            sp = self.pv[p].norm()
            if sp > vclamp:
                self.pv[p] *= vclamp / sp

            x[0] = ti.min(ti.max(x[0], eps), lx - eps)
            x[1] = ti.min(ti.max(x[1], eps), ly - eps)
            x[2] = ti.min(ti.max(x[2], eps), lz - eps)

            phi = self.sample_phi(x)
            if phi < 0.0:
                n = self.phi_gradient(x)
                x += (-phi + eps) * n
                vn = self.pv[p].dot(n)
                if vn < 0.0:
                    self.pv[p] -= vn * n
                x[0] = ti.min(ti.max(x[0], eps), lx - eps)
                x[1] = ti.min(ti.max(x[1], eps), ly - eps)
                x[2] = ti.min(ti.max(x[2], eps), lz - eps)

            self.px[p] = x

    # ===================================================================== #
    # 发射 / 交互 / 压缩
    # ===================================================================== #
    @ti.kernel
    def emit_block(self, x0: ti.f32, y0: ti.f32, z0: ti.f32, x1: ti.f32, y1: ti.f32, z1: ti.f32,
                   vx: ti.f32, vy: ti.f32, vz: ti.f32, count: ti.i32):
        for _k in range(count):
            idx = ti.atomic_add(self.n_particles[None], 1)
            if idx < self.max_particles:
                rx, ry, rz = ti.random(), ti.random(), ti.random()
                self.px[idx] = ti.Vector([x0 + rx * (x1 - x0), y0 + ry * (y1 - y0), z0 + rz * (z1 - z0)])
                self.pv[idx] = ti.Vector([vx, vy, vz])
                self.C[idx] = ti.Matrix.zero(ti.f32, 3, 3)
            else:
                self.n_particles[None] = self.max_particles

    @ti.kernel
    def apply_drag_force(self, cx: ti.f32, cy: ti.f32, cz: ti.f32,
                         vx: ti.f32, vy: ti.f32, vz: ti.f32, radius: ti.f32, strength: ti.f32):
        r2 = radius * radius
        for p in range(self.n_particles[None]):
            d = self.px[p] - ti.Vector([cx, cy, cz])
            dd = d.dot(d)
            if dd < r2:
                wgt = 1.0 - ti.sqrt(dd) / radius
                self.pv[p] += strength * wgt * ti.Vector([vx, vy, vz])

    @ti.kernel
    def _compact(self, sx0: ti.f32, sy0: ti.f32, sz0: ti.f32, sx1: ti.f32, sy1: ti.f32, sz1: ti.f32,
                 use_sink: ti.i32):
        self.n_tmp[None] = 0
        lx, ly, lz = self.nx * self.dx, self.ny * self.dx, self.nz * self.dx
        for p in range(self.n_particles[None]):
            x = self.px[p]
            alive = 1
            if x[0] < 0 or x[0] > lx or x[1] < 0 or x[1] > ly or x[2] < 0 or x[2] > lz:
                alive = 0
            if use_sink == 1 and (sx0 <= x[0] <= sx1 and sy0 <= x[1] <= sy1 and sz0 <= x[2] <= sz1):
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
        if sink is None:
            self._compact(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        else:
            self._compact(sink[0], sink[1], sink[2], sink[3], sink[4], sink[5], 1)
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
        for I in ti.grouped(self.divergence):
            if self.cell_type[I] == FLUID:
                ti.atomic_max(m, ti.abs(self.divergence[I]))
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
        self.add_gravity(sdt, cfg.gravity_x, cfg.gravity_y, cfg.gravity_z)
        self.apply_solid_boundaries()
        self.apply_viscosity(sdt)
        self.compute_divergence()
        self.solve_pressure(sdt)
        self.apply_pressure(sdt)
        self.apply_solid_boundaries()
        self.extrapolate()
        self.g2p(int(cfg.transfer.value), cfg.flip_ratio)
        vclamp = cfg.grid_velocity_clamp() * 4.0
        self.advect_particles(sdt, vclamp, cfg.vel_damping)

    def step(self):
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
