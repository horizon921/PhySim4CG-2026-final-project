"""
Taichi 刚体碰撞 Demo - Lab 1
已实现2个box的碰撞检测 collision_manifold，3D 立方体可视化
请手动实现刚体运动、冲量碰撞响应，地面处理
"""

import taichi as ti
import numpy as np

# GGUI 需要 GPU，若无 GPU 可改为 ti.cpu
ti.init(arch=ti.gpu, default_fp=ti.f32)

# 常量
# 这里只做两个方块的对撞测试
N_BODIES = 2
# 取消重力，只保留水平运动
GRAVITY = ti.Vector([0.0, 0.0, 0.0])
DT = 1.0 / 60.0
RESTITUTION = 0.6
EPSILON = 1e-6

# 立方体单位顶点 ([-1,1]^3)，8个顶点 - 用于 Taichi kernel 和 Python
CUBE_LOCAL_VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
], dtype=np.float32)

# 12 个三角形索引
CUBE_INDICES = np.array([
    0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
    0, 3, 7, 0, 7, 4, 1, 2, 6, 1, 6, 5,
    0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7,
], dtype=np.int32)

# Taichi 可用的几何数据
cube_local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
cube_indices_ti = ti.field(dtype=ti.i32, shape=36)
cube_local_verts.from_numpy(CUBE_LOCAL_VERTICES)
cube_indices_ti.from_numpy(CUBE_INDICES)

# 刚体状态字段
position = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N_BODIES)
angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
half_extent = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES)
mass = ti.field(dtype=ti.f32, shape=N_BODIES)

# 可视化：每个立方体 8 顶点，总共 N_BODIES * 8 个顶点
mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=N_BODIES * 8)
mesh_indices = ti.field(dtype=ti.i32, shape=N_BODIES * 36)



@ti.kernel
def init_rigid_bodies():
    """初始化 2 个立方体做对撞，无重力"""
    # 立方体 0：左边，向右运动
    position[0] = ti.Vector([-1.0, 0.75, 0.2])
    velocity[0] = ti.Vector([1.0, 0.0, 0.0])
    rotation[0] = ti.Matrix.identity(ti.f32, 3)
    angular_velocity[0] = ti.Vector([0.0, 0.0, 0.0])
    half_extent[0] = ti.Vector([0.3, 0.3, 0.3])
    mass[0] = 1.0

    # 立方体 1：右边，向左运动
    position[1] = ti.Vector([1.0, 0.5, 0.0])
    velocity[1] = ti.Vector([-1.0, 0.0, 0.0])
    rotation[1] = ti.Matrix.identity(ti.f32, 3)
    angular_velocity[1] = ti.Vector([0.0, 0.0, 0.0])
    half_extent[1] = ti.Vector([0.3, 0.3, 0.3])
    mass[1] = 1.0

    #########
    # This is an example of two rigid boxes
    # modify the code to support more complex demos
    #########


@ti.kernel
def integrate():
    """刚体运动积分"""
    #########
    # add your code here  
    # you may need some tool functions, e.g., skew(...)
    #########
    return


def get_box_vertices_correct(i: int) -> np.ndarray:
    """立方体顶点：局部 (±dx, ±dy, ±dz)"""
    pos = position[i].to_numpy()
    rot = rotation[i].to_numpy()
    ext = half_extent[i].to_numpy()
    verts = np.zeros((8, 3), dtype=np.float32)
    for k in range(8):
        local = CUBE_LOCAL_VERTICES[k] * ext
        verts[k] = rot @ local + pos
    return verts


def collision_manifold(i: int, j: int):
    """
    返回:
        collided: bool
        normal: (3,)
        penetration: float
        contact_point: (3,)
    """
    verts_a = get_box_vertices_correct(i)
    verts_b = get_box_vertices_correct(j)
    rot_a = rotation[i].to_numpy()
    rot_b = rotation[j].to_numpy()

    axes_a = [rot_a[:, k] for k in range(3)]
    axes_b = [rot_b[:, k] for k in range(3)]

    axes = axes_a + axes_b

    # 9 cross axes
    for ia in range(3):
        for ib in range(3):
            cross_axis = np.cross(axes_a[ia], axes_b[ib])
            n2 = np.dot(cross_axis, cross_axis)
            if n2 > EPSILON:
                axes.append(cross_axis / np.sqrt(n2))

    min_overlap = float('inf')
    best_axis = None

    center_a = position[i].to_numpy()
    center_b = position[j].to_numpy()

    for axis in axes:
        proj_a = verts_a @ axis
        proj_b = verts_b @ axis

        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()

        # 分离轴
        if max_a < min_b - EPSILON or max_b < min_a - EPSILON:
            return False, None, 0.0, None

        overlap = min(max_a - min_b, max_b - min_a)

        if overlap < min_overlap:
            min_overlap = overlap

            # 方向从 j 指向 i
            d = center_a - center_b
            if np.dot(axis, d) < 0:
                axis = -axis

            best_axis = axis

    if best_axis is None:
        return False, None, 0.0, None

    normal = best_axis / np.linalg.norm(best_axis)
    penetration = min_overlap

    # --------- 计算 contact point ---------
    # A 上沿 -normal 的 support 点
    idx_a = np.argmin(verts_a @ normal)
    pa = verts_a[idx_a]

    # B 上沿 +normal 的 support 点
    idx_b = np.argmax(verts_b @ normal)
    pb = verts_b[idx_b]

    contact_point = 0.5 * (pa + pb)

    return True, normal, penetration, contact_point


def resolve_collision_fixed(i: int, j: int, normal: np.ndarray, penetration: float, contact: np.ndarray):
    """冲量法碰撞响应 + 位置修正"""
    #########
    # add your code here to update position, velocity, and so on.
    #########

@ti.kernel
def update_mesh_vertices():
    """根据刚体状态更新可视化顶点"""
    for i in range(N_BODIES):
        pos = position[i]
        rot = rotation[i]
        ext = half_extent[i]
        for k in range(8):
            lv = cube_local_verts[k]
            local = ti.Vector([lv[0] * ext[0], lv[1] * ext[1], lv[2] * ext[2]])
            world = rot @ local + pos
            mesh_vertices[i * 8 + k] = world
        for t in range(12):
            for v in range(3):
                mesh_indices[i * 36 + t * 3 + v] = i * 8 + cube_indices_ti[t * 3 + v]


def main():
    init_rigid_bodies()
    update_mesh_vertices()

    window = ti.ui.Window("Rigid Body Collision - Lab 1", res=(1024, 768))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(3, 2, 3)
    camera.lookat(0, 0.5, 0)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.ambient_light((0.6, 0.6, 0.6))
    scene.point_light((5, 5, 5), (1.2, 1.2, 1.2))

    colors = [(0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.3, 0.8, 0.3)]

    # --------------------------
    # 创建地板
    floor_size = 5.0
    floor_color = (0.7, 0.7, 0.7)

    floor_vertices = np.array([
        [-floor_size, 0.0, -floor_size],
        [ floor_size, 0.0, -floor_size],
        [ floor_size, 0.0,  floor_size],
        [-floor_size, 0.0,  floor_size],
    ], dtype=np.float32)
    floor_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

    floor_vertices_ti = ti.Vector.field(3, dtype=ti.f32, shape=4)
    floor_indices_ti = ti.field(dtype=ti.i32, shape=6)
    floor_vertices_ti.from_numpy(floor_vertices)
    floor_indices_ti.from_numpy(floor_indices)
    # --------------------------

    while window.running:
        for _ in range(2):  # 子步提高稳定性
            integrate()
            ti.sync()
            for i in range(N_BODIES):
                for j in range(i + 1, N_BODIES):
                    collided, normal, penetration, contact = collision_manifold(i, j)

                    if collided:
                        resolve_collision_fixed(i, j, normal, penetration, contact)

        # 地面碰撞（取 OBB 顶点最小 y）
        for i in range(N_BODIES):
            verts = get_box_vertices_correct(i)
            min_y = float(verts[:, 1].min())
            if min_y < 0:
                print(i,verts)
                #########
                # add your code here to update position, velocity, and so on.
                # if min_y < 0: ...
                #########


        update_mesh_vertices()

        scene.set_camera(camera)
        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)

        # 设置背景色（例如浅灰）
        canvas = window.get_canvas()
        canvas.set_background_color((0.8, 0.8, 0.85))

        for body_id in range(N_BODIES):
            # 先画实心面片
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=colors[body_id],
                index_offset=body_id * 36,
                index_count=36,
            )
            # 再叠加一层线框轮廓
            scene.mesh(
                mesh_vertices,
                mesh_indices,
                color=(0.0, 0.0, 0.0),
                index_offset=body_id * 36,
                index_count=36,
                show_wireframe=True,
            )

        # 画地板
        scene.mesh(floor_vertices_ti, floor_indices_ti, color=floor_color)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
