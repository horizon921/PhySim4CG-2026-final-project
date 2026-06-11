"""Headless orthographic renderer for 3D fluid/soft-body scenes."""

import os

import numpy as np
import taichi as ti


def _camera_basis(azim_deg: float, elev_deg: float):
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    forward = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward


def render_frame(scene, width: int = 600, azim: float = 35.0, elev: float = 22.0,
                 vref: float = 3.0, max_fluid_particles: int | None = None,
                 draw_ui_overlay: bool = True, color_by_velocity: bool = True,
                 draw_solid_voxels: bool = True) -> np.ndarray:
    W = width
    H = width
    img = np.empty((W, H, 3), dtype=np.float32)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    bottom = np.array([0.020, 0.032, 0.050], dtype=np.float32)
    top = np.array([0.035, 0.060, 0.095], dtype=np.float32)
    img[:] = (bottom[None, :] * (1.0 - y[:, None]) + top[None, :] * y[:, None])[None, :, :]
    img[:, int(H * 0.86):, :] *= 0.45

    lx, ly, lz = scene.lx, scene.ly, scene.lz
    center = np.array([lx * 0.5, ly * 0.5, lz * 0.5])
    diag = np.sqrt(lx * lx + ly * ly + lz * lz)
    scale = 0.74 * W / diag
    y_origin = H * 0.42
    right, up, forward = _camera_basis(azim, elev)

    pts_list, col_list = [], []

    dx = scene.cfg.dx
    if draw_solid_voxels:
        phi = scene.solver.solid_phi.to_numpy()
        si, sj, sk = np.where(phi < 0.0)
        if len(si) > 0:
            ob = np.stack([(si + 0.5) * dx, (sj + 0.5) * dx, (sk + 0.5) * dx], axis=1)
            interior = ((ob[:, 0] > 2 * dx) & (ob[:, 0] < lx - 2 * dx) &
                        (ob[:, 1] > 2 * dx) & (ob[:, 1] < ly - 2 * dx) &
                        (ob[:, 2] > 2 * dx) & (ob[:, 2] < lz - 2 * dx))
            ob = ob[interior]
            if len(ob) > 0:
                pts_list.append(ob)
                col_list.append(np.tile(np.array([0.34, 0.38, 0.46], np.float32), (len(ob), 1)))

    pos = scene.solver.particle_positions_np()
    if len(pos) > 0:
        if max_fluid_particles is not None and len(pos) > max_fluid_particles:
            idx = np.linspace(0, len(pos) - 1, max_fluid_particles, dtype=np.int32)
            pos = pos[idx]
        if color_by_velocity:
            vel = scene.solver.particle_velocities_np()
            if max_fluid_particles is not None and len(vel) > len(pos):
                vel = vel[idx]
            sp = np.linalg.norm(vel, axis=1)
            t = np.clip(sp / vref, 0.0, 1.0)
            col = np.stack([0.08 + 0.34 * t ** 1.4,
                            0.30 + 0.46 * t,
                            0.66 + 0.30 * (1.0 - (1.0 - t) ** 2)], axis=1).astype(np.float32)
        else:
            h = np.clip(pos[:, 1] / max(ly, 1e-6), 0.0, 1.0)
            col = np.stack([0.07 + 0.10 * h,
                            0.30 + 0.28 * h,
                            0.70 + 0.18 * h], axis=1).astype(np.float32)
        pts_list.append(pos)
        col_list.append(col)

    for body in getattr(scene, "soft_bodies", []):
        bpos = _soft_body_points(body)
        if len(bpos) == 0:
            continue
        pts_list.append(bpos)
        col_list.append(np.tile(np.array([1.00, 0.62, 0.18], np.float32), (len(bpos), 1)))

    if pts_list:
        P = np.concatenate(pts_list, axis=0)
        Col = np.concatenate(col_list, axis=0)
        rel = P - center
        sx = rel @ right
        sy = rel @ up
        depth = rel @ forward
        order = np.argsort(-depth)
        sx, sy, depth, Col = sx[order], sy[order], depth[order], Col[order]
        dmin, dmax = depth.min(), depth.max()
        shade = 0.55 + 0.45 * (1.0 - (depth - dmin) / max(dmax - dmin, 1e-6))
        Col = Col * shade[:, None]
        px = np.clip((W * 0.5 + sx * scale).astype(int), 0, W - 1)
        py = np.clip((y_origin + sy * scale).astype(int), 0, H - 1)
        rad = max(1, min(2, int(round(0.09 * scale * dx))))
        for ox in range(-rad, rad + 1):
            for oy in range(-rad, rad + 1):
                if ox * ox + oy * oy > rad * rad + 1:
                    continue
                ix = np.clip(px + ox, 0, W - 1)
                iy = np.clip(py + oy, 0, H - 1)
                img[ix, iy] = Col

    for region in getattr(scene, "hazard_regions", []):
        _draw_region_fill(img, center, scale, right, up, region, W, H,
                          np.array([0.60, 0.04, 0.04], np.float32), alpha=0.14)
        _draw_region_wire(img, center, scale, right, up, region, W, H,
                          np.array([0.95, 0.18, 0.16], np.float32))
    target_regions = getattr(scene, "target_regions", [])
    current_target = getattr(scene, "current_target", 0)
    if target_regions:
        for idx, region in enumerate(target_regions):
            color = np.array([0.18, 0.95, 0.42], np.float32)
            fill = np.array([0.10, 0.55, 0.22], np.float32)
            alpha = 0.16
            if idx != current_target:
                color = np.array([0.12, 0.42, 0.22], np.float32)
                fill = np.array([0.05, 0.22, 0.11], np.float32)
                alpha = 0.08
            if idx == current_target:
                _draw_region_fill(img, center, scale, right, up, region, W, H,
                                  np.array([0.20, 0.95, 0.40], np.float32), alpha=0.08)
            _draw_region_fill(img, center, scale, right, up, region, W, H, fill, alpha=alpha)
            _draw_region_wire(img, center, scale, right, up, region, W, H, color)
    elif getattr(scene, "target", None) is not None:
        _draw_region_fill(img, center, scale, right, up, scene.target, W, H,
                          np.array([0.10, 0.55, 0.22], np.float32), alpha=0.16)
        _draw_region_wire(img, center, scale, right, up, scene.target, W, H,
                          np.array([0.18, 0.95, 0.42], np.float32))

    _draw_box_wire(img, center, scale, right, up, lx, ly, lz, W, H)
    _draw_floor_grid(img, center, scale, right, up, lx, lz, W, H)
    for body in getattr(scene, "soft_bodies", []):
        _draw_soft_body_wire(img, body, center, scale, right, up, W, H)
    for emitter in getattr(scene, "emitters", []):
        _draw_emitter_arrow(img, center, scale, right, up, emitter, W, H)
    if draw_ui_overlay:
        _draw_game_overlay(img, scene, W, H)
    return img


def render_map_frame(scene, width: int = 600, max_fluid_particles: int | None = None,
                     draw_ui_overlay: bool = True) -> np.ndarray:
    W = width
    H = width
    img = np.empty((W, H, 3), dtype=np.float32)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    bottom = np.array([0.018, 0.028, 0.043], dtype=np.float32)
    top = np.array([0.030, 0.052, 0.078], dtype=np.float32)
    img[:] = (bottom[None, :] * (1.0 - y[:, None]) + top[None, :] * y[:, None])[None, :, :]
    img[:, int(H * 0.86):, :] *= 0.45

    lx, lz = scene.lx, scene.lz
    x0, y0, x1, y1 = _map_bounds(W, H)
    _blend_rect(img, x0, y0, x1, y1, np.array([0.035, 0.055, 0.075], np.float32), 0.70)
    _draw_map_grid(img, x0, y0, x1, y1)

    def project(points):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        px = x0 + pts[:, 0] / max(lx, 1e-6) * (x1 - x0)
        py = y0 + pts[:, 2] / max(lz, 1e-6) * (y1 - y0)
        return np.clip(px.astype(np.int32), 0, W - 1), np.clip(py.astype(np.int32), 0, H - 1)

    for region in getattr(scene, "hazard_regions", []):
        _draw_map_region(img, region, x0, y0, x1, y1, lx, lz,
                         np.array([0.70, 0.04, 0.04], np.float32), 0.22)

    target_regions = getattr(scene, "target_regions", [])
    current_target = getattr(scene, "current_target", 0)
    for idx, region in enumerate(target_regions):
        active = idx == current_target
        fill = np.array([0.08, 0.48, 0.18], np.float32) if active else np.array([0.04, 0.22, 0.10], np.float32)
        alpha = 0.30 if active else 0.18
        _draw_map_region(img, region, x0, y0, x1, y1, lx, lz, fill, alpha)

    for shape in getattr(scene, "shapes", []):
        if all(hasattr(shape, name) for name in ("hx", "hz")):
            region = (shape.cx - shape.hx, 0.0, shape.cz - shape.hz,
                      shape.cx + shape.hx, 0.0, shape.cz + shape.hz)
            _draw_map_region(img, region, x0, y0, x1, y1, lx, lz,
                             np.array([0.34, 0.38, 0.46], np.float32), 0.55)
        elif hasattr(shape, "r"):
            px = int(x0 + shape.cx / max(lx, 1e-6) * (x1 - x0))
            py = int(y0 + shape.cz / max(lz, 1e-6) * (y1 - y0))
            rr = max(2, int(shape.r / max(lx, lz, 1e-6) * (x1 - x0)))
            _draw_disc(img, px, py, rr, np.array([0.34, 0.38, 0.46], np.float32))

    pos = scene.solver.particle_positions_np()
    if len(pos) > 0:
        if max_fluid_particles is not None and len(pos) > max_fluid_particles:
            idx = np.linspace(0, len(pos) - 1, max_fluid_particles, dtype=np.int32)
            pos = pos[idx]
        px, py = project(pos)
        h = np.clip(pos[:, 1] / max(scene.ly, 1e-6), 0.0, 1.0)
        colors = np.stack([0.06 + 0.08 * h, 0.28 + 0.24 * h, 0.72 + 0.16 * h], axis=1)
        for ox, oy in ((0, 0), (1, 0), (0, 1)):
            ix = np.clip(px + ox, 0, W - 1)
            iy = np.clip(py + oy, 0, H - 1)
            img[ix, iy] = colors

    for body in getattr(scene, "soft_bodies", []):
        pts = _soft_body_points(body)
        if len(pts) == 0:
            continue
        px, py = project(pts)
        for x, yy in zip(px, py):
            _draw_disc(img, int(x), int(yy), 3, np.array([1.00, 0.48, 0.10], np.float32))
        body_pos = getattr(scene, "primary_body_position", None)
        if body_pos is not None:
            cx, cy = project(np.asarray(body_pos, dtype=np.float32).reshape(1, 3))
            _draw_disc(img, int(cx[0]), int(cy[0]), 6, np.array([1.00, 0.72, 0.18], np.float32))

    for emitter in getattr(scene, "emitters", []):
        _draw_map_emitter(img, emitter, x0, y0, x1, y1, lx, lz)

    border = np.array([0.32, 0.39, 0.48], np.float32)
    _draw_screen_line(img, (x0, y0), (x1, y0), border, thickness=1)
    _draw_screen_line(img, (x1, y0), (x1, y1), border, thickness=1)
    _draw_screen_line(img, (x1, y1), (x0, y1), border, thickness=1)
    _draw_screen_line(img, (x0, y1), (x0, y0), border, thickness=1)

    if draw_ui_overlay:
        _draw_game_overlay(img, scene, W, H)
    return img


def _map_bounds(W, H):
    margin = max(26, W // 14)
    top_reserved = max(68, H // 4)
    bottom_reserved = max(24, H // 11)
    size = min(W - 2 * margin, H - top_reserved - bottom_reserved)
    x0 = (W - size) // 2
    y0 = bottom_reserved + max(0, (H - top_reserved - bottom_reserved - size) // 2)
    return x0, y0, x0 + size, y0 + size


def _draw_map_grid(img, x0, y0, x1, y1):
    color = np.array([0.12, 0.16, 0.20], np.float32)
    for i in range(6):
        t = i / 5
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        _draw_screen_line(img, (x, y0), (x, y1), color, thickness=0)
        _draw_screen_line(img, (x0, y), (x1, y), color, thickness=0)


def _draw_map_region(img, region, mx0, my0, mx1, my1, lx, lz, color, alpha):
    x0, _, z0, x1, _, z1 = region
    rx0 = int(mx0 + x0 / max(lx, 1e-6) * (mx1 - mx0))
    rx1 = int(mx0 + x1 / max(lx, 1e-6) * (mx1 - mx0))
    ry0 = int(my0 + z0 / max(lz, 1e-6) * (my1 - my0))
    ry1 = int(my0 + z1 / max(lz, 1e-6) * (my1 - my0))
    _blend_rect(img, rx0, ry0, rx1, ry1, color, alpha)
    _draw_screen_line(img, (rx0, ry0), (rx1, ry0), color, thickness=1)
    _draw_screen_line(img, (rx1, ry0), (rx1, ry1), color, thickness=1)
    _draw_screen_line(img, (rx1, ry1), (rx0, ry1), color, thickness=1)
    _draw_screen_line(img, (rx0, ry1), (rx0, ry0), color, thickness=1)


def _draw_disc(img, cx, cy, radius, color):
    W, H = img.shape[:2]
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    for x in range(x0, x1):
        for y in range(y0, y1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius:
                img[x, y] = color


def _draw_map_emitter(img, emitter, mx0, my0, mx1, my1, lx, lz):
    x0, _, z0, x1, _, z1 = emitter.region
    sx = mx0 + ((x0 + x1) * 0.5) / max(lx, 1e-6) * (mx1 - mx0)
    sy = my0 + ((z0 + z1) * 0.5) / max(lz, 1e-6) * (my1 - my0)
    vel = np.asarray(emitter.velocity, dtype=np.float32)
    direction = np.array([vel[0], vel[2]], dtype=np.float32)
    norm = max(float(np.linalg.norm(direction)), 1e-6)
    direction /= norm
    ex = sx + direction[0] * 42
    ey = sy + direction[1] * 42
    color = np.array([1.00, 0.86, 0.18], np.float32) if getattr(emitter, "enabled", True) else np.array([0.34, 0.36, 0.40], np.float32)
    _draw_screen_line(img, (sx, sy), (ex, ey), color, thickness=3)
    _draw_disc(img, int(ex), int(ey), 5, color)


def _soft_body_points(body) -> np.ndarray:
    if hasattr(body, "positions_np"):
        return np.asarray(body.positions_np(), dtype=np.float32).reshape(-1, 3)
    if hasattr(body, "surface_points_np"):
        return np.asarray(body.surface_points_np(), dtype=np.float32).reshape(-1, 3)
    return np.zeros((0, 3), dtype=np.float32)


def _draw_box_wire(img, center, scale, right, up, lx, ly, lz, W, H):
    _draw_region_wire(img, center, scale, right, up,
                      (0.0, 0.0, 0.0, lx, ly, lz), W, H,
                      np.array([0.20, 0.24, 0.30], np.float32))


def _draw_floor_grid(img, center, scale, right, up, lx, lz, W, H):
    color = np.array([0.13, 0.18, 0.24], np.float32)
    y = 0.002
    steps = 5
    for i in range(steps + 1):
        x = lx * i / steps
        p0 = _project_point((x, y, 0.0), center, scale, right, up, W, H)
        p1 = _project_point((x, y, lz), center, scale, right, up, W, H)
        _draw_screen_line(img, p0, p1, color, thickness=0)
    for k in range(steps + 1):
        z = lz * k / steps
        p0 = _project_point((0.0, y, z), center, scale, right, up, W, H)
        p1 = _project_point((lx, y, z), center, scale, right, up, W, H)
        _draw_screen_line(img, p0, p1, color, thickness=0)


def _draw_shape_wires(img, scene, center, scale, right, up, W, H):
    color = np.array([0.42, 0.47, 0.56], np.float32)
    for shape in getattr(scene, "shapes", []):
        if all(hasattr(shape, name) for name in ("hx", "hy", "hz")):
            _draw_region_wire(
                img, center, scale, right, up,
                (shape.cx - shape.hx, shape.cy - shape.hy, shape.cz - shape.hz,
                 shape.cx + shape.hx, shape.cy + shape.hy, shape.cz + shape.hz),
                W, H, color,
            )
        elif hasattr(shape, "r"):
            _draw_sphere_wire(img, center, scale, right, up, shape, W, H, color)


def _draw_sphere_wire(img, center, scale, right, up, sphere, W, H, color):
    c = np.array([sphere.cx, sphere.cy, sphere.cz], dtype=np.float32)
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        points = []
        for theta in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=True):
            p = c.copy()
            p[axis_a] += sphere.r * np.cos(theta)
            p[axis_b] += sphere.r * np.sin(theta)
            points.append(_project_point(p, center, scale, right, up, W, H))
        for p0, p1 in zip(points[:-1], points[1:]):
            _draw_screen_line(img, p0, p1, color, thickness=0)


def _draw_region_wire(img, center, scale, right, up, region, W, H, line_col):
    x0, y0, z0, x1, y1, z1 = region
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], np.float32)
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    rel = corners - center
    sx = rel @ right
    sy = rel @ up
    cx = (W * 0.5 + sx * scale)
    cy = (H * 0.43 + sy * scale)
    for a, b in edges:
        n = int(max(abs(cx[a] - cx[b]), abs(cy[a] - cy[b])) + 1)
        for t in np.linspace(0, 1, max(n, 2)):
            ix = int(np.clip(cx[a] * (1 - t) + cx[b] * t, 0, W - 1))
            iy = int(np.clip(cy[a] * (1 - t) + cy[b] * t, 0, H - 1))
            img[ix, iy] = line_col


def _project_point(point, center, scale, right, up, W, H):
    rel = np.asarray(point, dtype=np.float32) - center
    sx = rel @ right
    sy = rel @ up
    return (
        float(np.clip(W * 0.5 + sx * scale, 0, W - 1)),
        float(np.clip(H * 0.43 + sy * scale, 0, H - 1)),
    )


def _draw_screen_line(img, p0, p1, color, thickness=1):
    W, H = img.shape[:2]
    x0, y0 = p0
    x1, y1 = p1
    n = int(max(abs(x1 - x0), abs(y1 - y0)) + 1)
    for t in np.linspace(0, 1, max(n, 2)):
        cx = int(np.clip(x0 * (1 - t) + x1 * t, 0, W - 1))
        cy = int(np.clip(y0 * (1 - t) + y1 * t, 0, H - 1))
        img[max(0, cx - thickness):min(W, cx + thickness + 1),
            max(0, cy - thickness):min(H, cy + thickness + 1)] = color


def _draw_emitter_arrow(img, center, scale, right, up, emitter, W, H):
    if not getattr(emitter, "enabled", True):
        color = np.array([0.34, 0.36, 0.40], np.float32)
    else:
        color = np.array([1.00, 0.86, 0.18], np.float32)
    x0, y0, z0, x1, y1, z1 = emitter.region
    start = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], np.float32)
    vel = np.asarray(emitter.velocity, dtype=np.float32)
    speed = float(np.linalg.norm(vel))
    if speed < 1e-6:
        return
    direction = vel / speed
    end = start + direction * 0.36
    p0 = _project_point(start, center, scale, right, up, W, H)
    p1 = _project_point(end, center, scale, right, up, W, H)
    _draw_screen_line(img, p0, p1, color, thickness=2)
    head = 8.0
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / norm, dy / norm
    left = (p1[0] - head * ux - head * 0.55 * uy, p1[1] - head * uy + head * 0.55 * ux)
    right_pt = (p1[0] - head * ux + head * 0.55 * uy, p1[1] - head * uy - head * 0.55 * ux)
    _draw_screen_line(img, p1, left, color, thickness=2)
    _draw_screen_line(img, p1, right_pt, color, thickness=2)


def _project_region_bounds(center, scale, right, up, region, W, H):
    x0, y0, z0, x1, y1, z1 = region
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], np.float32)
    rel = corners - center
    sx = rel @ right
    sy = rel @ up
    cx = W * 0.5 + sx * scale
    cy = H * 0.43 + sy * scale
    return (
        int(np.clip(np.floor(cx.min()), 0, W - 1)),
        int(np.clip(np.floor(cy.min()), 0, H - 1)),
        int(np.clip(np.ceil(cx.max()), 0, W - 1)),
        int(np.clip(np.ceil(cy.max()), 0, H - 1)),
    )


def _draw_region_fill(img, center, scale, right, up, region, W, H, color, alpha=0.12):
    x0, y0, x1, y1 = _project_region_bounds(center, scale, right, up, region, W, H)
    if x1 <= x0 or y1 <= y0:
        return
    _blend_rect(img, x0, y0, x1, y1, color, alpha)


def _draw_soft_body_wire(img, body, center, scale, right, up, W, H):
    if not hasattr(body, "constraints"):
        return
    pts = _soft_body_points(body)
    if len(pts) == 0:
        return
    rel = pts - center
    sx = rel @ right
    sy = rel @ up
    cx = (W * 0.5 + sx * scale)
    cy = (H * 0.43 + sy * scale)
    line_col = np.array([1.00, 0.42, 0.10], np.float32)
    spacing = getattr(getattr(body, "cfg", None), "spacing", 1.0)
    for a, b, rest in body.constraints:
        if rest > spacing * 1.45:
            continue
        n = int(max(abs(cx[a] - cx[b]), abs(cy[a] - cy[b])) + 1)
        for t in np.linspace(0, 1, max(n, 2)):
            ix = int(np.clip(cx[a] * (1 - t) + cx[b] * t, 0, W - 1))
            iy = int(np.clip(cy[a] * (1 - t) + cy[b] * t, 0, H - 1))
            img[ix, iy] = line_col


def _blend_rect(img, x0, y0, x1, y1, color, alpha):
    W, H = img.shape[:2]
    x0 = int(np.clip(x0, 0, W))
    x1 = int(np.clip(x1, 0, W))
    y0 = int(np.clip(y0, 0, H))
    y1 = int(np.clip(y1, 0, H))
    if x1 <= x0 or y1 <= y0:
        return
    img[x0:x1, y0:y1] = img[x0:x1, y0:y1] * (1.0 - alpha) + color * alpha


def _draw_bar(img, x0, y0, x1, y1, frac, color, back=np.array([0.12, 0.16, 0.22], np.float32)):
    frac = float(np.clip(frac, 0.0, 1.0))
    _blend_rect(img, x0, y0, x1, y1, back, 0.95)
    fill_x = int(round(x0 + (x1 - x0) * frac))
    _blend_rect(img, x0, y0, fill_x, y1, color, 1.0)
    border = np.array([0.70, 0.78, 0.90], np.float32)
    img[x0:x1, y0:y0 + 1] = border
    img[x0:x1, y1 - 1:y1] = border
    img[x0:x0 + 1, y0:y1] = border
    img[x1 - 1:x1, y0:y1] = border


def _draw_game_overlay(img, scene, W, H):
    if not hasattr(scene, "game_status"):
        return

    panel = np.array([0.025, 0.040, 0.065], np.float32)
    _blend_rect(img, 0, H - 104, W, H, panel, 0.82)
    _blend_rect(img, 0, 0, W, 34, panel, 0.55)

    status = getattr(scene, "game_status", "playing")
    status_col = {
        "playing": np.array([0.18, 0.88, 0.48], np.float32),
        "won": np.array([0.25, 0.90, 0.95], np.float32),
        "lost": np.array([0.95, 0.18, 0.16], np.float32),
    }.get(status, np.array([0.85, 0.85, 0.85], np.float32))
    _blend_rect(img, 0, H - 104, 7, H, status_col, 1.0)

    margin = max(16, W // 36)
    bar_w = max(170, int(W * 0.42))
    _draw_bar(img, margin, H - 78, margin + bar_w, H - 67,
              getattr(scene, "target_progress", 0.0),
              np.array([0.16, 0.90, 0.38], np.float32))

    water_budget = getattr(scene, "water_budget", None)
    remaining = getattr(scene, "remaining_water", None)
    water_frac = 1.0 if water_budget in (None, 0) else (remaining or 0) / max(water_budget, 1)
    _draw_bar(img, margin, H - 96, margin + bar_w, H - 85,
              water_frac, np.array([0.16, 0.54, 0.96], np.float32))

    jet_on = getattr(scene, "player_jet_enabled", None)
    jet_col = np.array([0.16, 0.62, 1.00], np.float32) if jet_on else np.array([0.25, 0.30, 0.38], np.float32)
    _blend_rect(img, W - margin - 74, H - 96, W - margin, H - 67, jet_col, 0.80)

    legend_x = margin
    legend_y = 14
    swatches = [
        np.array([0.10, 0.36, 0.78], np.float32),
        np.array([0.95, 0.34, 0.12], np.float32),
        np.array([0.18, 0.95, 0.42], np.float32),
        np.array([0.95, 0.18, 0.16], np.float32),
        np.array([1.00, 0.86, 0.18], np.float32),
    ]
    for i, color in enumerate(swatches):
        x = legend_x + i * 42
        _blend_rect(img, x, legend_y, x + 17, legend_y + 12, color, 1.0)


def save_png(img: np.ndarray, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ti.tools.imwrite(img, path)
