"""Scene assembly for fluid scenes with dynamic soft-body boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import FluidConfig
from ..coupling import FluidSoftCoupler, Shape, static_solid_phi
from ..fluid import FlipSolver
from .scene import Emitter, Region


@dataclass
class GameZone:
    name: str
    region: Region
    hold_frames: int = 45
    score: int = 100


@dataclass
class CoupledScene:
    name: str
    cfg: FluidConfig
    soft_bodies: List[object] = field(default_factory=list)
    shapes: List[Shape] = field(default_factory=list)
    init_blocks: List[Region] = field(default_factory=list)
    emitters: List[Emitter] = field(default_factory=list)
    sink: Optional[Region] = None
    coupler: FluidSoftCoupler = field(default_factory=FluidSoftCoupler)
    target: Optional[Region] = None
    targets: List[GameZone] = field(default_factory=list)
    hazards: List[GameZone] = field(default_factory=list)
    success_hold_frames: int = 45
    max_frames: Optional[int] = None
    water_budget: Optional[int] = None
    player_jet_index: int = 0
    pulse_cooldown_frames: int = 45
    pulse_cost: int = 400
    hint: str = ""
    objective: str = ""
    tutorial: tuple[str, ...] = ()
    jet_assist: float = 0.0

    def __post_init__(self):
        self.solver = FlipSolver(self.cfg)
        self.static_phi = static_solid_phi(
            self.cfg.res_x, self.cfg.res_y, self.cfg.res_z, self.cfg.dx,
            shapes=self.shapes, with_walls=True,
        )
        self._initial_soft_bodies = [self._clone_soft_body(body) for body in self.soft_bodies]
        self.frame = 0
        self.target_frames = 0
        self.game_status = "playing"
        self.current_target = 0
        self.score = 0
        self.emitted_particles = 0
        self.hazard_frames = 0
        self.pulse_cooldown = 0
        self.completed_targets = 0
        self.last_event = ""
        self.reset()

    @property
    def lx(self) -> float:
        return self.cfg.res_x * self.cfg.dx

    @property
    def ly(self) -> float:
        return self.cfg.res_y * self.cfg.dx

    @property
    def lz(self) -> float:
        return self.cfg.res_z * self.cfg.dx

    def _update_solid_fields(self):
        self.coupler.rasterize_soft_boundary(self.solver, self.soft_bodies, self.shapes)

    def reset(self):
        self.solver.n_particles[None] = 0
        for body, initial in zip(self.soft_bodies, self._initial_soft_bodies):
            self._reset_soft_body(body, initial)
        self._update_solid_fields()
        for region in self.init_blocks:
            self.solver.add_particle_block(*region)
        self.frame = 0
        self.target_frames = 0
        self.game_status = "playing"
        self.current_target = 0
        self.score = 0
        self.emitted_particles = 0
        self.hazard_frames = 0
        self.pulse_cooldown = 0
        self.completed_targets = 0
        self.last_event = ""
        self._sync_legacy_target()
        if self.objective:
            self.last_event = "ready"

    def step(self) -> int:
        if self.game_status != "playing":
            return 0
        if self.pulse_cooldown > 0:
            self.pulse_cooldown -= 1
        self._apply_jet_assist()
        for e in self.emitters:
            if e.enabled and self.solver.n_particles[None] < self.cfg.max_particles:
                count = self._emitter_count(e)
                if count > 0:
                    self.solver.emit_block(*e.region, e.velocity[0], e.velocity[1], e.velocity[2], count)
                    self.emitted_particles += count
        self._update_solid_fields()
        nsub = self.solver.step()
        self.coupler.apply_fluid_forces(self.solver, self.soft_bodies)
        for body in self.soft_bodies:
            self._step_soft_body(body, self.cfg.dt)
        if self.sink is not None:
            self.solver.compact(self.sink)
        elif self.solver.n_particles[None] > 0.98 * self.cfg.max_particles:
            self.solver.compact(None)
        self.frame += 1
        self._update_game_status()
        return nsub

    @property
    def n_particles(self) -> int:
        return self.solver.n_particles[None]

    @property
    def target_progress(self) -> float:
        hold = self.active_target.hold_frames if self.active_target is not None else self.success_hold_frames
        return min(1.0, self.target_frames / max(hold, 1))

    @property
    def active_target(self) -> GameZone | None:
        if self.targets:
            if 0 <= self.current_target < len(self.targets):
                return self.targets[self.current_target]
            return None
        if self.target is None:
            return None
        return GameZone("target", self.target, self.success_hold_frames, 100)

    @property
    def target_regions(self) -> list[Region]:
        if self.targets:
            return [t.region for t in self.targets]
        return [] if self.target is None else [self.target]

    @property
    def hazard_regions(self) -> list[Region]:
        return [h.region for h in self.hazards]

    @property
    def remaining_water(self) -> int | None:
        if self.water_budget is None:
            return None
        return max(0, self.water_budget - self.emitted_particles)

    @property
    def status_line(self) -> str:
        active = self.active_target
        target_name = "done" if active is None else active.name
        water = "inf" if self.remaining_water is None else str(self.remaining_water)
        return (f"status={self.game_status} score={self.score} target={target_name} "
                f"{self.target_progress * 100:.0f}% water={water} "
                f"pulse_cd={self.pulse_cooldown} event={self.last_event}")

    @property
    def primary_body_position(self) -> np.ndarray | None:
        if not self.soft_bodies:
            return None
        body = self.soft_bodies[0]
        if hasattr(body, "position"):
            return np.asarray(body.position, dtype=np.float32)
        if hasattr(body, "surface_points_np"):
            pts = np.asarray(body.surface_points_np(), dtype=np.float32)
            if len(pts):
                return pts.mean(axis=0)
        return None

    @property
    def player_jet_velocity(self) -> tuple[float, float, float] | None:
        jet = self._player_jet()
        return None if jet is None else jet.velocity

    @property
    def player_jet_enabled(self) -> bool | None:
        jet = self._player_jet()
        return None if jet is None else jet.enabled

    def steer_player_jet(self, dy: float = 0.0, dz: float = 0.0, scale: float = 1.0):
        if self.game_status != "playing":
            return
        jet = self._player_jet()
        if jet is None:
            return
        vx, vy, vz = jet.velocity
        vec = np.array([vx, vy + dy, vz + dz], dtype=np.float32)
        speed = float(np.linalg.norm(vec))
        if speed < 1e-6:
            vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            speed = 1.0
        speed = float(np.clip(speed * scale, 0.5, 6.0))
        vec = vec / max(float(np.linalg.norm(vec)), 1e-6) * speed
        jet.velocity = (float(vec[0]), float(vec[1]), float(vec[2]))

    def toggle_player_jet(self):
        if self.game_status != "playing":
            return
        jet = self._player_jet()
        if jet is not None:
            jet.enabled = not jet.enabled
            self.last_event = "jet on" if jet.enabled else "jet off"

    def pulse_player_jet(self, strength: float = 1.5):
        jet = self._player_jet()
        body_pos = self.primary_body_position
        if jet is None or body_pos is None:
            self.last_event = "pulse unavailable"
            return
        if self.game_status != "playing":
            return
        if self.pulse_cooldown > 0:
            self.last_event = "pulse cooling"
            return
        if self.remaining_water is not None and self.remaining_water < self.pulse_cost:
            self.last_event = "not enough water"
            return
        vx, vy, vz = jet.velocity
        self.solver.apply_drag_force(float(body_pos[0]), float(body_pos[1]), float(body_pos[2]),
                                     vx, vy, vz, self.lx * 0.22, strength)
        self.emitted_particles += self.pulse_cost if self.water_budget is not None else 0
        self.pulse_cooldown = self.pulse_cooldown_frames
        self.last_event = "pulse"

    def _player_jet(self) -> Emitter | None:
        if 0 <= self.player_jet_index < len(self.emitters):
            return self.emitters[self.player_jet_index]
        return None

    def _apply_jet_assist(self):
        if self.jet_assist <= 0.0:
            return
        jet = self._player_jet()
        active = self.active_target
        body_pos = self.primary_body_position
        if jet is None or active is None or body_pos is None:
            return
        x0, y0, z0, x1, y1, z1 = active.region
        target = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=np.float32)
        vx, vy, vz = jet.velocity
        desired = np.array([max(2.4, vx), vy, vz], dtype=np.float32)
        desired[1] = float(np.clip(1.2 * (target[1] - body_pos[1]) + 0.45, 0.15, 1.8))
        desired[2] = float(np.clip(2.2 * (target[2] - body_pos[2]), -1.2, 1.2))
        speed = max(float(np.linalg.norm([vx, vy, vz])), 1.0)
        desired = desired / max(float(np.linalg.norm(desired)), 1e-6) * speed
        blend = float(np.clip(self.jet_assist, 0.0, 1.0))
        vec = (1.0 - blend) * np.array([vx, vy, vz], dtype=np.float32) + blend * desired
        jet.velocity = (float(vec[0]), float(vec[1]), float(vec[2]))

    def _emitter_count(self, emitter: Emitter) -> int:
        count = emitter.count
        if self.water_budget is not None:
            count = min(count, self.remaining_water or 0)
        return max(0, int(count))

    def _update_game_status(self):
        active_hazard = self._active_hazard()
        if active_hazard is not None:
            self.hazard_frames += 1
            self.target_frames = 0
            self.score = max(0, self.score - 1)
            self.last_event = active_hazard.name
        else:
            self.hazard_frames = max(0, self.hazard_frames - 1)

        active = self.active_target
        if active is not None and self._body_inside_region(active.region):
            self.target_frames += 1
        else:
            self.target_frames = max(0, self.target_frames - 2)

        if self._body_out_of_bounds():
            self.game_status = "lost"
            self.last_event = "plug escaped"
        elif active_hazard is not None and self.hazard_frames >= max(active_hazard.hold_frames, 1):
            self.game_status = "lost"
            self.last_event = "plug overheated"
        elif self.target_progress >= 1.0 and active is not None:
            self.score += active.score
            self.last_event = f"{active.name} complete"
            self.current_target += 1
            self.completed_targets = self.current_target
            self.target_frames = 0
            self._sync_legacy_target()
            if self.active_target is None:
                if self.max_frames is not None:
                    self.score += max(0, self.max_frames - self.frame) // 10
                if self.remaining_water is not None:
                    self.score += self.remaining_water // 100
                self.game_status = "won"
                self.last_event = "gate sealed"
        elif self.max_frames is not None and self.frame >= self.max_frames:
            self.game_status = "lost"
            self.last_event = "time out"
        elif self.water_budget is not None and self.remaining_water <= 0 and self.n_particles == 0:
            self.game_status = "lost"
            self.last_event = "out of water"

    def _body_inside_hazard(self) -> bool:
        return any(self._body_inside_region(h.region) for h in self.hazards)

    def _active_hazard(self) -> GameZone | None:
        for hazard in self.hazards:
            if self._body_inside_region(hazard.region):
                return hazard
        return None

    def _body_out_of_bounds(self) -> bool:
        pos = self.primary_body_position
        if pos is None:
            return False
        margin = 0.05 * self.cfg.dx
        return bool(pos[0] < -margin or pos[1] < -margin or pos[2] < -margin
                    or pos[0] > self.lx + margin or pos[1] > self.ly + margin
                    or pos[2] > self.lz + margin)

    def _body_inside_region(self, region: Region) -> bool:
        pos = self.primary_body_position
        if pos is None:
            return False
        x0, y0, z0, x1, y1, z1 = region
        return bool(x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1 and z0 <= pos[2] <= z1)

    def _sync_legacy_target(self):
        active = self.active_target
        self.target = None if active is None else active.region

    def _clone_soft_body(self, body):
        if hasattr(body, "clone"):
            return body.clone()
        return None

    def _reset_soft_body(self, body, initial):
        if initial is not None:
            try:
                body.reset(initial)
                return
            except TypeError:
                pass
        if hasattr(body, "reset"):
            body.reset()

    def _step_soft_body(self, body, dt: float):
        bounds = (self.lx, self.ly, self.lz)
        gravity = (self.cfg.gravity_x, self.cfg.gravity_y, self.cfg.gravity_z)
        try:
            body.step(dt, bounds=bounds, static_phi=self.static_phi, dx=self.cfg.dx, gravity=gravity)
        except TypeError:
            body.step(dt)
