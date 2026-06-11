# Soft Plug Game Prototype

The soft-body game levels are the current playable prototype for the
fluid-soft coupling track.  The player steers water jets to push an XPBD jelly
body through checkpoints, avoid hazards, and seal target gates.

## Run

```powershell
python -m coupledsim.app --scene soft_plug --res 32
python -m coupledsim.app --scene soft_slalom --res 32
python -m coupledsim.app --scene soft_rescue --res 32
```

For a quick headless smoke check:

```powershell
python -m coupledsim.app --scene soft_plug --res 4 --headless 1
```

## Controls

- `I/K`: steer the jet upward/downward
- `J/L`: steer the jet along the depth axis
- `Q/E`: decrease/increase jet speed
- `O`: toggle the jet
- `U`: pulse the soft plug using the current jet direction
- `R`: reset
- `Space`: pause
- Mouse drag: rotate camera

## Rules

- Green boxes are sequential checkpoints.
- The plug must stay in the active checkpoint long enough to pass it.
- Dim green boxes are later checkpoints.
- The red box is a drain hazard; staying there too long loses the game.
- Jet emission and pulses consume the water budget.
- Faster completion and unused water add score when the final gate is sealed.

## Levels

- `soft_plug`: two checkpoint tutorial level; avoid the central drain and seal
  the right gate.
- `soft_slalom`: three checkpoint route around static boxes/spheres, using two
  jet streams to bend the jelly through offset gates.
- `soft_rescue`: heavier jelly with a tighter water budget; lift it out of the
  lower drain and seal the upper outlet.

## Implementation Notes

- Soft bodies use an XPBD distance-constraint lattice (`XPBDSoftBody`).
- Soft-body nodes are rasterized into `solid_phi` and MAC solid velocities.
- The fluid solver uses those solid fields during pressure projection and boundary
  handling.
- Fluid velocity is sampled back onto XPBD nodes to apply drag feedback.
- `KinematicSoftBody` remains as a small test/proxy object, but gameplay levels
  now use the real XPBD soft body.

## Checks

Current lightweight checks cover:

- Dynamic solid field shape and nonzero moving-boundary velocity.
- MAC-grid velocity sampling.
- Fluid-to-soft drag force feedback.
- Checkpoint scoring, hazard loss, water budgeting, jet toggling, pulse cooldown,
  missing soft body / missing jet protection, and render-frame validity.
- All three XPBD game levels build and render at low resolution.
- A slow optional smoke test can step all soft game levels:
  `COUPLEDSIM_RUN_SLOW=1 python -m pytest tests/test_dynamic_coupling.py -q`.

Run:

```powershell
python -m pytest tests/test_dynamic_coupling.py -q
python -m compileall -q src/coupledsim tests/test_dynamic_coupling.py
```
