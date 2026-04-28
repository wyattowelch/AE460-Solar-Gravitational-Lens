# Synthetic Star Tracker + ADCS Module

This is a full **synthetic** star-tracker subsystem intended for the SGL OBC architecture.

It is **not** a full optical catalog-matching implementation. Instead, it provides the level of fidelity appropriate for system integration and power-aware flight software simulation:

- noisy star-tracker quaternion output
- confidence and tracked-star count
- dropout and false-star events
- gyro simulation with bias random walk
- fused attitude estimate
- PD attitude controller
- 4-wheel pyramid reaction wheel simulation
- event-driven ADCS power draw

## Files

- `include/sgl/star_tracker_sim.hpp`: synthetic star tracker
- `include/sgl/gyro_sim.hpp`: gyro simulator
- `include/sgl/attitude_filter.hpp`: tracker + gyro fusion
- `include/sgl/adcs_controller.hpp`: simple body-frame PD control
- `include/sgl/reaction_wheel_array.hpp`: 4-wheel pyramid wheel model
- `include/sgl/adcs_system.hpp`: reusable closed-loop ADCS system wrapper
- `examples/star_tracker_demo.cpp`: end-to-end example

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build . -j
./star_tracker_demo
```

This writes `star_tracker_demo.csv` with time history for:
- truth pointing error
- tracker validity/confidence
- tracked star count
- tracker power
- reaction wheel power
- total ADCS power

`ctest` also includes `test_adcs_system` for the reusable closed-loop wrapper.

## Intended integration points

The expected OBC integration is:

```text
AdcsSystem (wraps: StarTrackerSim -> AttitudeFilter -> AdcsController -> ReactionWheelArray)
                                           |
                                           +-> OBC telemetry / FDIR / power manager
```

## Next Steps

Integrate these classes into the Pi-side OBC loop so that:
- star tracker events drive ADCS loads
- ADCS loads reduce available compute power
- Jetson jobs are throttled when pointing correction is active
