# High-level architecture

## Pi responsibilities
- command & data handling
- mission mode and autonomy
- subsystem state aggregation
- event-driven power budgeting
- persistent storage and downlink queue ownership
- Jetson job authorization

## Jetson responsibilities
- coarse reconstruction
- ROI scoring
- adaptive refinement
- data reduction and product generation

## Transport
A framed TCP protocol is used:
- 8-byte header length
- 8-byte payload length
- textual key-value header block
- optional binary payload

This keeps the prototype dependency-free and easy to debug while remaining deployable to Pi+Jetson hardware.

## Power model
Subsystems expose dynamic loads based on sensing and actuation. The Pi computes

P_compute_allow = alpha * max(0, P_source - P_noncompute - P_reserve)

Only then are Jetson jobs allowed.
