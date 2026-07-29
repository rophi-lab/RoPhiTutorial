# Flexiv Bridge

This directory contains the first-pass C++ bridge for a Flexiv arm.

The intended architecture is:

1. Python controllers publish `joint_ctrl_t` over LCM.
2. This bridge subscribes to the control channel in C++.
3. The bridge runs the arm-side loop at 1 kHz.
4. The bridge publishes `joint_meas_t` back over LCM.

The initial scaffold runs in `mock_mode: true` by default so the LCM plumbing,
config loading, joint slicing, and safety fallback can be exercised before the
Flexiv RDK calls are connected.

## Build

```bash
./scripts/build_flexiv_bridge.sh
```

## Run

```bash
./scripts/run_flexiv_bridge.sh
```

## Key Notes

- This bridge is arm-only and uses the new `flexiv_arm` config namespace.
- It expects a 7-DoF arm command and publishes a 7-DoF arm measurement.
- The real Flexiv RDK hookup belongs in `src/FlexivClient.cpp`.
- The LCM serialization is implemented locally to match the repo's existing
  `joint_ctrl_t` and `joint_meas_t` wire formats without requiring C++ codegen.
