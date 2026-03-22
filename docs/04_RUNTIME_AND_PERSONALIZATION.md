# Runtime And Personalization

This document covers live scoring, streaming, playback, setup, and user profile personalization.

## Runtime Contract

Key constants are defined in `src/runtime/constants.py`.

Current canonical values:

- runtime sample rate: `250 Hz`
- runtime scoring window: `2.0` seconds
- runtime stride: `0.25` seconds
- canonical Cyton runtime channels: `Fp1,Fp2,C3,C4,P3,P4,O1,O2`
- setup/calibration channels: `F7,F3,F4,F8,P7,P8,O1,O2`

The code treats the runtime stream as fixed-shape EEG chunks that can be adapted, buffered, scored, and postprocessed in a consistent way regardless of where they came from.

## Main Runtime Modules

- `src/runtime/contracts.py`: runtime chunk dataclasses and transport-friendly structures
- `src/runtime/eeg_frame.py`: frame container used by the runtime engine
- `src/runtime/window_buffer.py`: sliding window emission
- `src/runtime/adaptation.py`: adaptation layer and default profile
- `src/runtime/baseline.py`: baseline inference wrapper for runtime use
- `src/runtime/decision_engine.py`: high-level decision loop
- `src/runtime/postprocessor.py`: smoothing and state postprocessing
- `src/runtime/engine.py`: shared streaming engine used by desktop interfaces
- `src/runtime/router.py`: source routing abstraction
- `src/runtime/stream_transport.py`: JSON and UDP transport helpers

## Source Types

The runtime stack can operate from three source families:

- synthetic: generated in-process by `SyntheticSource`
- playback: read from saved `.npz`, `.npy`, or `.csv` recordings by `PlaybackSource`
- live Cyton: streamed from BrainFlow with `CytonSource`

Implementation lives in `src/runtime/sources/`.

## Unified Streaming CLI

Entry point:

```bash
python -m src.runtime.run_engine --source synthetic --chunks 20
python -m src.runtime.run_engine --source playback --input artifacts/playback/demo.npz
python -m src.runtime.run_engine --source cyton --serial-port COM5
```

Key flags:

- `--source`
- `--chunks`
- `--warmup-seconds`
- `--chunk-seconds`
- `--concentration`
- `--stress`
- `--seed`
- `--input`
- `--serial-port`
- `--artifacts-root`
- `--calibration-path`

The runtime CLI validates that the loaded artifacts can score the expected runtime channel layout before streaming decisions.

## Streaming Transport

`src/runtime/stream_transport.py` supports:

- compact JSON encoding for chunks
- UDP-safe datagram packing and unpacking
- channel-name discovery for saved scorers

Two related CLIs are also present:

```bash
python -m src.runtime.stream_generate --transport udp --chunks 100
python -m src.runtime.stream_classify --transport udp --port 8765
```

These support generator-to-classifier streaming over UDP or TCP, with optional UDP output forwarding to an external consumer.

## Playback Tooling

Useful playback commands:

```bash
python -m src.runtime.demo_playback --output artifacts/playback/demo_playback.npz --duration-sec 20
python -m src.runtime.run_playback --input artifacts/playback/demo_playback.npz
python -m src.runtime.export_playback_formats --input artifacts/playback/demo_playback.npz --output-dir artifacts/playback/export
```

Related modules:

- `src/runtime/playback_tools.py`
- `src/runtime/run_playback.py`
- `src/runtime/export_playback_formats.py`
- `src/runtime/playback_interface.py`

## Desktop Interfaces

### Setup interface

`src/runtime/setup_interface.py` provides a guided Tk desktop UI.

Current behavior:

- user picks a setup source: `generator` or `cyton`
- user provides serial port, user ID, and runtime artifacts root
- the UI animates the current requested state
- the controller emits concentration, stress, quality, and stability updates
- successful completion saves a user profile for later personalization

### Engine interface

`src/runtime/engine_interface.py` is the runtime monitor UI.

It currently supports:

- UDP input
- direct Cyton input
- artifact root selection
- optional calibration profile selection
- live metrics and score history
- log output inside the desktop window

### Other interfaces

- `src/runtime/playback_interface.py`: playback and export UI
- `src/runtime/generator_interface.py`: generator-side UI surface

## Setup And Calibration Flow

The calibration controller lives in `src/runtime/calibration_controller.py`.

Default protocol phases:

- `concentration_high`
- `concentration_low`
- `stress_high`
- `stress_low`
- `detection_check`

Each phase defines:

- animation mode
- minimum quality
- required stability streak
- accepted-window target
- maximum allowed windows
- optional concentration and stress thresholds
- optional adaptive intensity ramps

The controller can build setup sources from either:

- a synthetic generator path
- a live Cyton path

## User Profile Model

Persistence and personalization live in `src/runtime/user_profile.py`.

Saved profile fields include:

- `user_id`
- `profile_id`
- `created_at_utc`
- `source_type`
- `channel_names`
- focus and stress anchors
- personalization threshold
- smoothing alpha
- phase summaries
- metadata

Each phase summary captures:

- accepted status
- accepted and total window counts
- stability score
- quality mean
- concentration quantiles
- stress quantiles
- feature anchors
- optional modifiers and notes

## Personalization Logic

Raw runtime outputs are normalized between low and high anchors derived from accepted calibration phases. The profile then maps those values into a quadrant-style state:

- `focused`
- `stressed`
- `strained`
- `idle`

This lets the runtime layer report both raw model outputs and user-relative states.

## Runtime Artifact Layout

The runtime bundle normally lives under `artifacts/runtime_v1/`.

Important contents:

- `concentration/`
- `stress/`
- `runtime_bundle_summary.json`
- `users/<user_id>/profile.json`
- `users/<user_id>/profile.pkl`
- `users/<user_id>/sessions/<session_id>/phase_*.npz`
- `users/<user_id>/sessions/<session_id>/summary.json`

## Runtime Compatibility Rule

The runtime code checks whether the loaded saved preprocessors require channels that are missing from the incoming runtime stream. If they do, runtime startup fails early and tells the caller to retrain or provide a compatible artifact bundle.
