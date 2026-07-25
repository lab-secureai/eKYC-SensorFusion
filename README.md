# Attestation and Sensor Fusion for Detecting Payload Injection in eKYC

This repository contains the simulation code, synthetic benchmark, and
experimental results for detecting pre-signing biometric payload injection
in Electronic Know Your Customer (eKYC) systems.

The proposed verification architecture combines:

- remote device and application attestation;
- nonce-based flash-response liveness;
- IMU–face motion consistency;
- latency, frame-rate, and timestamp-skew analysis;
- logistic-regression-based score fusion.

The implementation accompanies the study:

> **A Simulation-Based Evaluation of Attestation and Sensor Fusion for
> Detecting Capture-Layer Payload Injection in eKYC Systems**

## Overview

Modern eKYC systems commonly rely on face verification, liveness detection,
software protection, and device attestation. These mechanisms can detect
many presentation, replay, and application-layer attacks.

However, a stronger attacker may inject manipulated biometric content into
the acquisition pipeline before the final payload is signed. In this case,
the server may receive a payload containing apparently valid timestamps,
attestation evidence, and a hardware-backed signature, even though the
biometric content did not originate from a live physical user.

This repository evaluates a multi-layer defense in which attestation is
treated as one signal among several physical-consistency signals.

## Threat Model

The main threat is pre-signing biometric payload injection.

The attacker may partially control the client-side acquisition path,
including:

- a virtual camera stream;
- a modified camera interface;
- a compromised capture layer;
- an emulated or manipulated sensor stream;
- a real-time synthetic face pipeline.

The attacker is assumed not to break standard cryptographic primitives,
forge the server nonce, derive the protected signing key, or compromise the
verification server.

The repository contains defensive simulation and evaluation code only. It
does not contain offensive hardware-injection tooling.

## Proposed Feature Set

Each simulated eKYC session is represented using multiple verification
signals.

| Feature | Description |
|---|---|
| `attestation_score` | Simulated device and application integrity evidence |
| `attestation_pass` | Binary attestation decision |
| `flash_corr_zero_lag` | Zero-lag correlation between the challenge and facial brightness |
| `flash_corr_best_lag` | Maximum lag-aware flash-response correlation |
| `flash_best_lag_frames` | Estimated response delay in frames |
| `imu_face_corr` | Correlation between device motion and visual face motion |
| `latency_ms` | End-to-end challenge-response latency proxy |
| `fps` | Effective frame rate |
| `timestamp_skew_ms` | Temporal mismatch between video and sensor streams |

The final fusion model estimates an attack probability from the combined
feature vector.

## Simulated Session Categories

The benchmark contains 4,000 simulated sessions.

| Session type | Number of sessions |
|---|---:|
| Genuine | 1,000 |
| Software repackaging | 1,000 |
| Replay attack | 1,000 |
| Hardware or capture-layer injection | 1,000 |
| **Total** | **4,000** |

The simulated attack distributions intentionally overlap with genuine
sessions to avoid unrealistically perfect separation.

The hardware-injection category includes adaptive variants that may:

- partially respond to flash challenges;
- partially reproduce IMU consistency;
- introduce relay-like timing behavior;
- preserve valid or near-valid attestation evidence.

## Repository Structure

```text
.
├── notebooks/
│   ├── ekyc_adversarial_experiment_executed.ipynb
│   └── ekyc_adversarial_experiment_clean.ipynb
├── src/
│   └── ekyc_experiment.py
├── data/
│   └── synthetic_ekyc_sessions.csv
├── results/
│   ├── overall_metrics.csv
│   ├── confusion_matrix.csv
│   ├── operating_points.csv
│   ├── per_attack_type.csv
│   ├── ablation_results.csv
│   ├── ood_metrics.csv
│   ├── dataset_validation.json
│   └── result_manifest.csv
├── requirements.txt
├── LICENSE
└── README.md
