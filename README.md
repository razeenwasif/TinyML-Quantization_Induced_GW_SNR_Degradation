# TinyML: Quantization-Induced SNR Degradation

Toward Signal-Preserving TinyML for Gravitational Wave Triggers.

## Overview
This research explores the transition of Gravitational Wave (GW) detection from high-performance GPU clusters to ultra-low-power edge computing (TinyML). As GW observatories move toward space-based networks (e.g., LISA) and CubeSat-based detectors, the "Data Deluge" necessitates local, real-time processing of astrophysical events to overcome downlink capacity bottlenecks.

## Research Problem
While deep learning models have matched classical matched-filtering sensitivity, their reliance on high-precision (Float32) numerical representations is incompatible with the strict SRAM (256KB–512KB) and power constraints of embedded microcontrollers. Compressing these models to 8-bit integers (INT8) risks destroying the phase-coherence of "chirp" signals, fatally degrading the Signal-to-Noise Ratio (SNR).

## Objectives
- **Empirical Benchmarking:** Quantify how non-linear bit-width reduction (INT8) interferes with GW signal phase-coherence.
- **Signal-Preserving Optimization:** Test a framework that prioritizes astrophysical signal fidelity (SNR recovery) over traditional classification accuracy.
- **Hardware Deployment:** Optimize models for the ANU Rosella architecture (ARM Cortex-M) using custom C++ inference kernels.

## Methodology
The project follows a three-stage experimental design:
1.  **Data Synthesis:** Injecting mock PyCBC "chirp" signals into real LIGO O3 noise.
2.  **Optimization Tiers:** Comparing Baseline Float32 1D-CNN, Naive Post-Training Quantization (PTQ), and Quantization-Aware Training (QAT).
3.  **Evaluation:** Profiling instruction-set efficiency, RAM usage (< 128KB), and latency, targeting an SNR-Loss of < 5%.

## Key Metrics
- **SNR-Loss (Sensitivity):** Aiming for < 5% drop compared to the Float32 baseline.
- **False Alarm Rate (FAR):** Targeting 10⁻⁴ Hz for initial edge-level filtering.
- **Memory Efficiency:** Achieving an 8× reduction in footprint.
- **Latency:** Ensuring real-time viability on ARM Cortex-M hardware.
