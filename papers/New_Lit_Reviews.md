# Literature Reviews: TinyML and Gravitational Wave Detection

This document summarizes 13 key papers and topics relevant to the development of signal-preserving TinyML for low-latency gravitational wave (GW) triggers, specifically addressing the challenges of quantization-induced SNR degradation and computational efficiency.

---

### 1. Marx et al. (2024) - Aframe Pipeline
**Key Technical Findings:**
Marx et al. introduced **Aframe**, a machine-learning-based pipeline designed for the real-time detection of Binary Black Hole (BBH) mergers. Utilizing a modified **ResNet34** architecture, the pipeline achieves millisecond-scale inference and a total system latency of approximately 1 second. It leverages the **NVIDIA Triton Inference Server** and **TensorRT** for high-throughput processing on GPUs, matching the sensitivity of state-of-the-art matched-filtering searches in re-analyses of LIGO O3 data.

**Connection to "Signal-Preserving TinyML":**
Aframe demonstrates the feasibility of moving away from expensive template banks toward deep learning for real-time detection. However, its reliance on heavy GPU infrastructure (TensorRT) highlights the gap that TinyML must bridge: achieving similar sensitivity at the extreme edge (microcontrollers/FPGAs) where TensorRT-level optimizations must be translated into hardware-native quantization (INT8/FP16) without losing the "Aframe-level" SNR performance.

---

### 2. Mohith Niranjen et al. (2025) - Lightweight-Wavenet
**Key Technical Findings:**
Niranjen et al. proposed a hybrid architecture combining **dilated convolutions** (WaveNet-style) with **LSTM** layers for denoising and anomaly detection in GW data. The model achieved a high reconstruction fidelity with a PSNR of 33.34 dB and a low RMSE of 0.00228. The "lightweight" design is specifically optimized for memory-constrained environments, targeting the identification of signals buried in non-stationary detector noise.

**Connection to "Signal-Preserving TinyML":**
The focus on "lightweight" hybrid structures directly supports the TinyML goal of minimizing the memory footprint. The use of dilated convolutions allows for a large receptive field with fewer parameters, which is a critical strategy for mitigating quantization errors: by preserving long-range temporal dependencies in the architecture, the model remains robust to the precision loss (quantization noise) that typically degrades SNR in smaller, shallower networks.

---

### 3. AresGW Project (2024/2025) - ResNet for O3 Discovery
**Key Technical Findings:**
The AresGW project utilized a **54-layer 1D Deep Residual Network (ResNet)** to discover **8 new candidate BBH events** in the LIGO O3 dataset that were missed by traditional pipelines. Key innovations include a **Deep Adaptive Input Normalization (DAIN)** layer to handle non-stationary noise and a **curriculum learning** strategy that trains the model on progressively lower SNR signals.

**Connection to "Signal-Preserving TinyML":**
AresGW proves that deep learning can outperform matched-filtering in specific regimes. The connection to signal preservation lies in the **DAIN layer** and **curriculum learning**; these techniques ensure the model learns the "shape" of the signal in the presence of noise. For TinyML, this suggests that advanced normalization and training strategies are essential to maintain the integrity of "hidden" signals when moving to low-bitwidth precision.

---

### 4. Wu et al. (2025) - FPGA Edge Devices for Quantum Resolution
**Key Technical Findings:**
Wu et al. implemented a machine learning-based **Quantum State Tomography (QST)** pipeline on an **AMD Xilinx ZCU104 FPGA**. The system achieved a latency of **2.94 ms** (a 10x improvement over GPUs) while maintaining 99% fidelity. This work targets the real-time monitoring of "squeezed vacuum states" used in LIGO to surpass the Standard Quantum Limit.

**Connection to "Signal-Preserving TinyML":**
This is a direct application of TinyML on FPGA "at the edge" of the detector. By achieving microsecond-scale latency for quantum state optimization, it demonstrates that hardware-accelerated ML can preserve the "quantum resolution" of the detector. The high fidelity maintained during FPGA deployment is a benchmark for how quantization (on FPGAs) can be managed to ensure no loss in the underlying physical signal's sensitivity.

---

### 5. Rocha & Wendell (2024/2025) - Topological Data Analysis (TDA)
**Key Technical Findings:**
Rocha developed a framework using **Persistent Homology** (Vietoris-Rips filtrations) to detect GW "chirps" in extremely noisy environments (SNR 2-10). By mapping 1D time-series data to higher-dimensional point clouds, the TDA approach extracts "shape" features that are robust to Gaussian noise and detector glitches.

**Connection to "Signal-Preserving TinyML":**
TDA provides a mathematically grounded way to preserve signal topology regardless of amplitude. In a TinyML context, TDA-based features could be used as a "robust front-end" that is less sensitive to the rounding errors introduced by quantization than traditional spectral or time-domain features, effectively preserving the SNR in precision-constrained environments.

---

### 6. Bacon et al. (2022) - Dilated Convolutional Autoencoder
**Key Technical Findings:**
This work employs a **1D Dilated Convolutional Autoencoder (DAE)** for denoising BBH signals. A standout feature is the inclusion of **aleatoric uncertainty estimation**, allowing the model to provide a confidence measure alongside the denoised waveform. It successfully distinguishes astrophysical signals from transient noise "glitches."

**Connection to "Signal-Preserving TinyML":**
The use of autoencoders for denoising is a prerequisite for signal preservation in TinyML. By estimating uncertainty, the model can signal when quantization-induced noise is overwhelming the physical signal, providing a safety mechanism for low-latency triggers that prevents false alerts caused by precision loss.

---

### 7. Huerta et al. (2022) - TensorRT Optimization
**Key Technical Findings:**
Huerta et al. demonstrated that an ensemble of CNNs optimized with **NVIDIA TensorRT** could process one month of LIGO data in just 50 seconds on A100 GPUs, achieving a 3x speedup. The optimization involved layer fusion and precision calibration without compromising the detection sensitivity of known BBH mergers.

**Connection to "Signal-Preserving TinyML":**
While TensorRT is a high-level GPU tool, this paper serves as the "gold standard" for **Precision Calibration**. It proves that mixed-precision (FP16/INT8) and layer fusion—core tenets of TinyML—can be applied to GW detection without sacrificing astrophysical reach, provided the calibration is handled with signal-preservation in mind.

---

### 8. Mobilia et al. (2026) - Hybrid Models for Precessing Binaries
**Key Technical Findings:**
Mobilia et al. proposed a **hybrid search architecture** that feeds coarse matched-filtering SNR time-series into a CNN. This model is specifically designed to handle **precessing binaries**, where traditional template banks are computationally prohibitive. The CNN learns to recognize the complex modulations of spin-precession and eccentricity.

**Connection to "Signal-Preserving TinyML":**
Precessing signals are highly complex and easily "mangled" by quantization. The hybrid approach preserves the physical "anchors" of the signal (via matched-filtering) while using the ML component for pattern recognition. This "physics-guided" TinyML approach is a key strategy for maintaining SNR: keep the most sensitive parts of the calculation in higher precision or via traditional methods, while quantizing the classifier.

---

### 9. Krastev (2020) - CNNs for BNS Mergers
**Key Technical Findings:**
Krastev pioneered 1D CNNs for the real-time detection of **Binary Neutron Star (BNS)** mergers. The model performs a 3-way classification (BNS, BBH, Noise) and is designed for the sub-second latency required for multi-messenger astronomy alerts.

**Connection to "Signal-Preserving TinyML":**
BNS signals are longer and weaker than BBH signals, making them much more susceptible to quantization-induced SNR degradation. Krastev’s work highlights the need for architectures that can integrate over long time-windows (tens of seconds) in real-time, which is a primary challenge for TinyML devices with limited RAM.

---

### 10. LVK Collaboration (2025) - O4 Latency Requirements
**Key Technical Findings:**
The LVK Collaboration achieved median alert latencies of **~30 seconds** for standard alerts and **-3.1 seconds (Early Warning)** for BNS mergers during the O4 run. The infrastructure relies on rapid automated pipelines (GstLAL, PyCBC Live) to enable multi-messenger follow-up.

**Connection to "Signal-Preserving TinyML":**
The requirement for "negative latency" (detection before merger) is the ultimate driver for TinyML. To push latencies even lower (sub-second), the analysis must move from the central server to the "edge" (the detector site). This transition necessitates "Signal-Preserving TinyML" to ensure that the rapid alerts are based on high-fidelity signal reconstruction, not quantization-induced artifacts.

---

### 11. Scientific Co-Design (2026) - Knowledge Distillation for TinyML
**Key Technical Findings:**
This framework explores the use of **Knowledge Distillation (KD)** to compress large "Teacher" models (e.g., Aframe, ResNet-54) into "Student" models small enough (<1MB) for microcontrollers. Research indicates that distilled models can retain over 95% of the teacher's sensitivity while achieving 90-96% size reductions.

**Connection to "Signal-Preserving TinyML":**
KD is the "software side" of signal preservation. By training the student to mimic the teacher's soft probabilities and internal feature maps, the student preserves the "astrophysical knowledge" of the larger model even when its own weights are heavily quantized, mitigating the SNR loss typically seen in simple post-training quantization.

---

### 12. Quantization for Eccentric Binaries (2024) - QAT for Early-Warning
**Key Technical Findings:**
Recent 2024 studies (e.g., Martins et al.) focus on **Quantization-Aware Training (QAT)** for deploying eccentric GW models on FPGAs. QAT simulates quantization noise during training, allowing the model to adapt its weights to INT8 or lower precision. This approach improved detection accuracy from ~66% (with standard quantization) to ~76% (with QAT).

**Connection to "Signal-Preserving TinyML":**
QAT is the primary technical solution for **Quantization-Induced SNR Degradation**. By making the model "aware" of its own precision constraints, QAT preserves the signal-to-noise ratio in the final hardware implementation, enabling the microsecond-latency triggers needed for eccentric binary early-warning systems.

---

### 13. PINNs in Quantization (2023) - Gradient Errors in SciML
**Key Technical Findings:**
Xu et al. (2023) identified that **Physics-Informed Neural Networks (PINNs)** are highly sensitive to "precision-induced stalls." Training in FP32 often fails to resolve high-frequency PDE residuals that FP64 captures. Furthermore, quantization noise in the gradients of PINNs can lead to significant errors in the physical constraints (the "physics loss").

**Connection to "Signal-Preserving TinyML":**
Since GW detection is fundamentally a "physics-informed" task (detecting General Relativity waveforms), this work warns that simple quantization can break the physical consistency of the model. Signal-preserving TinyML must account for these **gradient errors** to ensure that the quantized model still obeys the underlying physics of the gravitational wave, preserving the "scientific SNR" over simple "classification accuracy."
