# High-Level Research Plan: TinyML GW-Trigger (HD Tier)

## 1. Experimental Variables
*   **Independent Variable:** Model Bit-Precision (32-bit, 8-bit, 4-bit).
*   **Dependent Variable:** SNR-threshold recovery and Inference Latency (ms).
*   **Control:** Data Whitening parameters and FAR (False Alarm Rate) fixed at $10^{-4}$.

## 2. Milestone Timeline

| Week | Phase | Key Activity | HD Criteria Evidence |
| :--- | :--- | :--- | :--- |
| **1-2** | **Data Synthesis** | Inject mock chirps into O3 Noise using `PyCBC`. | **Understanding:** Matching physical signal priors. |
| **3-4** | **Base Training** | Optimize a 1D-ResNet for peak AUC on Float32. | **Technical Depth:** Use of Residual networks. |
| **5-6** | **Quantization** | PTQ implementation & Signal-Loss profiling. | **Analysis:** Identifying "Signal Clipping" errors. |
| **7-8** | **Distillation** | Hardware-Aware Training (Student-Teacher). | **Creativity:** Advanced optimization technique. |
| **9-10**| **Deployment** | Profiling on Rosella/ARM Emulator. | **Results:** Empirical hardware benchmarks. |
| **11-12**| **Synthesis** | Final Paper & Pareto Curve Generation. | **Presentation:** High-quality visual data plots. |

## 3. Success Indicators
*   **Primary:** Model size < 100KB while maintaining 90% sensitivity for SNR > 10.
*   **Secondary:** Documentation of a "Quantization-SNR Correlation Map" to advise future mission designers.

### 1. What you DO need to understand (The "Functional" Physics)

You need to understand the **data's behavior**, not its mathematical derivation.

- **The Signal (The "Chirp"):** You need to know that a Gravitational Wave looks like a "chirp"—a wave that increases in frequency and amplitude over time. To an ML model, this is just a **pattern-matching task** in a 1D time series.
    
- **The Noise (The "Glitches"):** You need to know that the detectors are sensitive to trucks driving by, lightning, and sensor hiccups. These are your "False Positives." Your ML research is about telling the difference between a "Black Hole merger" and a "Truck driving past the lab."
    
- **SNR (Signal-to-Noise Ratio):** This is your most important metric. You don't need to derive it, but you need to know that a "Low SNR" means the signal is buried deep in noise and is very hard for a quantized (low-precision) model to "hear."
    

### 2. What you can IGNORE (The "Deep" Physics)

- **General Relativity:** You don't need to know how space-time curves.
    
- **Interferometry:** You don't need to know how the lasers or mirrors in the LIGO detector work.
    
- **Astrophysics:** You don't need to know the mass of the black holes or how they formed. To you, they are just "Signal Source A."
    

### 3. The "Bridge": Pre-processing

The most "Physics-heavy" part of your project will be **Data Pre-processing**, but you will use standard Python libraries (GWpy or PyCBC) to do it. You will need to understand two concepts:

- **Whitening:** This flattens the noise so the ML model can see the signal.
    
- **Bandpassing:** This filters out frequencies where we know there is no signal (like the 60Hz hum from power lines).
    
- Note: You just need to know what these do to the data, not the calculus behind them.
    

### 4. How to leverage this for an HD

In a COMP4450 proposal, demonstrating "exceptional understanding" (to get >80%) actually means showing you know **why the physics makes the computing hard.**

**The "HD" argument looks like this:**

> "I understand that gravitational wave data is non-stationary and stochastic. Unlike an audio file where 8-bit quantization is fine because the human ear is forgiving, an astrophysical signal is mathematically precise. My research is exceptional because it investigates whether 'Rounding Errors' in a low-spec CPU act as artificial noise that 'hides' the black hole merger."

---

## 5. Conceptual Stack: Signal-Preserving TinyML

The literature review (Wasif 2026, COMP4450) identifies an "empty region" in the GW ML literature: no published work simultaneously delivers astrophysical sensitivity, embedded efficiency, and noise robustness. Filling that region cleanly probably requires intervention at four different layers of the stack. The ideas below are not all to be implemented in the COMP4450 timeline. ASMP and PC-KD are the MVP; the other two are research debt to carry forward.

### 5.1 The four interlocking ideas

| Layer | Concept | Role | Priority |
| :--- | :--- | :--- | :--- |
| Architecture | **ASMP** — Astrophysically Segmented Mixed-Precision | *Which* layers get which bit-width, mapped to waveform phase | **Primary contribution** |
| Loss function | **PC-KD** — Phase-Coherence-Weighted Knowledge Distillation | *How* the quantised student is trained | **Primary contribution** |
| Training procedure | **AQT** — Adversarial Quantisation Training | Quantisation noise treated as a structured adversarial perturbation to defend against | Adopt as a method |
| Diagnostic | **TMQ** — Topological Manifold Quantisation | Persistent-homology audit of whether the chirp manifold survives compression | Stretch goal / follow-up |

ASMP + PC-KD compose because they intervene at orthogonal levels: ASMP says *where in the network* precision must be preserved, PC-KD says *how the training objective enforces it*. AQT layers on as the training procedure. TMQ would be a diagnostic to validate that the resulting model has not lost the structure of the waveform family.

### 5.2 Astrophysically Segmented Mixed-Precision (ASMP)

**Problem.** Uniform INT8 quantisation across a 1-D CNN treats every layer as if it carries the same kind of information. A compact-binary coalescence waveform does not work that way. It evolves from a long, low-amplitude, low-frequency *inspiral* (where phase coherence carries most of the SNR) through a short, high-amplitude *merger* and *ringdown* (where amplitude dominates). Quantisation noise hurts these regimes differently. INT8 rounding error in the merger layers is fine; INT8 rounding error in the inspiral layers shatters the phase information the matched-filter analogue relies on.

**Proposition.** Map the network's bit-width schedule to the temporal phase of the input waveform. Early convolutional layers, which have a receptive field that sees the inspiral, are scheduled at INT16 or FP16. Late layers, which see the merger/ringdown, are aggressively quantised to INT8 or INT4. The result is a network whose *precision schedule mirrors the waveform's information-density schedule*.

**Why it is novel.** The mixed-precision literature outside physics (Chaturvedi et al. 2022 on TensorRT; Lin et al. 2020 on MCUNet) chooses bit-width by hardware constraints or by activation-statistics calibration. ASMP chooses bit-width by *physics*. That is the move the literature has not yet made.

**Honest caveat.** The strong form — *"bit-width can be derived from the post-Newtonian expansion of the waveform"* — is overclaim. PN expansion gives amplitude and phase as functions of time; it does not directly give a layer-by-layer precision schedule. The defensible form is: bit-width allocation is *motivated by* the PN information-density profile of the waveform and *calibrated empirically* by sweeping per-layer precision and measuring the SNR-loss contribution. State it that way in writing.

### 5.3 Implementation sketch

1. **Receptive-field audit.** For each layer of the baseline 1-D ResNet, compute the receptive field in samples, and map that back to a window of the input waveform in seconds. Identify which layers' receptive fields fall predominantly in the inspiral, merger, or ringdown regions for a representative waveform set.
2. **Per-layer sensitivity sweep.** Holding all other layers at FP32, quantise one layer at a time to INT16, INT8, INT4. Record SNR-loss against a held-out injection set across BBH and BNS signal classes. The output is a *per-layer sensitivity profile* — empirical evidence for which layers need precision protection.
3. **Schedule synthesis.** Combine (1) and (2) into a precision schedule: layers whose receptive field is inspiral-dominated *and* whose per-layer sensitivity sweep shows a steep drop at INT8 get INT16 (or FP16). Everything else gets INT8.
4. **Joint training under the schedule.** Train with FakeQuant operators at the prescribed precision per layer. This is where PC-KD plugs in: the loss function is PC-KD, the architecture is ASMP, and the training procedure is (optionally) AQT.
5. **Deployment.** Mixed-precision kernels are awkward on TinyML targets. TensorFlow Lite Micro supports INT8 natively; INT16 is supported but slower, and INT4 only via vendor extensions (CMSIS-NN partially, Helium on Cortex-M55 better). The implementation will likely need a hand-rolled kernel for the INT16 → INT8 boundary, or a fallback to INT8 everywhere with selectively per-channel scaling on the inspiral layers as a poor man's substitute.

### 5.4 Open questions and risks

- **Does the receptive-field-to-waveform-phase mapping actually hold for a deep 1-D ResNet?** Residual connections and pooling layers make the mapping less clean than for a vanilla CNN. Worth checking on the actual architecture before committing to ASMP framing.
- **Cross-mass-range generalisation.** The receptive-field-to-phase mapping changes with chirp mass (heavier binaries have shorter inspirals). A single ASMP schedule may not be optimal across the full target mass range. Either restrict the COMP4450 scope to a fixed mass range, or treat the schedule as a function of expected chirp mass and bake that into the model card.
- **Hardware support for INT16/INT8 mixed kernels.** Confirm on the ANU Rosella / Cortex-M target before locking the schedule. If the hardware only supports uniform INT8 efficiently, ASMP degenerates to "per-channel scaling" rather than "per-layer bit-width", which is a weaker claim.
- **Acronym clash.** "ASMP" is a common-enough abbreviation in other fields (asymmetric multi-processing in operating systems, for instance). On first use, expand it in full; do not rely on the acronym alone.

### 5.5 Where ASMP slots into the existing milestone timeline

| Existing milestone | ASMP touchpoint |
| :--- | :--- |
| Weeks 3–4 (Base Training) | Run the receptive-field audit on the trained FP32 model. No code yet, just a layer-by-layer table. |
| Weeks 5–6 (Quantization) | This is the per-layer sensitivity sweep. The output is the data needed to *design* the ASMP schedule. Don't deploy ASMP yet; just generate the schedule. |
| Weeks 7–8 (Distillation) | Train under ASMP + PC-KD jointly. This is the core experiment of the project. |
| Weeks 9–10 (Deployment) | Implement the INT16/INT8 mixed kernels for the target. This is where hardware support gets tested honestly. |
| Weeks 11–12 (Synthesis) | The Pareto curve becomes a *family* of curves under different ASMP schedules. The headline result is the schedule that lands inside the empty region of Figure 2 from the literature review. |

### 5.6 Forward notes on PC-KD, AQT, TMQ

- **PC-KD** is documented in the literature review (§4.3, Equation 1). Implementation question to settle early: compute the phase-coherence weight in time-domain matched-filter pre-pass, or in a wavelet-coherence domain. Wavelet coherence handles non-stationary chirps more naturally but is more expensive at training time. Probably acceptable, since the weight only needs to be computed once per training sample.
- **AQT** is not novel as a method (existing ML literature on adversarial robustness of quantised networks). Adopt it as a *training procedure* and cite the prior work honestly. The novelty is the application domain, not the method.
- **TMQ** is high-risk. The connection between topological preservation of the latent space and SNR preservation is not established and needs to be shown empirically before TMQ becomes a load-bearing claim. Defer to a follow-up paper unless an experiment in weeks 5–6 shows that simpler diagnostics (FFT-domain reconstruction error of the chirp under quantisation) are insufficient.
