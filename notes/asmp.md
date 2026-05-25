# ASMP — Astrophysically Segmented Mixed-Precision: Working Notes

Working document. Lives next to the Implementation plan. Refer back as the project progresses, update with findings, do not treat as finalised.

---

## 0. TL;DR

ASMP is: assign per-layer bit-width as a function of which segment of the gravitational-wave (GW) waveform each layer's receptive field covers. Inspiral-dominated layers stay at INT16 or FP16 because phase coherence carries the SNR there. Merger/ringdown-dominated layers go to INT8 or INT4 because amplitude dominates and rounding noise is cheaper. Compose with PC-KD for the loss function. Validate by per-layer sensitivity sweep before committing the schedule.

The honest framing for write-up: bit-width allocation is *motivated by* the post-Newtonian (PN) information-density profile of the waveform and *calibrated empirically* by the sensitivity sweep. Do not claim the schedule is *derived* from PN; the bridge between PN amplitude/phase and layer-precision is empirical, not analytic.

---

## 1. The core idea, restated for myself

Uniform INT8 across the network is a hardware optimisation pretending to be a signal-processing decision. It works for image classification because every pixel of a cat is more or less worth the same. It does not work for GW chirp signals because the inspiral and the ringdown carry different *kinds* of information at different *amounts* per unit time.

Inspiral: long (seconds to minutes for BNS), low amplitude, frequency sweeping slowly upward. SNR accumulates because the matched-filter integrates phase-coherent signal energy over many wave cycles. *Phase precision matters more than amplitude precision here.*

Merger and ringdown: short (~10 ms for BBH, fewer than 100 ms for BNS), high amplitude, complex spectral content. *Amplitude precision matters more than phase precision here.*

A 1-D CNN run over whitened strain has early layers whose receptive fields cover seconds of input (the inspiral) and late layers whose receptive fields cover only the recent fraction of the input window (where the merger likely sits). ASMP claims that bit-width should track this division.

That is a hypothesis, not a theorem. The job of the experiment is to test whether it holds.

---

## 2. Receptive field to waveform-phase mapping

### 2.1 Receptive field math

For a sequential stack of 1-D conv layers with kernel sizes $k_i$ and strides $s_i$, the receptive field at the output of layer $L$ in input samples is:

$$
\text{RF}_L = 1 + \sum_{i=1}^{L} (k_i - 1) \prod_{j=1}^{i-1} s_j
$$

For LIGO whitened strain at $f_s = 4096$ Hz, receptive field in seconds is $\text{RF}_L / f_s$.

Skip connections in ResNet complicate this. A residual block sees both its direct input (small RF) and the accumulated RF from the trunk. The dominant RF is the larger of the two, but the smaller branch can still carry signal. For audit purposes, report both numbers per layer.

Dilated convolutions multiply the effective kernel size by the dilation factor; if Niranjen-style dilated layers are used (which the literature review mentioned but did not commit to), the RF balloons quickly. Check this if dilation is enabled.

### 2.2 Waveform timing, leading-order PN

Time-to-merger from instantaneous gravitational-wave frequency $f$, at leading PN order:

$$
\tau(f) \approx \frac{5}{256} \left( \pi f \right)^{-8/3} \left( \frac{G \mathcal{M}}{c^3} \right)^{-5/3}
$$

where $\mathcal{M} = (m_1 m_2)^{3/5} / (m_1 + m_2)^{1/5}$ is the chirp mass.

This says: how long the inspiral lasts in the detector band depends very strongly on chirp mass. For a fixed start frequency, doubling the chirp mass roughly halves $\tau$ to the power of $5/3$, which is to say it shortens the inspiral by ~3x.

Concrete numbers (start at $f = 30$ Hz, where LIGO becomes sensitive):

- BNS, $\mathcal{M} \approx 1.2 M_\odot$: $\tau \sim 100$ seconds
- BBH 30+30, $\mathcal{M} \approx 26 M_\odot$: $\tau \sim 1$ second
- IMBH 100+100, $\mathcal{M} \approx 87 M_\odot$: $\tau \sim 0.1$ seconds

Implication for ASMP: the schedule that's optimal for BBH may not be optimal for BNS. The receptive-field-to-phase mapping must be computed *per target mass range*.

### 2.3 Worked example, BBH 30+30

Suppose a 1-D ResNet with 10 conv layers, all kernel size 9, stride 2 in the first three blocks then stride 1, fed a window of 1 second at $f_s = 4096$ Hz (4096 input samples).

| Layer | $\text{RF}$ (samples) | $\text{RF}$ (seconds) | Waveform segment covered |
|---|---|---|---|
| 1 | 9 | 0.0022 | merger/ringdown only |
| 2 | 25 | 0.0061 | late merger |
| 3 | 57 | 0.014 | late inspiral + merger |
| 4 | 121 | 0.030 | mid inspiral + merger |
| 5 | 249 | 0.061 | most of inspiral |
| 6 | 505 | 0.123 | full inspiral |
| 7 | 1017 | 0.248 | full window, inspiral-weighted |
| 8 | 2041 | 0.498 | full window |
| 9 | 4089 | 0.998 | full window |
| 10 | 4096 | 1.000 | full window |

For BBH 30+30 the inspiral is ~1 second long in a 30+ Hz band. So layers 6–10 are inspiral-dominated and layers 1–3 are merger/ringdown-only. ASMP would say: 6–10 at INT16, 1–3 at INT8, 4–5 at the boundary.

Same exercise for BNS would give a different schedule because the inspiral runs longer than any feasible input window — every layer is inspiral-dominated to some degree. ASMP for BNS may degenerate to "uniform INT16", which is interesting because it explains why Qi et al. (2024) report BNS degradation under naive uniform INT8.

### 2.4 The mapping recipe

For each target mass range:

1. Pick a representative chirp mass and a representative starting frequency.
2. Compute $\tau$, the inspiral duration in band.
3. For each layer, compute RF in seconds.
4. Tag the layer with the fraction of its RF that is inspiral-dominated vs merger/ringdown-dominated.
5. The schedule is: high precision where inspiral fraction is large, low precision where it is small.

Report this as a table in the thesis. It is the closest ASMP has to a *derived* schedule, and it is honest about the limits of the derivation.

### 2.5 TikZ sketch for the proposal / thesis figure

Figure-quality sketch of the receptive-field-to-waveform-phase mapping below. Drops into a LaTeX doc as-is, given the same TikZ preamble the Literature Review uses (`\usepackage{tikz}` plus the `shapes.geometric, arrows, positioning` libraries — though this specific figure only needs the base TikZ).

A note on the schematic. The receptive field of layer $L$ at a single output position is a window of $\text{RF}_L$ input samples. Different output positions of the same layer correspond to *different* windows shifted across the input. The figure draws each layer's RF *anchored at the merger* on the right edge, because that is the output position that decides the classification of a chirp-bearing window. Layers with small RFs (shallow) are well-suited to features that vary rapidly (the merger and ringdown waveform). Layers with large RFs (deep) integrate phase information across the inspiral. ASMP exploits exactly that.

```latex
\begin{figure}[!htbp]
\centering
\begin{tikzpicture}[
  x=11cm, y=1cm,
  font=\footnotesize
]

% ===== background regions (cover both panels) =====
\fill[blue!8]    (0,    -1.0) rectangle (0.97, 3.5);
\fill[orange!18] (0.97, -1.0) rectangle (0.99, 3.5);
\fill[red!15]    (0.99, -1.0) rectangle (1.00, 3.5);

% ===== top panel: chirp waveform =====
% Inspiral: low amplitude early, growing late, frequency increasing
\draw[blue!55!black, thick] plot[smooth, tension=0.7] coordinates {
  (0.00, 2.55) (0.05, 2.59) (0.10, 2.51) (0.15, 2.60) (0.20, 2.50)
  (0.25, 2.62) (0.30, 2.48) (0.35, 2.64) (0.40, 2.46) (0.45, 2.66)
  (0.50, 2.44) (0.55, 2.69) (0.60, 2.41) (0.65, 2.72) (0.70, 2.38)
  (0.74, 2.76) (0.78, 2.34) (0.82, 2.81) (0.85, 2.29) (0.88, 2.87)
  (0.91, 2.22) (0.93, 2.94) (0.95, 2.12)
};
% Merger: a few high-amplitude cycles
\draw[orange!70!black, thick] plot[smooth, tension=0.5] coordinates {
  (0.95, 2.12) (0.96, 3.20) (0.97, 1.85) (0.975, 3.35)
  (0.98, 1.90) (0.985, 3.20) (0.99, 2.55)
};
% Ringdown: rapid decay
\draw[red!60!black, thick] plot[smooth, tension=0.7] coordinates {
  (0.99, 2.55) (0.993, 2.75) (0.996, 2.60) (1.00, 2.55)
};

% Y-axis label for the chirp
\node[left, font=\scriptsize] at (-0.005, 2.6) {$h(t)$};

% Segment labels above the chirp
\node[blue!55!black, font=\scriptsize] at (0.45, 3.65) {inspiral};
\node[orange!70!black, font=\scriptsize] at (0.985, 3.65) {merger};
\node[red!60!black, font=\scriptsize, anchor=south west]
  at (1.005, 3.55) {ringdown};

% ===== middle: time axis =====
\draw[->] (-0.01, 1.55) -- (1.03, 1.55);
\foreach \t/\lab in {0.0/{$-1.0$}, 0.25/{$-0.75$}, 0.5/{$-0.5$},
                    0.75/{$-0.25$}, 0.97/{$0$}} {
  \draw (\t, 1.53) -- (\t, 1.57);
  \node[below, font=\scriptsize] at (\t, 1.50) {\lab};
}
\node[below, font=\scriptsize] at (0.5, 1.18) {time relative to merger (s)};

% Merger vertical guide across both panels
\draw[dashed, orange!60!black, thin] (0.97, -0.95) -- (0.97, 3.55);

% ===== bottom panel: layer receptive fields =====
% name / start_x / y / colour / bit-width label
\foreach \name/\start/\y/\col/\bits in {
  L1/0.9678/0.95/orange!75!black/INT8,
  L2/0.9639/0.75/orange!75!black/INT8,
  L3/0.9560/0.55/orange!75!black/INT8,
  L4/0.9400/0.35/orange!75!black/INT8,
  L5/0.9090/0.15/gray!55!black/{empirical},
  L6/0.8470/-0.05/blue!55!black/INT16,
  L7/0.7220/-0.25/blue!55!black/INT16,
  L8/0.4720/-0.45/blue!55!black/INT16,
  L9/0.0020/-0.65/blue!55!black/INT16,
  L10/0.0000/-0.85/blue!55!black/INT16
} {
  \draw[\col, line width=2pt, line cap=round] (\start, \y) -- (0.97, \y);
  \node[anchor=east, font=\scriptsize] at (-0.005, \y) {\name};
  \node[anchor=west, font=\scriptsize] at (1.005, \y) {\bits};
}

% Bottom-panel description
\node[font=\scriptsize, align=center] at (0.5, -1.30) {
  layer (top: shallow / small RF, bottom: deep / large RF); \\
  bar extent = receptive field anchored at the merger
};

\end{tikzpicture}
\caption{Receptive-field-to-waveform-phase mapping for a 10-layer 1-D CNN
on a 1-second input window with the merger at $t=0$. \textit{Top}: a
synthetic BBH 30+30 chirp showing inspiral (blue), merger (orange), and
ringdown (red) regions. \textit{Bottom}: each layer drawn as a horizontal
bar whose extent is the layer's receptive field projected back onto the
input window, anchored at the merger. Shallow layers (L1--L4) have receptive
fields that fit inside the merger and late ringdown; under ASMP these are
scheduled at INT8 because amplitude dominates SNR at these short timescales.
Deep layers (L6--L10) have receptive fields integrating across the
inspiral; these are scheduled at INT16 because phase coherence accumulates
SNR over many wave cycles and is sensitive to rounding noise. L5 straddles
the boundary and is assigned empirically via the per-layer sensitivity
sweep of \S3.}
\label{fig:rf-phase-mapping}
\end{figure}
```

### 2.5.1 What this figure does and does not claim

The figure is a schematic, not a derivation. Three honest framing points to keep alongside it:

1. **Anchored at the merger.** Each layer's RF bar is drawn anchored at $t = 0$ (the merger). In reality, a convolutional layer's RF tiles across the input — different output positions cover different windows. The merger-anchored view is the slice of that tiling that matters for *classifying a chirp-bearing window*.
2. **The mass-range disclaimer.** The plot is for BBH 30+30. For BNS, the inspiral runs orders of magnitude longer than any feasible input window, so every layer's RF sits inside the inspiral and ASMP degenerates toward "all INT16". For higher-mass IMBH binaries the picture inverts and more layers can drop to INT8. State the chirp mass when reproducing the figure.
3. **The L5 honest answer.** L5's RF is small enough to sit mostly in the late inspiral but large enough that a chunk of it can include the merger when the window is aligned. The "empirical" label on L5 is not a placeholder — it is the right answer until the per-layer sensitivity sweep produces evidence.

### 2.5.2 Reproducing the figure in practice

When the per-layer sensitivity sweep (§3) produces actual numbers, redraw the bottom panel with the bar colours determined by *measured* SNR-loss thresholds rather than by the receptive-field-based hypothesis. The figure then becomes empirical: bars stay coloured by the precision tier each layer was assigned, but the colour assignment is now backed by data rather than by argument. That is the version of the figure that should appear in the thesis. The schematic version above is for the *Proposal*, where the argument is what's being made.

---

## 3. Per-layer sensitivity sweep

This is the empirical step that calibrates the schedule. The output is what makes ASMP defensible.

### 3.1 Pseudocode

```python
baseline = train_fp32_model()
metrics = {}
for layer_idx in range(num_layers):
    for precision in [INT16, INT8, INT4]:
        model = copy(baseline)
        model.layers[layer_idx] = fake_quantize(
            model.layers[layer_idx], precision
        )
        # do NOT retrain; this measures the susceptibility
        # of the trained weights to precision drop
        metrics[(layer_idx, precision)] = evaluate(
            model, injection_set, metric="snr_loss_at_far"
        )
```

Note: this is the *post-training* sensitivity sweep. There is also a *training-aware* version where you retrain the model under per-layer quantisation and re-evaluate, which gives a more realistic picture of where QAT can compensate. Do both. Report the difference.

### 3.2 Choice of metric

Three candidates. Listed in increasing order of how directly they capture the science:

- **Classification accuracy on injection vs noise.** Easy to compute. Insensitive to the thing we actually care about. *Do not rely on this alone.*
- **SNR-loss at fixed FAR.** Compute the optimal SNR threshold of the FP32 baseline at FAR $= 10^{-4}$ Hz, then measure how much higher the quantised model's threshold has to be to hold the same FAR. The delta is the SNR-loss. *This is the primary metric.*
- **Distance reach at fixed FAR.** Compute the distance at which a fiducial binary just clears the threshold. Distance scales inversely with SNR, so distance reach is the astrophysical translation of SNR-loss. Use this for the headline plot.

The PROGRESS.md or thesis should report all three. The marker will want to see at least SNR-loss; the astrophysics community will want distance reach.

### 3.3 Injection set design

Coverage matters. Use:

- Mass coverage: a grid in $(m_1, m_2)$ covering the BBH range (5–80 $M_\odot$ each) and the BNS range (1–2.5 $M_\odot$ each), as separate experiments.
- SNR coverage: log-spaced from SNR = 5 (well below threshold) to SNR = 30 (well above). The interesting failure mode is at SNR 5–10, where quantisation can flip a marginal detection.
- Noise realisation coverage: at least 1000 injections per (mass, SNR) cell, in independent O3 noise segments. PyCBC has utilities for this.
- Glitch coverage: a separate set of injections in segments known to contain blip transients, drawn from the Abbott et al. (2016) catalogue. This tests robustness, not just sensitivity.

### 3.4 What "the knee" looks like

If ASMP works, the per-layer SNR-loss curve should show a clean *knee*. Layers with strong inspiral coverage should show large SNR-loss at INT8 and recover at INT16. Layers with merger-dominated coverage should be flat across INT16 → INT8 → INT4, recovering only when pushed below INT4.

If the curves are flat across all layers, ASMP has no signal — the network is not differentially sensitive to per-layer precision, and the right thing is to fall back to uniform precision plus per-channel scaling. Report this honestly and pivot the thesis story.

---

## 4. Schedule synthesis

### 4.1 Heuristics

- Default everything to INT8.
- Promote to INT16 only the layers where the sensitivity sweep shows a knee above some SNR-loss threshold (suggest: 2%).
- Consider INT4 only for layers that show essentially flat sensitivity through INT8 — the savings are real (half the memory) but the support in hardware is patchy.
- Keep the input layer at INT8 minimum, never INT4. Calibration of input quantisation parameters is fragile.

### 4.2 Number of precision tiers

Two tiers (INT8 + INT16) is the safe choice. Three (INT4 + INT8 + INT16) is more aggressive and harder to deploy. The thesis should probably commit to two and discuss three as an extension.

### 4.3 The 30% rule

If more than ~30% of layers end up at INT16, the compression benefit is largely gone. Either the model is being over-protected (lower the SNR-loss threshold for promotion to INT16) or the network architecture is not suited to ASMP (in which case the thesis story changes to "uniform INT8 + per-channel scaling is the floor"; still publishable, but not the ASMP story).

---

## 5. Joint training with PC-KD

### 5.1 FakeQuant per layer

PyTorch's `torch.ao.quantization` provides per-layer `FakeQuantize` modules. TensorFlow's QAT API supports per-layer quantisation configs. Either works.

The per-layer precision is set as a fixed schedule before training starts; it is not a learned parameter. If learned precision is needed (it is not for ASMP MVP), look at the Differentiable Quantisation literature — but that's a stretch.

### 5.2 PC-KD weight schedule

Recall PC-KD's loss:

$$
\mathcal{L}_{\text{PC-KD}}(x) = \alpha \cdot \mathrm{KL}\bigl(p_T(x) \| p_S(x)\bigr) + \beta \cdot w(x) \cdot \| f_T(x) - f_S(x) \|^2
$$

Composition with ASMP: the teacher is FP32, the student is ASMP-scheduled (mixed INT8/INT16). The feature-matching term uses penultimate-layer features. The phase-coherence weight $w(x)$ is computed from a one-time matched-filter pre-pass against a coarse template bank, cached per training sample.

Open implementation question: should $w(x)$ be applied per-sample (one weight per window) or per-time-step (a weight vector across the window)? The latter is more correct in principle — phase coherence varies *within* the window — but only matters if the feature map preserves the time axis. Penultimate-layer features in a typical classification ResNet are pooled across time, so per-sample is the realistic choice for the MVP.

### 5.3 Hyperparameters to sweep

- $\alpha$ (KL weight): 0.5 to 2.0
- $\beta$ (feature-matching weight): 1.0 to 10.0
- Temperature in the KL term: 2 to 8
- Bit-width promotion threshold (for ASMP): 1% to 3% SNR-loss

Eight-cell grid in $\alpha$ × $\beta$ is fine. Don't get lost in hyperparameter optimisation; the headline experiment is ASMP-vs-uniform, not perfect tuning.

---

## 6. Hardware deployment

This is the part that can sink the whole story if not checked early.

### 6.1 Cortex-M / CMSIS-NN audit

CMSIS-NN, as of the version pinned by TensorFlow Lite Micro in late 2025:

- INT8: fully supported for conv, depthwise conv, fully connected, pooling, activation. Fast SIMD via SMLAD on M4/M7.
- INT16: supported but slower. Conv and FC have INT16 variants; some activations don't.
- Mixed INT8/INT16 within a single layer: not natively supported. The boundary between layers is the natural place to switch.
- Mixed-bitwidth in the same network across layers: supported in principle by TFLite Micro's per-tensor quantisation params, but the inference kernels must handle each tensor at its declared bit-width. Verify by running a smoke test with one INT16 conv sandwiched between INT8 convs.

### 6.2 Helium / Cortex-M55

Cortex-M55 with the Helium MVE extension has better mixed-precision support. Vector MAC instructions can operate on 8-bit and 16-bit operands with widening accumulation. The Rosella platform spec needs to be checked: if Rosella is M55, ASMP has a clean implementation path. If Rosella is M4 or M7, the implementation is harder.

*Action item: verify which Cortex variant Rosella uses before week 7.*

### 6.3 FPGA fallback

If MCU mixed-precision is too painful, an FPGA target (Xilinx Zynq, Lattice iCE40 Ultra) gives much more flexibility — per-layer bit-width is essentially free on FPGA because each layer can be a separate hardware block. This costs the "sub-watt MCU" framing of the proposal but salvages the ASMP story.

The Qi et al. (2024) precedent makes FPGA a defensible target.

### 6.4 Fallback to per-channel scaling

If neither mixed-bitwidth nor FPGA is feasible, the weak version of ASMP is per-channel scaling within uniform INT8. The schedule then becomes: use larger per-channel scale factors on the inspiral-dominated layers (effectively reducing quantisation noise at the cost of dynamic range), smaller scale factors on the merger-dominated layers. This is a watered-down ASMP — not as interesting, but still defensible.

---

## 7. Validation methodology

### 7.1 What "ASMP wins" looks like

A claim like *"ASMP recovers $X\%$ of the FP32 sensitivity that uniform INT8 loses"* needs:

- A FP32 baseline number (the teacher).
- A uniform INT8 baseline number (the deployment-clique-style naive approach, which is what Qi et al.\ 2024 effectively did).
- An ASMP number under the same training procedure (PC-KD or plain QAT).
- All three measured on the same injection set, with the same FAR threshold, reported as both SNR-loss and distance reach.

If ASMP doesn't beat uniform INT8 + PC-KD by at least 1% distance reach at FAR $= 10^{-4}$ Hz, ASMP isn't winning. Pre-commit to that threshold before running the experiment, so the conclusion isn't post-hoc.

### 7.2 Baselines to compare against

| Baseline | Why include |
|---|---|
| FP32 teacher | The ceiling. |
| Uniform INT8, no QAT | The naive deployment. Floor. |
| Uniform INT8, with QAT | Strong baseline. The fair comparison for ASMP. |
| Uniform INT8, with PC-KD | Tests whether ASMP adds anything over PC-KD alone. |
| ASMP + plain QAT | Tests whether ASMP adds anything over PC-KD on its own. |
| ASMP + PC-KD | The proposed method. |

The 2x2 design (ASMP × PC-KD) is the right experimental structure. It cleanly separates the two contributions.

### 7.3 The Pareto curve

Two-axis plot: peak SRAM (or model size) on the x-axis, distance reach at FAR $= 10^{-4}$ Hz on the y-axis. Each baseline is a point. The FP32 teacher is in the top right. The naive INT8 is bottom left. The job of ASMP + PC-KD is to be in the top left — small *and* sensitive. If it's not, the proposal hasn't delivered.

---

## 8. Risks and decision tree

| Risk | How to detect | Fallback |
|---|---|---|
| Receptive-field-to-phase mapping is fuzzy on ResNet | Week 3 audit shows ambiguous RF assignment | Switch to plain CNN (no skip connections) for the MVP. Lose some accuracy, gain a cleaner ASMP story. |
| Per-layer sensitivity sweep shows no knee | Week 5–6 results are flat | ASMP doesn't apply to this architecture. Pivot to per-channel scaling + PC-KD. |
| Hardware doesn't support INT8/INT16 mixing | Week 7 smoke test on Cortex-M fails | Pivot to FPGA, or fall back to per-channel scaling. |
| ASMP + PC-KD doesn't beat uniform INT8 + PC-KD | Week 11 Pareto curve shows overlap | Report the negative result honestly. PC-KD alone is still a contribution. |
| Cross-mass-range generalisation fails | Schedule optimal for BBH crashes on BNS | Restrict thesis scope to a single mass range; flag generalisation as future work. |

The decision tree is: if any of these triggers fire, the thesis story shifts but the contribution survives. Plan for graceful degradation, not for everything working.

---

## 9. Literature to read, priority-ordered

Already in the LR (re-read with ASMP lens):

1. **Qi et al.\ (2024)** — the BNS-on-FPGA paper. Read carefully for *which layers* of their CNN showed the most INT8 sensitivity. If they report per-layer numbers, the ASMP receptive-field hypothesis can be sanity-checked against their data before any new experiment runs.
2. **Chaturvedi et al.\ (2022)** — TensorRT mixed FP16/INT8. Read for the *layer-fusion strategy* they use. Even though theirs is hardware-driven, the layers they choose to keep in FP16 are an interesting empirical signal.
3. **Krastev (2020)** — BNS detection. Read for the *receptive field* of his model and what fraction of the window is inspiral.
4. **Jacob et al.\ (2018)** — STE and integer-only inference. Re-read with FakeQuant implementation in mind.
5. **Hinton et al.\ (2015)** — distillation. Foundational, only needs a re-skim.

Not in the LR, worth chasing:

6. **MCUNet (Lin et al.\ 2020)** — per-stage memory-aware design. Their search space is the closest analogue to what ASMP is doing on a different axis.
7. **HAWQ-V3 (Yao et al.\ 2021)** — mixed-precision neural network search. Verify the citation, but if it exists, it's a useful baseline for "how does the wider ML literature pick mixed-precision schedules?"
8. **The PyCBC documentation on `pycbc.waveform.get_td_waveform`** — for generating the chirp templates needed for the phase-coherence weight $w(x)$.
9. **The GstLAL paper (Messick et al.\ 2017, already cited in LR §3.6)** — for understanding what a real trigger-stage candidate stream looks like, so the downstream-confirmation framing in the LR is honest.

Things to verify and add to the bibliography when running this experiment:

- HAWQ-V3 reference (year and venue, do not cite from memory)
- Any 2025–2026 papers on quantisation-aware ResNets for time series (search arXiv eess.SP + cs.LG)
- Any recent CMSIS-NN release notes about Helium support

---

## 10. Open questions for thesis advisor

Listing these in priority order. Most are decisions that should be made before week 5.

1. **Target hardware: Rosella variant (M4/M7/M55), FPGA, or both?** The whole ASMP feasibility story hinges on this.
2. **Target mass range for the COMP4450 experiment: BBH-only, BNS-only, or both?** ASMP probably has a cleaner story on BBH; BNS exposes the bigger gap. Pick one for the MVP.
3. **Architecture: 1-D ResNet (matches the existing literature) or plain 1-D CNN (matches the ASMP framing more cleanly)?**
4. **How much PyCBC waveform-bank generation is realistic in the timeline?** PC-KD needs a coarse template bank for the $w(x)$ pre-pass. Even a 1000-template bank is substantial work.
5. **Negative result tolerance.** If ASMP doesn't beat the strong baseline, the thesis story changes. Confirm that the supervisor is comfortable with a "PC-KD wins, ASMP does not" outcome before committing to ASMP as the headline.

---

## 11. Status log

(Maintain this as the project progresses. Date, what changed, what was learned.)

- *2026-05-25* — file created, working notes drafted from the COMP4450 literature review framing.
