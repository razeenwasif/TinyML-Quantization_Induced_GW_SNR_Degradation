# Literature Review: Toward Signal-Preserving TinyML for Gravitational Wave Triggers
## Resolving the Tension Between Astrophysical Fidelity and Sub-Watt Model Compression

## Identifiers
**Name:** Razeen Wasif
**Uni ID:** u7283652
**Course Code:** COMP4450
**Date:** May 25, 2026

## Abstract
Detection of compact-binary coalescences within the millisecond budget of multi-messenger astronomy is now constrained by hardware as much as by algorithms. Eighteen papers from gravitational-wave (GW) machine learning, embedded inference, and quantisation theory are synthesised to ask whether the trigger stage of a GW pipeline can be pushed onto sub-watt microcontrollers without losing astrophysical sensitivity. The literature separates into two weakly connected camps. One has shown that deep networks can match matched filtering on high-power GPUs. The other has shown that aggressive compression is feasible, but evaluates it on images and audio rather than on chirp waveforms. The hinge between the two is INT8 quantisation, which acts on a phase-coherent waveform as a structured perturbation rather than the well-behaved rounding error assumed by computer-vision benchmarks. Existing quantisation-aware training and distillation work does not address this, leaving a clear opening for physics-informed, signal-preserving compression.

## Keywords
TinyML; quantisation-aware training; gravitational-wave detection; signal-to-noise ratio; embedded machine learning; low-latency triggers; multi-messenger astronomy.

**ACM CCS:** Computing methodologies $\rightarrow$ Machine learning algorithms; Computer systems organization $\rightarrow$ Embedded systems; Applied computing $\rightarrow$ Astronomy.

---

## 1. Introduction

### 1.1 Summary of the Research Proposal
This review provides the literature foundation for the COMP4450 Research Proposal *"Quantisation-Induced SNR Degradation: Toward Signal-Preserving TinyML for Gravitational Wave Triggers"* (Wasif 2026). The proposal sits at the intersection of three computer-science sub-fields. *Intelligent Systems* is the primary home; *Embedded Computer Systems* and *Computational Astrophysics* are secondary contexts. Its central claim is that next-generation gravitational-wave detectors face a data-deluge problem GPU-bound deep-learning pipelines cannot solve. Missions such as LISA and CubeSat-scale autonomous arrays simply cannot downlink the raw data, so the trigger stage has to move onto microcontrollers running in 128 KB of SRAM at sub-watt power. That move forces aggressive INT8 quantisation, and the proposal argues this will destroy the phase coherence of the chirp waveform. The research question follows directly: *to what extent does non-linear bit-width reduction interfere with the phase coherence on which reliable GW triggering depends?*

### 1.2 Aim and research questions
The aim of this review is to survey the literature that approaches the gap from two directions: GW machine learning on one side, and TinyML and quantisation theory on the other. The case made here is that the intersection remains unresolved. Two questions structure the survey. *RQ1* asks how prior work has traded model size, latency, and astrophysical sensitivity. *RQ2* asks which quantisation-aware techniques, if any, have been validated against astrophysical metrics such as false alarm rate and distance reach, rather than against the top-1-accuracy metrics inherited from computer vision. The technical context matters here. Detecting strains of $h \approx 10^{-21}$ by matched filtering is statistically optimal but computationally heavy (Abbott et al. 2020), and template-bank growth is now the limiting factor in multi-messenger follow-up (Cuoco et al. 2024). Deep learning matches matched-filter sensitivity in published results (George & Huerta 2018; Gabbard et al. 2018), but those results all assume FP32 arithmetic. What happens when that arithmetic is compressed to INT8 is the question this review traces.

---

## 2. Materials and Methods

A targeted, reproducible review was conducted between March and April 2026. It is lighter than a formal systematic review, but disciplined enough that someone else could re-run the search.

### 2.1 Search strategy
Four databases were used: arXiv (gr-qc, cs.LG, eess.SP), IEEE Xplore, the ACM Digital Library, and Google Scholar. Query strings paired GW terms (*"gravitational wave" OR "binary black hole" OR "binary neutron star"*) with ML/hardware terms (*"deep learning" OR "convolutional" OR "quantisation" OR "FPGA" OR "microcontroller" OR "TinyML"*). GWOSC was checked for datasets and observational papers, and citation chasing used George and Huerta (2018), Gebhard et al. (2019), and Gholami et al. (2021) as seeds.

### 2.2 Inclusion and exclusion criteria
Papers were included if they did one of three things. First, propose or evaluate a machine-learning model for GW detection, denoising, or trigger generation. Second, supply foundational technique work on quantisation, distillation, or embedded inference frameworks the GW side draws on. Third, report quantitative results on either sensitivity (SNR, FAR, detection probability) or efficiency (latency, memory, parameter count). Papers were excluded if they focused only on offline Bayesian parameter estimation, on cosmology without a trigger stage, or used neural networks merely for post-hoc visualisation. Murphy et al. (2021) on the EIRSAT-1 CubeSat gamma-ray detector is included on the deployment side as the only autonomous, downlink-constrained instrument paper in the set with directly comparable on-board triggering constraints.

### 2.3 Inclusion outcome
Eighty-six records were identified. After deduplication and title/abstract screening 32 remained for full-text assessment, and 18 were finally included for in-depth review (Figure 1). The included set divides into four thematic groups: seven papers showing ML matching matched filtering, four on quantisation theory and embedded frameworks, three on FPGA-class hardware deployment in physics, and four critical or foundational papers on noise, glitches, and benchmarks.

### 2.4 Validity
Three measures support reproducibility. First, every included paper is cited with a DOI or arXiv ID, so the search can be replicated in five years. Second, a "clique map" was maintained that assigns each paper to one or more of three thematic axes (sensitivity, efficiency, robustness). The map makes visible where the literature is thick and where it is thin. Third, claims were cross-checked between independent works. The adversarial-noise critique of George and Huerta (2018) by Gebhard et al. (2019) is one example, and Banbury et al.'s (2021) microcontroller benchmarks, when read against the SRAM budgets assumed by David et al. (2021), are another.

```
            ┌──────────────────────────────┐
 Identific. │ Records identified           │
            │ arXiv (54) · IEEE (12)       │
            │ ACM (4)   · Scholar (16)     │
            │            Total = 86         │
            └──────────────┬────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
 Screening  │ After deduplication: 71      │
            │ Title/abstract screened: 71  │
            │ Excluded (off-topic): 39     │
            └──────────────┬────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
 Eligibility│ Full text assessed: 32       │
            │ Excluded (no quant. results,│
            │ off-line PE only, no GW or  │
            │ no compression link): 14    │
            └──────────────┬────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
 Included   │ Papers included for in-depth │
            │ synthesis: 18                │
            │ (sensitivity 7 · efficiency  │
            │ 4 · hardware 3 · noise/      │
            │ benchmarks 4)                │
            └──────────────────────────────┘
```
**Figure 1.** *PRISMA-style flow of identification, screening, eligibility, and inclusion (monospaced font preserved for structural alignment).*

---

## 3. Results

**Key message.** The literature splits into two cliques that rarely cite each other. A sensitivity-first clique (George and Huerta 2018; Gabbard et al. 2018; Krastev 2020; Schäfer et al. 2023; Dax et al. 2021; Chaturvedi et al. 2022) shows that deep learning can match matched filtering, but only on GPU. A deployment-first clique (David et al. 2021; Banbury et al. 2021; Gholami et al. 2021; Jacob et al. 2018; Hinton et al. 2015) shows that aggressive compression is feasible, but evaluates models on images, audio, or generic benchmarks rather than on astrophysical strain. Sitting between the two are critical papers (Gebhard et al. 2019; Abbott et al. 2016; Abbott et al. 2020) that warn the resulting models are fragile in ways simple accuracy metrics cannot detect. Mapped onto contribution axes the 18 papers split roughly seven on sensitivity, six on efficiency, and three on robustness; only Qi et al. (2024) and Chaturvedi et al. (2022) contribute primarily on more than one, and no paper contributes primarily on all three. Figure 2 projects the same literature onto the axes the proposal's hypothesis actually lives on: weight precision and sensitivity preservation.

### 3.1 The sensitivity-first clique
George and Huerta (2018) and Gabbard et al. (2018) form the foundation of ML-based GW detection. Both show that a relatively shallow one-dimensional CNN can recover binary black hole (BBH) signals from Advanced LIGO strain at sensitivity comparable to matched filtering. They are best read as a clique: they converge on the same architectural choice (deep 1-D convolutions over whitened strain) and report the same null result (no measurable loss of true-positive rate at high SNR), so their joint claim is stronger than either paper alone.

Krastev (2020) extended the design to binary neutron star (BNS) signals, and in doing so exposed a weakness the BBH papers had concealed. BNS waveforms are an order of magnitude longer in the detector band, so the receptive field of any TinyML candidate model has to grow proportionally. That requirement runs directly into the SRAM budget assumed by David et al. (2021).

Schäfer et al. (2023) formalised the comparison through the first ML mock-data challenge. The headline result was that ML pipelines were competitive with PyCBC at high false-alarm-rate thresholds, but lagged at the low-FAR regime that astrophysical confirmation actually requires. Dax et al. (2021) and Chaturvedi et al. (2022) close out the clique on the inference side. The former demonstrates real-time *posterior* estimation in seconds. The latter applies NVIDIA TensorRT mixed-precision (FP16/INT8) layer fusion and reports processing approximately one month of LIGO data in about 50 seconds, a roughly 3× speedup, with negligible sensitivity loss for known BBH events. The limitation is that this result depends on A100-class datacentre GPUs and TensorRT calibration, not on memory-static microcontroller inference. The collective blind spot of the clique is therefore hardware scale.

### 3.2 The deployment-first clique
Where the sensitivity clique assumes GPUs, the deployment clique assumes microcontrollers. David et al. (2021) define the de facto runtime for TinyML in the form of TensorFlow Lite Micro, and lay out the engineering primitives that govern what can and cannot run in approximately 256 KB of SRAM: a single pre-allocated tensor arena, op-by-op execution, no dynamic allocation after initialisation, and vendor-specific kernels such as CMSIS-NN. Their evaluation reports interpreter overhead below 1%. Banbury et al. (2021) supply the calibrating benchmark suite (MLPerf Tiny). Both target keyword spotting, image classification, and anomaly detection, not astrophysical time series.

The theoretical machinery sits with Gholami et al. (2021), whose taxonomy of post-training quantisation (PTQ) and quantisation-aware training (QAT) is the canonical reference, and with Jacob et al. (2018), whose integer-arithmetic-only inference paper supplies the mathematical foundation, including the straight-through estimator for backpropagating through a discontinuous rounding operator. Knowledge distillation (Hinton et al. 2015) is the complementary tool: a high-precision teacher transfers softened class probabilities to a quantised student, recovering accuracy PTQ discards.

The clique is internally coherent and well-cited. But its evaluation regime is uniformly computer-vision-centric: top-1 ImageNet accuracy, word-error rate, mean average precision. None of these metrics quantifies the loss of phase coherence that matters for a chirp signal. That gap is the central observation of this review.

### 3.3 Hardware-software co-design on FPGA-class targets
A small group of papers sits between the two cliques and provides the closest evidence for the proposal's hardware claim. Qi et al. (2024) implemented a CNN for BNS early-warning on an FPGA, achieved sub-millisecond inference, and showed empirically that INT8 quantisation was tolerable at stronger signal regimes but degraded sensitivity for the long-duration BNS signals that motivated the work. Within the included set, this is the clearest empirical evidence for the SNR–precision trade-off the proposal hypothesised. Chaturvedi et al. (2022) span both cliques. Their TensorRT result is on GPU, but it operationalises the same mixed-precision idea microcontroller deployments will eventually need: higher precision in sensitive layers, INT8 elsewhere. The two papers are the closest existing analogues to "signal-preserving TinyML". Neither, however, tests a sub-watt microcontroller target, reports peak SRAM under a TinyML memory arena, uses astrophysical distance reach as a metric, or generalises the underlying framework.

### 3.4 Three classes of detector noise
Gebhard et al. (2019) is the critical pivot of the review. It re-examines the George and Huerta (2018) and Gabbard et al. (2018) results, and shows the high reported accuracies disguise something the authors did not test for: adversarial brittleness. Small, structured perturbations to the input strain flip the network output with high confidence, and the network confidently classifies inputs that bear no visible resemblance to a chirp. Gebhard et al. conclude that CNNs should be used as trigger generators only, not as confirmation tools.

Abbott et al. (2016) catalogue the morphology of "blip" transients in real LIGO data. These are exactly the kind of high-amplitude, short-duration response a quantised CNN is most likely to misclassify. Abbott et al. (2020) document the data-quality vetoes the LVK collaboration uses to mitigate them offline.

Three structurally distinct sources of detector noise therefore threaten an ML-based GW trigger: instrumental glitches in the strain itself (Abbott et al. 2016; Abbott et al. 2020); adversarial perturbations of the learned decision surface (Gebhard et al. 2019); and the algorithmic noise injected by the inference pipeline, of which INT8 quantisation is the largest single contributor (Gholami et al. 2021; Jacob et al. 2018). Treating quantisation as a third source of detector noise, algorithmic rather than instrumental, is the conceptual move this review proposes. It also explains why computer-vision quantisation results do not transfer to GW triggers: image and audio benchmarks have no analogue of detector noise, so quantisation error there is essentially rounding, whereas in a GW trigger it is a structured perturbation that interacts with chirp phase evolution and therefore directly affects astrophysical sensitivity. None of the deployment-first papers acknowledge this, which is a serious omission.

### 3.5 Toward signal-preserving TinyML
Cuoco et al. (2021) provide the earlier of two field-shaping reviews of ML in GW science; Cuoco et al. (2024) extend that coverage to current detectors. Neither devotes more than a paragraph to sub-watt deployment. The methodological building blocks for closing the gap already exist. Hinton et al.'s (2015) distillation framework supplies the training objective. Jacob et al.'s (2018) integer-only inference and Gholami et al.'s (2021) QAT taxonomy supply the quantisation machinery. What the literature does not contain is a composition of those blocks under astrophysical, rather than computer-vision, loss functions. Murphy et al. (2021) on the EIRSAT-1 CubeSat gamma-ray detector illustrates the deployment niche the proposal is aimed at: a compact, autonomous, downlink-constrained instrument where on-board triggering is operationally required.

### 3.6 The wrong benchmark
The sensitivity-first clique evaluates ML detectors by matched-filter parity at fixed FAR. That benchmark suits confirmation, but an edge trigger only needs to flag candidate windows cheaply enough for a higher-precision second stage. This hierarchy already exists: GstLAL is a rapid trigger feeding deeper analyses (Messick et al. 2017). Matched-filter parity at the trigger stage is therefore over-specification, and the FP32 assumption that helps meet it blocks TinyML deployment. Read this way, Qi et al.'s (2024) INT8 BNS result is not simply a failure: under a hierarchical-trigger benchmark, the relevant metric is candidate recall under tractable downstream load. "Signal-preserving" therefore means preserving enough phase information for confirmation, not replacing matched filtering.

```
Sensitivity Preserved vs FP32 Baseline
100% |---------------------------------------- [FP32 Baseline] (G&H, Gabbard, Krastev)
     |           \
 85% |            \    +-----------------+
     |             \   |  EMPTY REGION   |     [Chaturvedi 2022 (Mixed FP16/INT8)]
 70% |              \  |  (Sub-Watt      |
     |               \ |   INT8 BNS)     |
 50% |                \+-----------------+
     |                 \
     |                  \ [Qi 2024 (INT8 BNS)]
     |                   \
     +----------------------------------------
       4(INT4)     8(INT8)    16(FP16)    32(FP32)  Weight Precision (bits)
```
**Figure 2.** *Synthesis sketch of sensitivity preservation versus weight precision. BBH results remain close to FP32 through mixed FP16/INT8 (Chaturvedi et al. 2022), while BNS sensitivity is conjectured to degrade more sharply from the single GW INT8 anchor (Qi et al. 2024). The green region is the missing target: BNS-class sensitivity at INT8 or below on a sub-watt microcontroller.*

\
The empty green region in Figure 2 is the research gap PC-KD (Section 4.3) is designed to fill.

**Table 1.** *Trade-off positions of the 18 included papers along three axes (✓ = primary contribution; ○ = secondary; — = not addressed).*

| Paper | Sensitivity | Efficiency | Robustness/Noise |
|---|---|---|---|
| George & Huerta 2018 | ✓ | — | — |
| Gabbard et al. 2018 | ✓ | — | — |
| Krastev 2020 | ✓ | ○ | — |
| Schäfer et al. 2023 | ✓ | — | ○ |
| Dax et al. 2021 | ✓ | ○ | — |
| Chaturvedi et al. 2022 | ✓ | ✓ | — |
| Qi et al. 2024 | ✓ | ✓ | ○ |
| Gebhard et al. 2019 | ○ | — | ✓ |
| Abbott et al. 2016 | — | — | ✓ |
| Abbott et al. 2020 | ○ | — | ✓ |
| David et al. 2021 | — | ✓ | — |
| Banbury et al. 2021 | — | ✓ | — |
| Gholami et al. 2021 | — | ✓ | ○ |
| Jacob et al. 2018 | — | ✓ | — |
| Hinton et al. 2015 | ○ | ✓ | — |
| Cuoco et al. 2021 | ○ | — | ○ |
| Cuoco et al. 2024 | ○ | ○ | ○ |
| Murphy et al. 2021 | — | ✓ | — |

The lower-right quadrant of Table 1—papers that are simultaneously strong on astrophysical sensitivity, embedded efficiency and noise robustness—is empty. That empty quadrant *is* the research gap the proposal targets.

---

## 4. Discussion

**Key message.** The literature provides every ingredient for signal-preserving TinyML—architectures, runtimes, quantisation theory, distillation, hardware—but no published work integrates them under astrophysical loss functions, and the closest analogue (Qi et al. 2024) already shows that naive INT8 erodes BNS sensitivity. The gap is therefore one of *evaluation discipline* as much as of technique.

### 4.1 Principal results
Four findings emerge. First, deep learning reaches matched-filter parity for high-SNR BBH signals, but weakens at low FAR and on long-duration BNS waveforms (George & Huerta 2018; Gabbard et al. 2018; Schäfer et al. 2023). Second, embedded-ML measures quantisation cost in classification accuracy, not phase-coherent SNR loss (Jacob et al. 2018; Gholami et al. 2021; Banbury et al. 2021). Third, the only FPGA-class GW deployment shows naive INT8 degrading BNS sensitivity (Qi et al. 2024). Fourth, matched-filter parity is the wrong target for a first-stage edge trigger; under a hierarchical benchmark, Qi et al. (2024) is a partial success.

### 4.2 Limitations
Three limitations should be flagged. The 18-paper cap excluded adjacent work on mixed-precision FPGA inference outside astronomy and transformer-based GW pipelines. Some recent papers are arXiv preprints, so peer-review status varies. Figure 2 is also a coarse two-axis projection: it suppresses activation precision, layer-wise mixed precision, calibration choices, and heterogeneity within BBH and BNS tasks. The BNS curve is hypothesised from one empirical anchor (Qi et al. 2024) and should be redrawn as more INT8 BNS deployments appear.

### 4.3 Proposed future work: Phase-Coherence-Weighted Knowledge Distillation
The empty region of Figure 2 motivates a named contribution: *Phase-Coherence-Weighted Knowledge Distillation* (PC-KD). PC-KD weights the student loss sample-by-sample by the time-frequency phase coherence of the input window against a chirp template. Quantisation noise in low-coherence regions is penalised softly; noise in late-inspiral and merger-adjacent regions, where phase evolution dominates SNR, is penalised hard. Concretely,

$$\mathcal{L}_{\text{PC-KD}}(x) \;=\; \alpha \, \mathrm{KL}\big( p_T(x) \,\|\, p_S(x) \big) \;+\; \beta \, w(x) \, \big\| f_T(x) - f_S(x) \big\|^2,$$

where $p_T, p_S$ are teacher and student probabilities, $f_T, f_S$ are penultimate-layer features, and $w(x)$ is a phase-coherence weight from a coarse matched-filter pre-pass. The KL term transfers teacher decisions; the weighted feature term is the astrophysical component, making compression care where quantisation error damages SNR. PC-KD should be tested through microcontroller-class mixed precision (Chaturvedi et al. 2022), glitch diagnostics (Gebhard et al. 2019; Abbott et al. 2016; Abbott et al. 2020), and an MLPerf-Tiny-style benchmark parameterised by astrophysical reach (Banbury et al. 2021).

### 4.4 Comparison with prior work
Cuoco et al. (2021; 2024) review ML in GW science but treat hardware deployment as downstream engineering. Warden and Situnayake (2019), Lin et al. (2020), and Sze et al. (2017) survey embedded inference without astrophysical content. This review differs by treating the intersection as the problem, using Gebhard et al. (2019) and Qi et al. (2024) as load-bearing evidence, reframing matched-filter parity as the wrong trigger benchmark, and proposing PC-KD as a named methodological contribution.

### 4.5 Conclusions
TinyML for gravitational-wave detection cannot be achieved by stacking generic compression onto GPU-oriented ML pipelines. It also cannot treat quantisation as ordinary rounding or demand matched-filter parity from a first-stage trigger. The ingredients already exist: sensitive architectures, 256 KB-class runtimes, quantisation procedures, and distillation. What is missing is their composition under astrophysical metrics. PC-KD, as proposed in Section 4.3 and adopted by Wasif (2026), is the missing piece this review identifies.

---

## 5. Self-Reflection
Generative AI was used in three roles: Google Gemini CLI for early section skeletons and reference discovery, and Anthropic Claude for one near-final cross-check. The drafting help was useful, but every paragraph was rewritten by hand. Reference discovery required stricter control. An initial prompt ("list TinyML papers for gravitational wave detection") produced plausible but fabricated DOIs; a refined prompt asking for verifiable FPGA/CNN/BNS papers returned fewer results, with Qi et al. (2024) the only usable hit. The main benefit was speed in moving between gravitational-wave physics and embedded systems. The main risk was academic-integrity failure through fabricated or over-stated sources. The mitigation was to verify every AI-suggested paper independently and keep no claim without an author-checked source. For future work, summaries should be drafted from PDFs first, with GenAI used only for comparison or copy-editing.

---

## 6. List of References

1. Abbott, BP et al. 2016, 'Characterization of transient noise in Advanced LIGO relevant to gravitational-wave signal GW150914', *Classical and Quantum Gravity*, vol. 33, no. 13, 134001, doi:10.1088/0264-9381/33/13/134001. https://doi.org/10.1088/0264-9381/33/13/134001 .*
2. Abbott, BP et al. 2020, 'A guide to LIGO–Virgo detector noise and extraction of transient gravitational-wave signals', *Classical and Quantum Gravity*, vol. 37, no. 5, 055002, doi:10.1088/1361-6382/ab685e. https://doi.org/10.1088/1361-6382/ab685e .*
3. Banbury, CR, Reddi, VJ, Lam, P, Fu, W, Fazel, A, Holleman, J, Huang, X, Hurtado, R, Kanter, D, Lokhmotov, A, Patterson, D, Pau, D, Seo, J, Sieracki, J, Thakker, U, Verhelst, M & Yadav, P 2021, 'MLPerf Tiny benchmark', in *Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*, arXiv:2106.07597. https://arxiv.org/abs/2106.07597 .*
4. Cuoco, E, Powell, J, Cavaglià, M, Ackley, K, Bejger, M, Chatterjee, C, Coughlin, M, Coughlin, S, Easter, P, Essick, R, Gabbard, H, Gebhard, T, Ghosh, S, Haegel, L, Iess, A, Keitel, D, Márka, Z, Márka, S, Morawski, F, Nguyen, T, Ormiston, R, Pürrer, M, Razzano, M, Salemi, F, Sasaoka, S & Williamson, A 2021, 'Enhancing gravitational-wave science with machine learning', *Machine Learning: Science and Technology*, vol. 2, no. 1, 011002, doi:10.1088/2632-2153/abb93a. https://doi.org/10.1088/2632-2153/abb93a .*
5. Cuoco, E, Cavaglià, M, Heng, IS, Keitel, D & Messenger, C 2024, 'Applications of machine learning in gravitational-wave research with current interferometric detectors', arXiv:2412.15046. https://arxiv.org/abs/2412.15046 .*
6. David, R, Duke, J, Jain, A, Janapa Reddi, V, Jeffries, N, Li, J, Kreeger, N, Nappier, I, Natraj, M, Regev, S, Rhodes, R, Wang, T & Warden, P 2021, 'TensorFlow Lite Micro: embedded machine learning on TinyML systems', in *Proceedings of Machine Learning and Systems (MLSys)*, vol. 3, pp. 800–811, arXiv:2010.08678. https://arxiv.org/abs/2010.08678 .*
7. Dax, M, Green, SR, Gair, J, Macke, JH, Buonanno, A & Schölkopf, B 2021, 'Real-time gravitational-wave science with neural posterior estimation', *Physical Review Letters*, vol. 127, no. 24, 241103, doi:10.1103/PhysRevLett.127.241103. https://doi.org/10.1103/PhysRevLett.127.241103 .*
8. Gabbard, H, Williams, M, Hayes, F & Messenger, C 2018, 'Matching matched filtering with deep networks for gravitational-wave astronomy', *Physical Review Letters*, vol. 120, no. 14, 141103, doi:10.1103/PhysRevLett.120.141103. https://doi.org/10.1103/PhysRevLett.120.141103 .*
9. Gebhard, TD, Kilbertus, N, Harry, I & Schölkopf, B 2019, 'Convolutional neural networks: a magic bullet for gravitational-wave detection?', *Physical Review D*, vol. 100, no. 6, 063015, doi:10.1103/PhysRevD.100.063015. https://doi.org/10.1103/PhysRevD.100.063015 .*
10. George, D & Huerta, EA 2018, 'Deep learning for real-time gravitational-wave detection and parameter estimation: results with Advanced LIGO data', *Physics Letters B*, vol. 778, pp. 64–70, doi:10.1016/j.physletb.2017.12.053. https://doi.org/10.1016/j.physletb.2017.12.053 .*
11. Gholami, A, Kim, S, Dong, Z, Yao, Z, Mahoney, MW & Keutzer, K 2021, 'A survey of quantization methods for efficient neural network inference', arXiv:2103.13630. https://arxiv.org/abs/2103.13630 .*
12. Hinton, G, Vinyals, O & Dean, J 2015, 'Distilling the knowledge in a neural network', arXiv:1503.02531. https://arxiv.org/abs/1503.02531 .*
13. Chaturvedi, P, Khan, A, Tian, M, Huerta, EA & Zheng, H 2022, 'Inference-optimized AI and high performance computing for gravitational wave detection at scale', *Frontiers in Artificial Intelligence*, vol. 5, 828672, doi:10.3389/frai.2022.828672. https://doi.org/10.3389/frai.2022.828672 .*
14. Jacob, B, Kligys, S, Chen, B, Zhu, M, Tang, M, Howard, A, Adam, H & Kalenichenko, D 2018, 'Quantization and training of neural networks for efficient integer-arithmetic-only inference', in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2704–2713, arXiv:1712.05877. https://arxiv.org/abs/1712.05877 .*
15. Krastev, PG 2020, 'Real-time detection of gravitational waves from binary neutron-star mergers using artificial neural networks', *Physics Letters B*, vol. 803, 135330, doi:10.1016/j.physletb.2020.135330. https://doi.org/10.1016/j.physletb.2020.135330 .*
16. Murphy, D, Ulyanov, A, McBreen, S, Doyle, M, Dunwoody, R, Mangan, J, Erkal, J, Hibbert, B, Marshall, A, Martin-Carrillo, A, Mathews, J, Murphy, D, Norman, C, O'Toole, L, Reilly, J, Salmon, L, Sherwin, J, Shortt, B, Thompson, J, Walsh, S, Wall, S, Wells, J, Wesseling, F & Hanlon, L 2021, 'A compact instrument for gamma-ray burst detection on a CubeSat platform (EIRSAT-1 GMOD)', *Experimental Astronomy*, vol. 52, pp. 1–32, doi:10.1007/s10686-021-09779-9. https://doi.org/10.1007/s10686-021-09779-9 .*
17. Qi, H, Chen, H-Y, Harms, J, Cusin, G & Sesana, A 2024, 'Improving early detection of gravitational waves from binary neutron stars using CNNs and FPGAs', arXiv:2409.05068. https://arxiv.org/abs/2409.05068 .*
18. Schäfer, MB, Zelenka, O, Nitz, AH, Wang, H, Wu, S, Guo, Z-K, Cao, Z, Ren, Z, Nousi, P, Stergioulas, N, Iosif, P, Koloniari, AE, Tefas, A, Passalis, N, Salemi, F, Vedovato, G, Klimenko, S, Mishra, T, Brügmann, B, Cuoco, E, Huerta, EA, Messenger, C & Ohme, F 2023, 'MLGWSC-1: the first machine learning gravitational-wave search mock data challenge', *Physical Review D*, vol. 107, no. 2, 023021, doi:10.1103/PhysRevD.107.023021. https://doi.org/10.1103/PhysRevD.107.023021 .*

### Additional references cited but not part of the 18 in-depth included set
19. Lin, J, Chen, W-M, Lin, Y, Cohn, J, Gan, C & Han, S 2020, 'MCUNet: tiny deep learning on IoT devices', in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 11711–11722, arXiv:2007.10319. https://arxiv.org/abs/2007.10319
20. Sze, V, Chen, Y-H, Yang, T-J & Emer, JS 2017, 'Efficient processing of deep neural networks: a tutorial and survey', *Proceedings of the IEEE*, vol. 105, no. 12, pp. 2295–2329, doi:10.1109/JPROC.2017.2761740. https://doi.org/10.1109/JPROC.2017.2761740
21. Warden, P & Situnayake, D 2019, *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*, O'Reilly Media, Sebastopol, CA. ISBN 978-1492052043.
22. Wasif, R 2026, *Quantisation-Induced SNR Degradation: Toward Signal-Preserving TinyML for Gravitational Wave Triggers*, Research Proposal, COMP4450, Australian National University.
