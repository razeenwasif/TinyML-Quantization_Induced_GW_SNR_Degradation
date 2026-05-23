# Literature Review: Toward Signal-Preserving TinyML for Gravitational Wave Triggers
## Resolving the Tension Between Astrophysical Fidelity and Sub-Watt Model Compression

## Identifiers
**Name:** Razeen Wasif
**Uni ID:** u7283652
**Course Code:** COMP4450

## Abstract
Detecting compact-binary coalescences within the millisecond latency budget of multi-messenger astronomy is increasingly bound by hardware, not by algorithms. Twenty papers spanning gravitational-wave (GW) machine learning, embedded inference, and numerical quantisation theory are synthesised here to chart the path from GPU-bound deep filtering toward TinyML triggers running on sub-watt microcontrollers and field-programmable gate arrays. Two cliques are identified: a "sensitivity-first" cluster that matches or surpasses matched-filtering on high-precision hardware, and a "deployment-first" cluster that compresses models without explicit reference to astrophysical fidelity. Neither closes the gap. Quantisation to INT8 is shown to be the principal threat to phase-coherence and therefore to signal-to-noise ratio (SNR), yet existing quantisation-aware training, knowledge distillation and topological methods are evaluated under computer-vision metrics rather than false-alarm rate or distance reach. A research opening therefore exists for physics-informed, signal-preserving quantisation.

## Keywords
TinyML; quantisation-aware training; gravitational-wave detection; signal-to-noise ratio; embedded machine learning; low-latency triggers; multi-messenger astronomy.

## 1. Introduction

### 1.1 Summary of the Research Proposal
This literature review extends the COMP4450 Research Proposal *"Quantisation-Induced SNR Degradation: Toward Signal-Preserving TinyML for Gravitational Wave Triggers"* (Wasif 2026). That proposal sited the work at the intersection of three computer-science sub-fields—*Intelligent Systems* (primary), *Embedded Computer Systems* and *Computational Astrophysics* (secondary)—and argued that next-generation observatories, including space-based LISA and CubeSat-scale autonomous arrays, face a "data deluge" that GPU-centric deep-learning pipelines cannot resolve. Shifting the trigger stage onto microcontrollers with ≤ 128 KB of SRAM demands aggressive INT8 quantisation; the proposal hypothesised that such quantisation may destroy the phase-coherence of the "chirp" waveform and so collapse the SNR on which detection depends. Its central question—*to what extent does non-linear bit-width (INT8) reduction interfere with the phase-coherence required for reliable GW signal triggering?*—motivates this review.

### 1.2 Aim and research questions
This review's aim is to survey, compare and critique the existing body of work that touches that gap from either side—GW machine learning on one side, and TinyML and quantisation theory on the other—and to argue that no current paper closes it. Two questions are addressed: *(RQ1)* how have prior works traded model size, latency and astrophysical sensitivity in GW machine learning? and *(RQ2)* what quantisation-aware techniques have been validated against astrophysical metrics (false alarm rate, distance reach) rather than computer-vision metrics? Detecting strains of $h \approx 10^{-21}$ with matched filtering is statistically optimal but computationally expensive (Abbott et al. 2020), and exponential template-bank growth has turned classical pipelines into the limiting factor for multi-messenger follow-up (Cuoco et al. 2024). Deep learning matches matched-filter sensitivity (George & Huerta 2018; Gabbard et al. 2018) but assumes FP32 arithmetic. This review synthesises the consequences of compressing that arithmetic to INT8, revealing a critical, unaddressed methodology gap between high-fidelity physics metrics and standard hardware-compression evaluation frameworks.

## 2. Materials and Methods

A targeted, reproducible review—lighter than a formal systematic review but disciplined enough that another researcher could replicate the search—was conducted between March and April 2026.

### 2.1 Search strategy
Four databases were queried: arXiv (gr-qc, cs.LG, eess.SP), IEEE Xplore, the ACM Digital Library and Google Scholar. Search strings combined the GW side (*"gravitational wave" OR "binary black hole" OR "binary neutron star"*) with the ML/hardware side (*"deep learning" OR "convolutional" OR "quantisation" OR "FPGA" OR "microcontroller" OR "TinyML"*). The Gravitational Wave Open Science Center (GWOSC) was used to locate datasets and observational papers, and forward/backward citation chasing was performed on three seed papers (George & Huerta 2018; Gebhard et al. 2019; Gholami et al. 2021).

### 2.2 Inclusion and exclusion criteria
Papers were *included* if they (i) proposed or evaluated an ML model for GW detection, denoising or trigger generation; or (ii) provided foundational technique work on quantisation, knowledge distillation or embedded inference frameworks directly applicable to (i); and (iii) reported quantitative results on either sensitivity (SNR, FAR, detection probability) or efficiency (latency, memory footprint, parameter count). Papers were *excluded* if they focused solely on offline Bayesian parameter estimation, on cosmology-only inference without a trigger stage, or used neural networks merely as a post-hoc visualisation tool.

### 2.3 Inclusion outcome
A total of 86 records were identified; after deduplication and title/abstract screening 32 remained for full-text assessment, of which 18 were included for in-depth review (Figure 1). The included set spans seven that demonstrate ML matching matched filtering, four on quantisation theory and embedded frameworks, three on FPGA-class hardware deployment in physics, and four critical/foundational papers on noise, glitches and benchmarks.

### 2.4 Validity
Three measures support reproducibility. First, every included paper is cited with a persistent identifier (DOI or arXiv ID) so the search is replicable in five years. Second, a "clique map" was maintained that assigns each paper to one or more of three thematic axes (sensitivity, efficiency, robustness), which exposes where the literature is thick and where it is thin. Third, claims were cross-checked between independent works—for example, Gebhard et al.'s (2019) adversarial-noise critique of George & Huerta (2018), or Banbury et al.'s (2021) microcontroller benchmarks against the SRAM constraints assumed by David et al. (2021).

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

**Figure 1.** *PRISMA-style flow of identification, screening, eligibility and inclusion (monospaced font preserved for structural alignment).*

## 3. Results

**Key message.** The literature partitions into two largely non-overlapping cliques. A "sensitivity-first" clique (George & Huerta 2018; Gabbard et al. 2018; Krastev 2020; Schäfer et al. 2023; Dax et al. 2021; Chaturvedi et al. 2022) shows that deep learning can match matched filtering, but only on GPUs. A "deployment-first" clique (David et al. 2021; Banbury et al. 2021; Gholami et al. 2021; Jacob et al. 2018; Hinton et al. 2015) shows that aggressive compression is feasible, but evaluates models on images, audio or generic benchmarks rather than astrophysical strain. Critical papers (Gebhard et al. 2019; Abbott et al. 2016; Abbott et al. 2020) sit between the two and warn that the resulting models are fragile in ways that simple accuracy metrics cannot detect. Table 1 maps each included paper to the trade-off space.

### 3.1 The sensitivity-first clique: deep learning matches matched filtering
George & Huerta (2018) and the contemporaneous, independent work of Gabbard et al. (2018) are the foundation of ML-based GW detection. Both demonstrate that a relatively shallow 1-D CNN can recover binary black hole (BBH) signals from Advanced LIGO strain at sensitivity comparable to matched filtering. The two papers are best read as a clique rather than as separate contributions: they converge on the same architectural choice (deep 1-D convolutions over whitened strain) and the same null result (no measurable loss of true-positive rate at high SNR), and so their joint claim is stronger than either alone. Krastev (2020) extended this design to binary neutron star (BNS) signals, exposing a weakness that the BBH papers had hidden—BNS waveforms are an order of magnitude longer in the detector band, which forces the receptive field of any TinyML candidate model to grow correspondingly, in direct tension with the SRAM budget assumed by David et al. (2021). Schäfer et al. (2023) formalised the comparison by running the first ML mock-data challenge, finding that the ML pipelines were competitive with PyCBC at high false-alarm-rate thresholds but lagged at the low-FAR regime that matters most for astrophysical confirmation. Dax et al. (2021) and Chaturvedi et al. (2022) close the clique on the inference side: the former shows real-time *posterior* estimation in seconds, while the latter applies NVIDIA TensorRT mixed-precision (FP16/INT8) layer fusion to demonstrate a 3× speedup with negligible sensitivity loss—on A100 GPUs. The clique's collective blind spot is hardware scale: all six papers assume hundreds of watts of compute.

### 3.2 The deployment-first clique: compression in the abstract
Where the sensitivity clique assumes GPUs, the deployment clique assumes microcontrollers. David et al. (2021) define the de facto runtime for TinyML, TensorFlow Lite Micro, exposing the engineering primitives—arena allocation, op-by-op execution, fragmentation-free memory—that govern what is and is not deployable in ≤ 256 KB of SRAM. Banbury et al. (2021) supply the benchmark suite (MLPerf Tiny) that calibrates those primitives across heterogeneous microcontrollers; both papers, however, target keyword spotting, image classification and anomaly detection, not astrophysical time series. The theoretical machinery sits with Gholami et al. (2021), whose taxonomy of post-training quantisation (PTQ) and quantisation-aware training (QAT) is the canonical reference, and with Jacob et al. (2018), whose integer-arithmetic-only inference paper provides the mathematical foundation—including the straight-through estimator that lets gradients propagate through a discontinuous rounding operator. Knowledge distillation (Hinton et al. 2015) is the complementary tool: a high-precision "teacher" transfers its softened class probabilities to a quantised "student", recovering accuracy that PTQ alone discards. The clique is internally coherent and well-cited, but its evaluation regime is computer-vision-centric: top-1 ImageNet accuracy, word-error rate, mean-average-precision. None of these metrics quantifies the loss of phase-coherence that matters for a chirp signal, which is the central observation that motivates this review.

### 3.3 Hardware-software co-design on FPGA-class targets
A small but significant group sits between the two cliques. Qi et al. (2024) implemented a CNN for BNS early-warning on an FPGA, reporting sub-millisecond inference and demonstrating that quantisation to INT8 was tolerable *for BBH-like SNR*, but eroded sensitivity for the long-duration BNS signals that motivated the work. This is the first concrete empirical confirmation of the SNR–precision trade-off hypothesised in the proposal. Chaturvedi et al. (2022) span both cliques: their TensorRT result is on GPU but it operationalises the same mixed-precision idea (FP16 in sensitive layers, INT8 elsewhere) that microcontroller deployments will need. These two papers are the closest existing analogues to "signal-preserving TinyML", yet neither tests sub-watt microcontroller deployment, neither uses astrophysical distance reach as a metric, and neither offers a generalisable framework.

### 3.4 Noise, glitches and the brittleness of ML triggers
Gebhard et al. (2019) is the critical pivot of the review. It re-examines the George & Huerta (2018) / Gabbard et al. (2018) result and shows that the high reported accuracies disguise *adversarial brittleness*: small, structured perturbations to the input strain can flip the network's output with high confidence, and—more importantly for this review—the network can confidently classify inputs that bear no visible resemblance to a chirp. The paper recommends that CNNs be used as *trigger generators* rather than as confirmation tools. Abbott et al. (2016) catalogue the morphology of "blip" transients in real LIGO data that produce exactly the kind of high-amplitude, short-duration response a quantised CNN is likely to mis-classify, and Abbott et al. (2020) provide the data-quality vetoes used to mitigate them offline. The implicit chain of reasoning is uncomfortable for TinyML: if FP32 networks are already adversarially fragile, then INT8 quantisation—which is itself a structured, deterministic perturbation of the network's response surface—is likely to *amplify* that fragility rather than be neutral to it. Yet none of the deployment-first papers acknowledge the adversarial-noise literature.

### 3.5 Toward signal-preserving TinyML
Cuoco et al. (2021) provide the earlier of two field-shaping reviews of ML in GW science; the more recent Cuoco et al. (2024) update extends that coverage but neither addresses sub-watt deployment in any depth. Hinton et al.'s (2015) distillation framework, together with Jacob et al.'s (2018) integer-only inference and Gholami et al.'s (2021) QAT taxonomy, supplies the methodological building blocks; the question this review identifies is how to *compose* those blocks under astrophysical—not computer-vision—loss functions. The Murphy et al. (2021) work on the EIRSAT-1 CubeSat gamma-ray detector, representing a crucial out-of-domain hardware analog within our included set, illustrates the deployment niche the proposal targets: compact, autonomous, downlink-constrained instruments for which on-board triggering is a hard requirement, not a research curiosity.

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

## 4. Discussion

**Key message.** The literature provides every ingredient for signal-preserving TinyML—architectures, runtimes, quantisation theory, distillation, hardware—but no published work integrates them under astrophysical loss functions, and the closest analogue (Qi et al. 2024) already shows that naive INT8 erodes BNS sensitivity. The gap is therefore one of *evaluation discipline* as much as of technique.

### 4.1 Principal results
Three principal findings emerge. First, deep learning has reached parity with matched filtering for high-SNR BBH signals (George & Huerta 2018; Gabbard et al. 2018; Schäfer et al. 2023), but the parity is fragile at low FAR and on long-duration BNS waveforms. Second, the embedded-ML literature treats quantisation as a technique whose cost is measured in classification accuracy (Jacob et al. 2018; Gholami et al. 2021; Banbury et al. 2021), a metric that is structurally insensitive to phase-coherent SNR loss. Third, the only published FPGA-class GW deployment (Qi et al. 2024) directly demonstrates that naive quantisation erodes sensitivity in exactly the regime—long, weak BNS signals—where edge triggers would be most valuable.

### 4.2 Limitations
Three limitations of the review should be flagged. The 18-paper cap forced exclusion of several adjacent works (e.g. mixed-precision FPGA inference outside astronomy, recent transformer-based GW pipelines); these are not absent from the discipline, only from the in-depth synthesis here. Reliance on arXiv pre-prints for very recent work means peer-review status varies. Finally, the "clique" mapping in Table 1 is a coarse abstraction; some papers contribute incrementally on several axes, and the binary ✓/—/○ encoding suppresses that nuance.

### 4.3 Proposed future work
The empty quadrant of Table 1 suggests four concrete research directions, each of which the proposal's experimental design can address. First, define an *astrophysical loss* that penalises SNR-loss directly, in place of cross-entropy, building on Hinton et al.'s (2015) distillation objective. Second, generalise the mixed-precision strategy of Chaturvedi et al. (2022) from GPU TensorRT to microcontroller-class INT8/INT16 kernels, preserving FP16 (or INT16) precision only in the phase-sensitive early-warning layers identified by Krastev (2020). Third, transfer the adversarial-robustness diagnostics of Gebhard et al. (2019) into the embedded setting, treating INT8 quantisation as a structured perturbation and stress-testing the resulting model against real LIGO glitch morphologies (Abbott et al. 2016; Abbott et al. 2020). Fourth, develop a benchmark in the spirit of MLPerf Tiny (Banbury et al. 2021) but parameterised by astrophysical reach rather than top-1 accuracy.

### 4.4 Comparison with prior work
Cuoco et al. (2021; 2024) provide the most comprehensive prior reviews of ML in GW science, but both treat hardware deployment as a downstream engineering concern and devote little space to bit-width effects. The textbook and benchmark literature on the TinyML side surveys the embedded landscape with equal authority and no astrophysical content: Warden and Situnayake (2019) on the production side, and more recent on-device inference work such as Lin et al. (2020) on MCUNet. Sze et al. (2017) survey efficient hardware processing of deep networks with similar scope, again without a GW application. This review sits at the seam between the two literatures, and treats the seam itself as the problem. Critical mid-clique papers (Gebhard et al. 2019; Qi et al. 2024) are used as load-bearing evidence rather than as decorative citations. The framing departs from the black-box accuracy tradition of conventional model-compression surveys (Gholami et al. 2021; Jacob et al. 2018) by foregrounding adversarial fragility (Gebhard et al. 2019) and the morphology of detector noise (Abbott et al. 2016).

### 4.5 Conclusions
The integration of TinyML into gravitational-wave detection is a prerequisite for the next generation of multi-messenger observatories, but it cannot be achieved by stacking off-the-shelf compression onto off-the-shelf ML pipelines. The literature contains every ingredient required—architectures that match matched filtering, runtimes that fit in 256 KB, quantisation machinery proven on millions of devices, distillation procedures that preserve teacher knowledge—but no work composes them under the astrophysical metrics that the science actually demands. The contribution proposed by Wasif (2026), and motivated by this review, is precisely that composition: a framework in which quantisation is measured by what it does to SNR, not by what it does to accuracy.

## 5. Self-Reflection
Generative AI (Google Gemini CLI and Anthropic Claude) was utilised across three discrete roles: generating first-pass section skeletons for drafting, copy-editing paragraphs for academic style, and querying candidate paper lists to accelerate multidisciplinary search. The primary advantage was speed in cross-walking between gravitational-wave physics and embedded-systems literature. The principal disadvantage was the models' tendency to fabricate authoritative-sounding citations with invalid DOIs. To protect academic integrity, a strict verification protocol was implemented: every AI-suggested paper was independently searched, read, and verified via DOI/arXiv databases before inclusion, leading to the immediate removal of several plausible-looking but fabricated references. Ultimately, no text or claims were adopted without manual verification and extensive rewriting. This exercise demonstrated that while GenAI is a powerful cognitive accelerant for organizing research, it introduces substantial validation overhead, making independent verification the non-negotiable core of responsible AI-assisted scholarship.

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
