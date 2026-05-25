# Digital Signal Processing for Gravitational Wave Detection

## 1. The Physics of the Signal: Chirps and Strain

Gravitational waves (GWs) are ripples in spacetime fabric caused by accelerating massive objects. The signal is measured as **Strain ($h$)**, defined as the fractional change in length:

$$h = \frac{\Delta L}{L}$$

Because GWs are quadrupolar, for a detector with arm length $L$, one arm stretches while the other shrinks.

### The Chirp Signal

For Compact Binary Coalescences (CBCs), such as two black holes merging, the signal is a **Chirp**. As the objects spiral closer:

1. **Frequency ($f$)** increases.
    
2. **Amplitude ($A$)** increases.
    
    The resulting waveform is non-stationary and follows a specific mathematical evolution determined by the "Chirp Mass" ($\mathcal{M}$):
    
    $$\mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}$$
    

---

## 2. Matched Filtering: The Optimal Linear Filter

In GW data, the signal $s(t)$ is buried in stochastic noise $n(t)$. The total strain data is $d(t) = s(t) + n(t)$.

**Matched Filtering** is the process of correlating the raw data with a predicted theoretical "template" $h(t)$. It is the optimal linear filter for maximizing the **Signal-to-Noise Ratio (SNR)** when the signal shape is known.

### The Inner Product in Frequency Domain

To account for the fact that detector noise is not "white" (it varies by frequency), we use a weighted inner product:

$$\langle d|h \rangle = 4 \text{Re} \int_{f_{low}}^{f_{high}} \frac{\tilde{d}(f)\tilde{h}^*(f)}{S_n(f)} df$$

Where:

- $\tilde{d}(f)$ is the Fourier transform of the data.
    
- $\tilde{h}^*(f)$ is the complex conjugate of the template.
    
- $S_n(f)$ is the **Power Spectral Density (PSD)** of the noise (the weighting factor).
    

---

## 3. Phase Coherence

A critical requirement for matched filtering is **Phase Coherence**.

- The template and the signal must stay "in sync" over hundreds or thousands of cycles.
    
- If the template's phase drifts by even half a cycle ($\pi$ radians) relative to the signal, the correlation cancels out, and the signal disappears into the noise.
    
- **Search Banks:** Because we don't know the exact masses or spins of the black holes, researchers use a "Bank" of hundreds of thousands of templates, tiled closely enough that any real signal will have at least 97% overlap with at least one template.
    

---

## 4. Signal-to-Noise Ratio (SNR) and Statistics

In GW physics, we calculate the **Optimal SNR** ($\rho$):

$$\rho^2 = \langle s|s \rangle = 4 \int_{0}^{\infty} \frac{|\tilde{s}(f)|^2}{S_n(f)} df$$

### False Alarm Rate (FAR)

Because the noise $n(t)$ is Gaussian (mostly), it can occasionally "mimic" a chirp signal by pure chance.

- **False Alarm Probability:** The chance that a noise fluctuation produces an SNR above a certain threshold.
    
- **FAR:** Usually expressed as "1 event per 10,000 years."
    
- To lower the FAR, we use **Coincidence Analysis**: a signal must appear in multiple detectors (e.g., LIGO Hanford and LIGO Livingston) with the correct time-of-flight delay (up to 10ms for Earth-crossing).
    

---

## 5. Processing Astrophysical Strain Data

The raw strain data is notoriously "dirty." Before matched filtering, the data undergoes:

1. **Calibration:** Converting the interference fringe patterns (photons) into units of strain ($10^{-21}$).
    
2. **Whitening:** Dividing the data by the noise PSD to ensure all frequency bins have equal noise variance. This "flattens" the spectrum.
    
3. **Band-passing:** Removing low-frequency seismic noise (< 20Hz) and high-frequency photon shot noise.
    
4. **Glitch Removal:** Identifying and "vetoing" non-Gaussian transient noise (e.g., "blip" glitches) that can trigger high SNR but are terrestrial in origin.
    

---

## 6. Summary Table for Researchers

| **Concept**                     | **Role in Detection**                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Strain ($h$)**                | The fundamental measurement of spacetime distortion.                                                               |
| **Chirp**                       | The characteristic "swept-frequency" signature of a merger.                                                        |
| **Template Bank**               | A collection of theoretical waveforms covering the parameter space.                                                |
| **PSD ($S_n(f)$)**              | Characterizes the "sensitivity" of the detector at different frequencies.                                          |
| **$\chi^2$ (Chi-Squared) Test** | Used to distinguish real signals from "glitches" by checking if SNR is distributed across frequencies as expected. |