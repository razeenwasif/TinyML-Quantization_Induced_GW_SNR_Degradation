### Summary of Key Findings

The paper by Abbott et al. (2020) provides a comprehensive overview of the noise properties of the Advanced LIGO and Virgo detectors and the data analysis pipelines used to identify transient gravitational-wave signals.

- The foundation of signal detection relies on matched filtering, which correlates the incoming detector data with theoretical waveform templates to compute the optimal signal-to-noise ratio (SNR).
    
- To account for varying sensitivities across different frequency bands, the data is whitened by dividing the Fourier coefficients by an estimate of the noise amplitude spectral density.
    
- Because the detector noise is frequently non-stationary and non-Gaussian, the pipelines employ waveform consistency tests, such as chi-squared statistics, to measure how well the residuals match expected noise models after a signal is subtracted.
    
- Tools like time-frequency scalograms (Q-scans) and wavelet-based algorithms (BayesWave) are utilized to identify and reconstruct incoherent transient noise artifacts (glitches) that could otherwise corrupt the SNR or parameter estimation.
    

### Major Gaps and Limitations

While the LIGO-Virgo pipelines are highly robust, the paper highlights several vulnerabilities and computational limitations inherent in the current methodologies:

- Transient noise artifacts occur frequently (on the order of once per minute) and can easily mimic the morphology of true astrophysical signals, triggering false alarms if not aggressively vetoed or down-weighted.
    
- The data is subject to slow, continuous adiabatic drifts in the power spectrum, meaning the noise floor is rarely perfectly stationary over long periods.
    
- When the noise deviates significantly from the modeled Gaussian distribution, the analysis can suffer from systematic biases.
    
- The tools required to mitigate these issues and estimate parameters—such as the LALInference library utilizing parallel tempering Markov chain Monte Carlo (MCMC) or nested sampling—are highly computationally intensive.
    

### Connection to my Research Proposal

My research on **"Quantization-Induced SNR Degradation: Toward Signal-Preserving TinyML for Gravitational Wave Triggers"** directly addresses the computational bottlenecks outlined in Abbott et al., while navigating the strict SNR preservation requirements they define.

I can use this paper to establish the baseline fragility of gravitational-wave detection: Abbott et al. (2020) demonstrate that the matched filter's efficacy is highly sensitive to phase and amplitude precision, where even minor mismatches lead to significant drop-offs in the detection statistic. By proposing TinyML for low-latency triggers, I am aiming to solve the computational heaviness of traditional pipelines. However, I can cite this paper to argue that the model quantization inherent to TinyML introduces a new form of "algorithmic noise." If this quantization degrades the fidelity of the trigger's internal representations, it acts mathematically similarly to the instrumental noise and glitches described by Abbott et al., ultimately degrading the SNR.

My proposal can position Signal-Preserving TinyML as the necessary bridge: a method to achieve edge-computing speeds for early-warning triggers without introducing quantization artifacts that would fail the rigorous signal consistency and $\chi^2$ tests required by the LIGO-Virgo standards.