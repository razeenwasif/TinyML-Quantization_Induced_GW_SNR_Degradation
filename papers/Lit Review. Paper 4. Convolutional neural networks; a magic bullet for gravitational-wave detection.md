The 2019 paper by Gebhard et al. provides a highly critical, interdisciplinary evaluation of Convolutional Neural Networks (CNNs) for detecting compact binary coalescences (CBCs). While previous works enthusiastically framed CNNs as a direct replacement for traditional matched filtering, the authors argue for a more nuanced application.

**Key Findings and Innovations:**

- **Statistical Significance Limitations:** The authors mathematically argue that CNNs cannot be used to claim statistically significant gravitational-wave detections. Standard machine learning metrics, like False Positive Rates (FPR) derived from training populations, cannot be translated into the rigorous False Alarm Rates (FAR) required by physicists to reject the null hypothesis on a single event.
    
- **From Classification to Streaming:** Previous ML models relied on binary classification of fixed-length data snippets, which creates overlapping redundancies and interpretive ambiguities when applied to continuous streaming data. To solve this, the authors propose a fully convolutional architecture (lacking dense layers) capable of processing time series of arbitrary lengths.
    
- **The Role of CNNs as Triggers:** Despite their statistical limitations, CNNs offer inference times that are orders of magnitude faster than matched filtering, scaling efficiently across multiple detectors. Therefore, their ideal use-case is not as a final detection oracle, but as a real-time "trigger generator" to flag potential signals for slower, high-precision Bayesian follow-up.
    

**Adversarial Vulnerabilities:**

- A major contribution of the paper is exposing the "brittleness" of deep neural networks in this domain.
    
- Through adversarial attacks, the authors demonstrate that CNNs can be easily fooled into predicting a high-confidence gravitational-wave signal from inputs that contain no astrophysical features.
    
- Even pure noise, when subjected to microscopic, unphysical manipulations, can trigger a false positive, highlighting a severe failure mode for ML models operating on unpredictable, non-stationary detector noise.