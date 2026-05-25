### Literature Review: Characterization of Transient Noise in Advanced LIGO

The 2016 paper by Abbott et al. details the exhaustive detector characterization and noise analysis conducted to confirm that the first gravitational-wave detection (GW150914) was a genuine astrophysical event rather than an instrumental or environmental artifact. This paper is foundational for understanding the strict data quality requirements and the specific morphology of transient noise that my TinyML models will need to navigate.

**Key Findings on Noise Characterization:**

- **The PEM Array and Auxiliary Channels:** LIGO utilizes over 200,000 auxiliary channels and a Physical Environment Monitor (PEM) array consisting of seismometers, magnetometers, and microphones to witness environmental disturbances with a higher Signal-to-Noise Ratio (SNR) than the main gravitational-wave strain channel.
    
- **Correlated vs. Uncorrelated Noise:** The authors investigated global correlated noise (like lightning strikes triggering Schumann resonances) and local uncorrelated noise (like anthropogenic vibrations or equipment faults). They concluded that environmental influences were orders of magnitude too weak to account for the GW150914 signal.
    
- **The Threat of "Blip Transients":** A major hurdle identified in the paper is the presence of "blip transients"—short, teardrop-shaped noise bursts in the 30–250 Hz range of unknown instrumental origin. These blips frequently mimic the morphology of high-mass binary black hole mergers, creating significant background triggers that complicate the False Alarm Rate (FAR).
    
- **Data Quality Vetoes:** To mitigate noise, the pipeline employs strict "vetoes" (Categories 1, 2, and 3) to exclude data periods corrupted by known instrumental or environmental faults. A critical requirement is that these vetoes must be "safe," meaning they mathematically cannot accidentally remove a true gravitational-wave signal.
    

**Connection to Your Research Proposal:**

For your work on Signal-Preserving TinyML, this paper is highly relevant to your False Alarm Rate (FAR) and SNR-loss metrics. Because INT8 quantization acts as a physical perturbation to the signal, your custom inference kernels must be robust enough not to accidentally morph a true chirp signal into something that looks like a "blip transient" (which would cause a false negative) or, conversely, amplify the noise floor so that standard glitches trigger a false positive.