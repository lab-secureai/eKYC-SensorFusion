# Attestation and Sensor Fusion for Detecting Payload Injection in eKYC

This repository contains the simulation code, synthetic benchmark, and
experimental results for detecting pre-signing biometric payload injection
in Electronic Know Your Customer (eKYC) systems.

The proposed verification architecture combines:

- remote device and application attestation;
- nonce-based flash-response liveness;
- IMU–face motion consistency;
- latency, frame-rate, and timestamp-skew analysis;
- logistic-regression-based score fusion.

The implementation accompanies the study:

> **A Simulation-Based Evaluation of Attestation and Sensor Fusion for
> Detecting Capture-Layer Payload Injection in eKYC Systems**

## Overview

Modern eKYC systems commonly rely on face verification, liveness detection,
software protection, and device attestation. These mechanisms can detect
many presentation, replay, and application-layer attacks.

However, a stronger attacker may inject manipulated biometric content into
the acquisition pipeline before the final payload is signed. In this case,
the server may receive a payload containing apparently valid timestamps,
attestation evidence, and a hardware-backed signature, even though the
biometric content did not originate from a live physical user.

This repository evaluates a multi-layer defense in which attestation is
treated as one signal among several physical-consistency signals.

## Threat Model

The main threat is pre-signing biometric payload injection.

The attacker may partially control the client-side acquisition path,
including:

- a virtual camera stream;
- a modified camera interface;
- a compromised capture layer;
- an emulated or manipulated sensor stream;
- a real-time synthetic face pipeline.

The attacker is assumed not to break standard cryptographic primitives,
forge the server nonce, derive the protected signing key, or compromise the
verification server.

The repository contains defensive simulation and evaluation code only. It
does not contain offensive hardware-injection tooling.

## Proposed Feature Set

Each simulated eKYC session is represented using multiple verification
signals.

| Feature | Description |
|---|---|
| `attestation_score` | Simulated device and application integrity evidence |
| `attestation_pass` | Binary attestation decision |
| `flash_corr_zero_lag` | Zero-lag correlation between the challenge and facial brightness |
| `flash_corr_best_lag` | Maximum lag-aware flash-response correlation |
| `flash_best_lag_frames` | Estimated response delay in frames |
| `imu_face_corr` | Correlation between device motion and visual face motion |
| `latency_ms` | End-to-end challenge-response latency proxy |
| `fps` | Effective frame rate |
| `timestamp_skew_ms` | Temporal mismatch between video and sensor streams |

The final fusion model estimates an attack probability from the combined
feature vector.

## Simulated Session Categories

The benchmark contains 4,000 simulated sessions.

| Session type | Number of sessions |
|---|---:|
| Genuine | 1,000 |
| Software repackaging | 1,000 |
| Replay attack | 1,000 |
| Hardware or capture-layer injection | 1,000 |
| **Total** | **4,000** |

The simulated attack distributions intentionally overlap with genuine
sessions to avoid unrealistically perfect separation.

The hardware-injection category includes adaptive variants that may:

- partially respond to flash challenges;
- partially reproduce IMU consistency;
- introduce relay-like timing behavior;
- preserve valid or near-valid attestation evidence.

## Repository Structure

```text
.
├── notebooks/
│   ├── ekyc_adversarial_experiment_executed.ipynb
│   └── ekyc_adversarial_experiment_clean.ipynb
├── src/
│   └── ekyc_experiment.py
├── data/
│   └── synthetic_ekyc_sessions.csv
├── results/
│   ├── overall_metrics.csv
│   ├── confusion_matrix.csv
│   ├── operating_points.csv
│   ├── per_attack_type.csv
│   ├── ablation_results.csv
│   ├── ood_metrics.csv
│   ├── dataset_validation.json
│   └── result_manifest.csv
├── requirements.txt
├── LICENSE
└── README.md


## Notebook files
ekyc_adversarial_experiment_executed.ipynb contains the evaluated run
and stored outputs.
ekyc_adversarial_experiment_clean.ipynb contains the same experiment
without stored outputs and is intended for reproducible execution.
Python source

src/ekyc_experiment.py contains the code cells extracted from the final
notebook in their original execution order.

## Main Results

The full fusion model achieved the following closed-set results.

Metric	Value
ROC-AUC	0.9013
PR-AUC	0.9638
Equal error rate	18.56%
FAR at the EER threshold	18.44%
FRR at the EER threshold	18.67%
Attack detection rate	81.56%
Genuine acceptance rate	81.33%
Accuracy	81.50%

The confusion matrix at the EER threshold is:

Actual class	Predicted genuine	Predicted attack
Genuine	244	56
Attack	166	734
Operating-Point Analysis

The model was also evaluated under constrained false-acceptance settings.

Target FAR	Actual FAR	FRR	Attack detection	Genuine acceptance
20%	19.89%	17.00%	80.11%	83.00%
10%	9.89%	30.33%	90.11%	69.67%
5%	5.00%	46.00%	95.00%	54.00%
1%	1.00%	70.67%	99.00%	29.33%

These results illustrate a clear security–usability trade-off. Lower FAR
values improve attack detection but increase rejection of genuine users.

The operating points should be interpreted as simulation results rather
than recommended production thresholds.

Per-Attack Results
Session type	Mean attack probability	Detection rate	FAR
Software repackaging	0.6956	73.00%	27.00%
Replay attack	0.8941	93.67%	6.33%
Hardware injection	0.7356	78.00%	22.00%

For genuine sessions, the mean attack probability is 0.2546, and the
false rejection rate is 18.67%.

Ablation Study
Model	ROC-AUC	PR-AUC	EER	Hardware-injection FAR
Attestation only	0.7203	0.8984	34.06%	69.00%
Flash only	0.8561	0.9460	23.61%	25.67%
IMU only	0.8056	0.9238	26.94%	25.00%
Latency/FPS only	0.6936	0.8783	36.67%	22.33%
Full fusion	0.9013	0.9638	18.56%	22.00%

Under the simulated threat model, attestation alone performs poorly against
hardware-level injection. The full fusion model provides the strongest
overall result by combining complementary signals.

Out-of-Distribution Evaluation

An out-of-distribution experiment was conducted in which the model was
trained without hardware-injection samples and then evaluated on genuine
and unseen hardware-injection sessions.

Metric	Value
ROC-AUC	0.8535
PR-AUC	0.8543
EER	22.90%
Hardware-injection detection rate	77.10%
Hardware-injection FAR	22.90%
Genuine FRR	22.90%

The OOD result is weaker than the closed-set result, indicating that unseen
adaptive injection remains challenging.

##Metric Definitions

In this repository:

FAR is the proportion of attack sessions incorrectly accepted as
genuine.
FRR is the proportion of genuine sessions incorrectly rejected as
attacks.
Attack detection rate is the proportion of attacks correctly detected.
Genuine acceptance rate is the proportion of genuine sessions correctly
accepted.

Because attacks are encoded as the positive class, the security-oriented
FAR definition corresponds to the classifier false-negative rate.

##Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/ekyc-attestation-sensor-fusion.git
cd ekyc-attestation-sensor-fusion

Create a virtual environment:

python -m venv .venv

Activate the environment.

Linux or macOS:

source .venv/bin/activate

Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Running the Experiment

##Launch the clean notebook:

jupyter notebook notebooks/ekyc_adversarial_experiment_clean.ipynb

Alternatively, run the extracted Python source:

python src/ekyc_experiment.py

The notebook is the canonical executable artifact because some sections
preserve notebook-style execution order and intermediate variables.

Google Colab

The clean notebook can be uploaded directly to Google Colab.

After opening the notebook:

select Runtime → Run all;
wait for the synthetic benchmark and evaluation pipeline to complete;
inspect the exported CSV files in the results directory.
Reproducibility

The repository provides:

the final simulation notebook;
the extracted Python source;
the full synthetic session dataset;
closed-set evaluation metrics;
constrained-FAR operating points;
per-attack-type results;
ablation results;
OOD hardware-injection results;
dataset validation metadata.

The executed notebook is included for traceability. The clean notebook is
provided for rerunning the experiment without relying on stored outputs.

Intended Use

The project is intended for:

defensive eKYC security research;
simulation-based evaluation of sensor fusion;
analysis of attestation limitations;
research on biometric payload integrity;
education and reproducibility.

The project is not intended to certify a production eKYC system.

##Limitations

This study has several important limitations.

The benchmark is simulation-based.
No real Android or iOS application is implemented.
No real Play Integrity or App Attest integration is included.
Camera and IMU traces are synthetically generated.
No physical hardware-level injection attack is implemented.
The current evaluation does not include demographic fairness analysis.
Device heterogeneity and environmental variability are only approximated.
The reported results are not production-ready biometric error rates.

The results should therefore be interpreted as proof-of-concept evidence
for the proposed architecture.

##Security and Privacy

This repository contains synthetic data only. It does not contain real
identity documents, face videos, biometric templates, or personal sensor
records.

The implementation focuses on defensive verification and does not provide
instructions or tools for compromising camera drivers, secure hardware, or
mobile attestation services.

##Citation

When the associated paper becomes publicly available, use the following
template:

@inproceedings{hoa2026ekyc,
  author    = {Tran Thi Hoa and Nguyen Viet Nhat},
  title     = {A Simulation-Based Evaluation of Attestation and Sensor
               Fusion for Detecting Capture-Layer Payload Injection
               in eKYC Systems},
  booktitle = {Proceedings of the Vietnam Conference on Cryptography
               and Information Security},
  year      = {2026}
}

The title, venue, pages, DOI, and author information according to the
final camera-ready publication will be updated after review process.

##License

This project is released under the MIT License. See the
LICENSE file for details.
