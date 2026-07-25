# ============================================
# 0. Imports
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hashlib
import hmac
import base64
import json
import time
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    log_loss,
    brier_score_loss
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True


# ============================================
# 1. Mock remote attestation and signed payload
# ============================================

SERVER_SECRET = b"server-side-demo-secret-not-for-production"

def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

def make_nonce(session_id: str, server_secret: bytes = SERVER_SECRET) -> str:
    """
    Server creates an unpredictable nonce bound to the session.
    """
    ts = str(int(time.time_ns()))
    msg = f"{session_id}|{ts}|server_nonce".encode()
    digest = hmac.new(server_secret, msg, hashlib.sha256).digest()
    return b64url(digest[:24])

def mock_attestation_token(
    device_id: str,
    nonce: str,
    boot_state: str,
    app_hash: str,
    hardware_backed: bool,
    server_secret: bytes = SERVER_SECRET
) -> str:
    """
    Mock token format:
    header.payload.signature

    In a real system, this would be returned by a platform attestation API.
    """
    header = {"alg": "HS256", "typ": "MOCK-ATTESTATION"}
    payload = {
        "device_id": device_id,
        "nonce": nonce,
        "boot_state": boot_state,
        "app_hash": app_hash,
        "hardware_backed": hardware_backed,
        "iat": int(time.time())
    }
    header_b64 = b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(server_secret, signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{b64url(sig)}"

def verify_mock_attestation_token(token: str, expected_nonce: str, server_secret: bytes = SERVER_SECRET) -> dict:
    """
    Server verifies signature, nonce, and integrity claims.
    Returns a dict with pass/fail and a numeric score.
    """
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = b64url(hmac.new(server_secret, signing_input, hashlib.sha256).digest())

        padded_payload = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded_payload.encode()))

        sig_ok = hmac.compare_digest(expected_sig, sig_b64)
        nonce_ok = payload.get("nonce") == expected_nonce
        boot_ok = payload.get("boot_state") == "LOCKED"
        app_ok = payload.get("app_hash") == "KNOWN_GOOD_APP_HASH"
        hw_ok = payload.get("hardware_backed") is True

        passed = sig_ok and nonce_ok and boot_ok and app_ok and hw_ok

        score = (
            0.25 * sig_ok +
            0.25 * nonce_ok +
            0.20 * boot_ok +
            0.15 * app_ok +
            0.15 * hw_ok
        )
        return {"passed": bool(passed), "score": float(score), "claims": payload}
    except Exception as exc:
        return {"passed": False, "score": 0.0, "claims": {}, "error": str(exc)}

# Demo
session_id = "session_demo_001"
nonce = make_nonce(session_id)
token = mock_attestation_token(
    device_id="pixel_demo",
    nonce=nonce,
    boot_state="LOCKED",
    app_hash="KNOWN_GOOD_APP_HASH",
    hardware_backed=True
)
verify_mock_attestation_token(token, nonce)


# ============================================
# 2. Flash challenge generation
# ============================================

def challenge_from_nonce(nonce: str, T: int = 90, block: int = 10) -> np.ndarray:
    """
    Create a pseudo-random RGB flash challenge with piecewise-constant color blocks.
    Output shape: (T, 3), values in [0, 1].
    """
    colors = np.array([
        [1.00, 0.15, 0.15],
        [0.15, 1.00, 0.15],
        [0.15, 0.15, 1.00],
        [1.00, 1.00, 0.20],
        [0.20, 1.00, 1.00],
        [1.00, 0.20, 1.00],
        [0.85, 0.85, 0.85],
    ])

    seq = []
    n_blocks = int(np.ceil(T / block))
    for b in range(n_blocks):
        digest = hashlib.sha256(f"{nonce}|flash|{b}".encode()).digest()
        idx = digest[0] % len(colors)
        seq.extend([colors[idx]] * block)

    return np.array(seq[:T], dtype=float)

def flash_intensity(challenge_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB challenge to a scalar illumination signal.
    """
    weights = np.array([0.299, 0.587, 0.114])
    return challenge_rgb @ weights

def safe_corr(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def best_lag_corr(x, y, max_lag=8):
    """
    Estimate best cross-correlation and lag.
    Positive lag means y follows x with delay.
    """
    best = (-1.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = safe_corr(x[-lag:], y[:lag])
        elif lag > 0:
            c = safe_corr(x[:-lag], y[lag:])
        else:
            c = safe_corr(x, y)
        if c > best[0]:
            best = (c, lag)
    return best

demo_challenge = challenge_from_nonce(nonce, T=90, block=10)
demo_light = flash_intensity(demo_challenge)

plt.figure()
plt.plot(demo_light)
plt.title("Server-generated flash challenge intensity")
plt.xlabel("Frame")
plt.ylabel("Intensity")
plt.show()


# ============================================
# 3. Synthetic eKYC session simulator - V3 adversarial stress benchmark
# ============================================

@dataclass
class SessionResult:
    features: dict
    traces: dict

def generate_smooth_signal(T, scale=1.0, noise=0.05, rng=None):
    rng = rng or np.random.default_rng()
    steps = rng.normal(0, noise, size=T)
    signal = np.cumsum(steps)
    signal = signal - signal.mean()
    denom = np.std(signal) + 1e-8
    return scale * signal / denom

def simulate_session(session_type: str, session_idx: int, T: int = 90, rng=None) -> SessionResult:
    """
    V3 adversarial simulator.

    Design principle:
    - Do not let any single feature perfectly separate genuine and attack.
    - Genuine sessions may have poor lighting, weak camera, motion blur, and natural delay.
    - Hardware-injection attacks may partially adapt to flash and partially spoof IMU.
    - Attestation remains highly overlapping between genuine and hardware_injection.
    """
    rng = rng or np.random.default_rng()

    session_id = f"{session_type}_{session_idx:06d}"
    nonce = make_nonce(session_id)
    challenge = challenge_from_nonce(nonce, T=T, block=10)
    light = flash_intensity(challenge)

    # -----------------------------
    # A. Overlapping attestation model
    # -----------------------------
    if session_type == "genuine":
        attestation_score = float(np.clip(rng.normal(0.88, 0.14), 0.25, 1.00))
        attestation_pass = int(attestation_score > 0.78 and rng.random() > 0.08)

    elif session_type == "hardware_injection":
        # Main threat model: malicious payload can still be hardware-signed.
        attestation_score = float(np.clip(rng.normal(0.89, 0.13), 0.25, 1.00))
        attestation_pass = int(attestation_score > 0.78 and rng.random() > 0.12)

    elif session_type == "software_repack":
        attestation_score = float(np.clip(rng.normal(0.63, 0.24), 0.05, 1.00))
        attestation_pass = int(attestation_score > 0.78 and rng.random() > 0.40)

    elif session_type == "replay_attack":
        attestation_score = float(np.clip(rng.normal(0.70, 0.22), 0.05, 1.00))
        attestation_pass = int(attestation_score > 0.78 and rng.random() > 0.35)

    else:
        raise ValueError(f"Unknown session_type: {session_type}")

    # -----------------------------
    # B. Flash-response liveness with strong overlap
    # -----------------------------
    low_quality = rng.random() < (0.28 if session_type == "genuine" else 0.22)
    base_skin = 0.50 + rng.normal(0, 0.06)
    physiological = 0.025 * np.sin(np.linspace(0, 3 * np.pi, T) + rng.uniform(0, 1))
    ambient = (
        rng.normal(0, 0.05, T)
        + generate_smooth_signal(T, scale=rng.uniform(0.03, 0.10), noise=0.04, rng=rng)
    )

    if session_type == "genuine":
        gain = rng.normal(0.31 if low_quality else 0.37, 0.13)
        delay = int(rng.integers(0, 7 if low_quality else 4))
        noise = rng.normal(0, 0.10 if low_quality else 0.075, T)
        response = base_skin + gain * np.roll(light, delay) + physiological + ambient + noise

    elif session_type == "software_repack":
        # It may still use a live face, but with questionable app/device integrity.
        gain = rng.normal(0.30, 0.15)
        delay = int(rng.integers(0, 8))
        response = base_skin + gain * np.roll(light, delay) + 0.60 * physiological + ambient + rng.normal(0, 0.105, T)

    elif session_type == "replay_attack":
        # Some replay/deepfake pipelines apply global brightness adaptation.
        mode = rng.choice(["static", "adaptive"], p=[0.55, 0.45])
        if mode == "adaptive":
            gain = rng.normal(0.27, 0.14)
            delay = int(rng.integers(1, 10))
            response = (
                base_skin
                + gain * np.roll(light, delay)
                + ambient
                + generate_smooth_signal(T, scale=0.08, noise=0.04, rng=rng)
                + rng.normal(0, 0.11, T)
            )
        else:
            response = (
                base_skin
                + ambient
                + generate_smooth_signal(T, scale=0.20, noise=0.04, rng=rng)
                + rng.normal(0, 0.10, T)
            )

    elif session_type == "hardware_injection":
        # Adaptive attacker variants. None is a real attack implementation;
        # this only models measurement distributions for defensive evaluation.
        mode = rng.choice(
            ["naive", "adaptive_flash", "sensor_spoof", "relay_like"],
            p=[0.18, 0.38, 0.28, 0.16]
        )

        if mode == "naive":
            gain = rng.normal(0.20, 0.12)
            delay = int(rng.integers(5, 15))
            noise_sigma = 0.12
        elif mode == "adaptive_flash":
            gain = rng.normal(0.31, 0.13)
            delay = int(rng.integers(1, 9))
            noise_sigma = 0.115
        elif mode == "sensor_spoof":
            gain = rng.normal(0.30, 0.13)
            delay = int(rng.integers(1, 8))
            noise_sigma = 0.115
        else:
            gain = rng.normal(0.35, 0.12)
            delay = int(rng.integers(0, 5))
            noise_sigma = 0.11

        response = base_skin + gain * np.roll(light, delay) + 0.35 * physiological + ambient + rng.normal(0, noise_sigma, T)

    zero_lag_flash_corr = safe_corr(light, response)
    best_corr, best_lag = best_lag_corr(light, response, max_lag=15)

    # -----------------------------
    # C. IMU-face physical consistency with partial spoofing
    # -----------------------------
    imu_x = generate_smooth_signal(T, scale=1.0, noise=0.06, rng=rng)
    imu_y = generate_smooth_signal(T, scale=0.8, noise=0.05, rng=rng)
    imu_mag = np.sqrt(imu_x**2 + imu_y**2)

    if session_type == "genuine":
        coef = rng.normal(0.52, 0.20)
        face_motion = coef * imu_mag + generate_smooth_signal(T, scale=0.16, noise=0.04, rng=rng) + rng.normal(0, 0.30, T)

    elif session_type == "software_repack":
        coef = rng.normal(0.45, 0.22)
        face_motion = coef * imu_mag + generate_smooth_signal(T, scale=0.25, noise=0.04, rng=rng) + rng.normal(0, 0.34, T)

    elif session_type == "replay_attack":
        if rng.random() < 0.25:
            face_motion = rng.normal(0.35, 0.18) * imu_mag + generate_smooth_signal(T, scale=0.35, noise=0.04, rng=rng) + rng.normal(0, 0.35, T)
        else:
            face_motion = generate_smooth_signal(T, scale=0.75, noise=0.04, rng=rng) + rng.normal(0, 0.34, T)

    elif session_type == "hardware_injection":
        imu_attack_mode = rng.choice(["none", "partial", "spoofed"], p=[0.30, 0.45, 0.25])
        if imu_attack_mode == "none":
            face_motion = generate_smooth_signal(T, scale=0.75, noise=0.04, rng=rng) + rng.normal(0, 0.36, T)
        elif imu_attack_mode == "partial":
            face_motion = 0.35 * imu_mag + generate_smooth_signal(T, scale=0.45, noise=0.04, rng=rng) + rng.normal(0, 0.34, T)
        else:
            face_motion = 0.55 * imu_mag + generate_smooth_signal(T, scale=0.28, noise=0.04, rng=rng) + rng.normal(0, 0.32, T)

    imu_face_corr = safe_corr(imu_mag, face_motion)

    # -----------------------------
    # D. Latency/FPS with network and device overlap
    # -----------------------------
    if session_type == "genuine":
        latency_ms = rng.normal(155, 75)
        fps = rng.normal(24, 6)
        timestamp_skew_ms = abs(rng.normal(45, 45))

    elif session_type == "software_repack":
        latency_ms = rng.normal(180, 85)
        fps = rng.normal(23, 6)
        timestamp_skew_ms = abs(rng.normal(65, 60))

    elif session_type == "replay_attack":
        latency_ms = rng.normal(175, 80)
        fps = rng.normal(24, 6)
        timestamp_skew_ms = abs(rng.normal(60, 55))

    elif session_type == "hardware_injection":
        latency_ms = rng.normal(215, 95)
        fps = rng.normal(21, 7)
        timestamp_skew_ms = abs(rng.normal(80, 70))

    latency_ms = float(np.clip(latency_ms, 60, 650))
    fps = float(np.clip(fps, 6, 35))
    timestamp_skew_ms = float(np.clip(timestamp_skew_ms, 0, 350))

    label = 0 if session_type == "genuine" else 1

    features = {
        "session_id": session_id,
        "type": session_type,
        "label_attack": label,
        "attestation_pass": attestation_pass,
        "attestation_score": attestation_score,
        "flash_corr_zero_lag": zero_lag_flash_corr,
        "flash_corr_best_lag": best_corr,
        "flash_best_lag_frames": best_lag,
        "imu_face_corr": imu_face_corr,
        "latency_ms": latency_ms,
        "fps": fps,
        "timestamp_skew_ms": timestamp_skew_ms,
        "response_std": float(np.std(response)),
    }

    traces = {
        "nonce": nonce,
        "challenge_rgb": challenge,
        "light": light,
        "face_reflectance": response,
        "imu_mag": imu_mag,
        "face_motion": face_motion,
    }

    return SessionResult(features=features, traces=traces)

demo_real = simulate_session("genuine", 1, rng=rng)
demo_hw = simulate_session("hardware_injection", 2, rng=rng)

plt.figure()
plt.plot(demo_real.traces["light"], label="Flash challenge")
plt.plot(demo_real.traces["face_reflectance"], label="Genuine face reflectance")
plt.title("V3 genuine session: noisy real-world flash response")
plt.xlabel("Frame")
plt.legend()
plt.show()

plt.figure()
plt.plot(demo_hw.traces["light"], label="Flash challenge")
plt.plot(demo_hw.traces["face_reflectance"], label="Hardware injection: adaptive but imperfect response")
plt.title("V3 hardware injection: adversarial stress trace")
plt.xlabel("Frame")
plt.legend()
plt.show()


# ============================================
# 4. Dataset generation
# ============================================

N_PER_TYPE = 1000
SESSION_TYPES = ["genuine", "software_repack", "replay_attack", "hardware_injection"]

rows = []
example_traces = {}

for stype in SESSION_TYPES:
    for i in range(N_PER_TYPE):
        result = simulate_session(stype, i, T=90, rng=rng)
        rows.append(result.features)

        if stype not in example_traces:
            example_traces[stype] = result.traces

df = pd.DataFrame(rows)
df.head()


# Basic sanity checks
print(df["type"].value_counts())
print()
print(df.groupby("type")[[
    "attestation_score",
    "flash_corr_zero_lag",
    "flash_corr_best_lag",
    "flash_best_lag_frames",
    "imu_face_corr",
    "latency_ms",
    "fps",
    "timestamp_skew_ms"
]].mean().round(3))


# ============================================
# 5. Visualization
# ============================================

def boxplot_by_type(metric, title, ylabel=None):
    data = [df[df["type"] == stype][metric].values for stype in SESSION_TYPES]
    plt.figure()
    plt.boxplot(data, labels=SESSION_TYPES, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel or metric)
    plt.xticks(rotation=20)
    plt.show()

boxplot_by_type("attestation_score", "Remote attestation score by session type", "Score")
boxplot_by_type("flash_corr_zero_lag", "Flash-response zero-lag correlation", "Correlation")
boxplot_by_type("imu_face_corr", "IMU-face motion correlation", "Correlation")
boxplot_by_type("latency_ms", "Latency analysis", "Milliseconds")
boxplot_by_type("fps", "FPS degradation analysis", "Frames per second")


# ============================================
# 5B. Feature-overlap diagnostics
# ============================================
# In a credible synthetic benchmark, features should overlap across classes.
# If every feature separates perfectly, the evaluation is too easy.

overlap_metrics = [
    "attestation_score",
    "flash_corr_zero_lag",
    "flash_corr_best_lag",
    "imu_face_corr",
    "latency_ms",
    "fps",
    "timestamp_skew_ms"
]

summary = df.groupby("type")[overlap_metrics].agg(["mean", "std"]).round(3)
summary


# Example sensor traces
for stype in ["genuine", "hardware_injection"]:
    tr = example_traces[stype]
    plt.figure()
    plt.plot(tr["imu_mag"], label="IMU magnitude")
    plt.plot(tr["face_motion"], label="Face motion")
    plt.title(f"IMU-face temporal trace: {stype}")
    plt.xlabel("Frame")
    plt.legend()
    plt.show()


# ============================================
# 6. Fusion model training
# ============================================

# Conservative feature set:
# response_std is intentionally excluded from the main model because it can become
# a synthetic shortcut in simulation-based studies.
FEATURES = [
    "attestation_score",
    "attestation_pass",
    "flash_corr_zero_lag",
    "flash_corr_best_lag",
    "flash_best_lag_frames",
    "imu_face_corr",
    "latency_ms",
    "fps",
    "timestamp_skew_ms",
]

X = df[FEATURES]
y = df["label_attack"].astype(int)

X_train, X_test, y_train, y_test, type_train, type_test = train_test_split(
    X, y, df["type"],
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["type"]
)

fusion_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
])

fusion_model.fit(X_train, y_train)
proba_attack = fusion_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, proba_attack)
pr_auc = average_precision_score(y_test, proba_attack)
ll = log_loss(y_test, proba_attack)
brier = brier_score_loss(y_test, proba_attack)

print(f"Fusion ROC-AUC : {roc_auc:.4f}")
print(f"Fusion PR-AUC  : {pr_auc:.4f}")
print(f"LogLoss        : {ll:.4f}")
print(f"Brier score    : {brier:.4f}")


# ============================================
# 7. EER and threshold metrics
# ============================================

def compute_eer(y_true, scores_attack):
    """
    y_true: 1 = attack, 0 = genuine
    scores_attack: higher means more likely attack
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores_attack)
    # For attack detector:
    # fpr = genuine rejected as attack = FRR
    # fnr = attack accepted as genuine = FAR
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx]), fpr, tpr, fnr

eer, eer_threshold, fpr, tpr, fnr = compute_eer(y_test, proba_attack)
print(f"EER              : {eer:.4f}")
print(f"EER threshold    : {eer_threshold:.4f}")

chosen_threshold = eer_threshold
pred_attack = (proba_attack >= chosen_threshold).astype(int)

cm = confusion_matrix(y_test, pred_attack, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

FRR = fp / (fp + tn)
FAR = fn / (fn + tp)
TPR = tp / (tp + fn)
TNR = tn / (tn + fp)

print()
print("Confusion matrix labels [genuine=0, attack=1]:")
print(cm)
print()
print(f"FAR / attack accepted     : {FAR:.4f}")
print(f"FRR / genuine rejected    : {FRR:.4f}")
print(f"TPR / attack detected     : {TPR:.4f}")
print(f"TNR / genuine accepted    : {TNR:.4f}")
print()
print(classification_report(y_test, pred_attack, target_names=["genuine", "attack"]))


# ============================================
# Operating points under target FAR
# ============================================

def metrics_at_threshold(y_true, scores_attack, threshold):
    pred_attack = (scores_attack >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred_attack, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    far = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    frr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "threshold": threshold,
        "FAR": far,
        "FRR": frr,
        "TPR_attack_detection": tpr,
        "TNR_genuine_acceptance": tnr,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

def find_threshold_for_target_far(y_true, scores_attack, target_far):
    thresholds = np.unique(scores_attack)
    rows = []

    for thr in thresholds:
        m = metrics_at_threshold(y_true, scores_attack, thr)
        rows.append(m)

    result = pd.DataFrame(rows)

    # Vì score càng cao càng giống attack,
    # threshold càng thấp thì block càng nhiều attack => FAR giảm nhưng FRR tăng.
    feasible = result[result["FAR"] <= target_far].copy()

    if len(feasible) == 0:
        return None

    # Chọn threshold có FRR thấp nhất trong nhóm đạt target FAR.
    best = feasible.sort_values(["FRR", "FAR"]).iloc[0]
    return best

target_fars = [0.20, 0.10, 0.05, 0.01]

operating_rows = []
for target in target_fars:
    best = find_threshold_for_target_far(y_test, proba_attack, target)
    if best is not None:
        operating_rows.append({
            "target_FAR": target,
            "threshold": best["threshold"],
            "actual_FAR": best["FAR"],
            "FRR": best["FRR"],
            "attack_detection_rate_TPR": best["TPR_attack_detection"],
            "genuine_acceptance_TNR": best["TNR_genuine_acceptance"]
        })

operating_points = pd.DataFrame(operating_rows)
operating_points.round(4)


# ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"Fusion ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.scatter([eer], [1 - eer], label=f"EER = {eer:.4f}")
plt.title("ROC curve for attack detection")
plt.xlabel("FRR-like FPR: genuine rejected as attack")
plt.ylabel("Attack detection rate")
plt.legend()
plt.show()

# Precision-recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, proba_attack)
plt.figure()
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
plt.title("Precision-recall curve for attack detection")
plt.xlabel("Recall / attack detection rate")
plt.ylabel("Precision")
plt.legend()
plt.show()


# ============================================
# 8. Per-attack-type FAR and detection rate
# ============================================

test_result = X_test.copy()
test_result["true_label"] = y_test.values
test_result["type"] = type_test.values
test_result["proba_attack"] = proba_attack
test_result["pred_attack"] = pred_attack

rows = []
for stype in SESSION_TYPES:
    sub = test_result[test_result["type"] == stype]
    if stype == "genuine":
        false_reject = (sub["pred_attack"] == 1).mean()
        rows.append({
            "type": stype,
            "n": len(sub),
            "mean_proba_attack": sub["proba_attack"].mean(),
            "false_reject_rate_FRR": false_reject,
            "attack_detection_rate": np.nan,
            "false_accept_rate_FAR": np.nan
        })
    else:
        detection = (sub["pred_attack"] == 1).mean()
        far_type = (sub["pred_attack"] == 0).mean()
        rows.append({
            "type": stype,
            "n": len(sub),
            "mean_proba_attack": sub["proba_attack"].mean(),
            "false_reject_rate_FRR": np.nan,
            "attack_detection_rate": detection,
            "false_accept_rate_FAR": far_type
        })

per_type = pd.DataFrame(rows)
per_type.round(4)


# ============================================
# 9. Ablation study
# ============================================

FEATURE_GROUPS = {
    "Attestation only": ["attestation_score", "attestation_pass"],
    "Flash only": ["flash_corr_zero_lag", "flash_corr_best_lag", "flash_best_lag_frames", "response_std"],
    "IMU only": ["imu_face_corr"],
    "Latency/FPS only": ["latency_ms", "fps", "timestamp_skew_ms"],
    "Full fusion": FEATURES,
}

def train_eval_feature_group(feature_list):
    Xg = df[feature_list]
    X_train_g, X_test_g, y_train_g, y_test_g, type_train_g, type_test_g = train_test_split(
        Xg, y, df["type"],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["type"]
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    model.fit(X_train_g, y_train_g)
    s = model.predict_proba(X_test_g)[:, 1]
    eer_g, thr_g, _, _, _ = compute_eer(y_test_g, s)
    pred_g = (s >= thr_g).astype(int)

    cm_g = confusion_matrix(y_test_g, pred_g, labels=[0, 1])
    tn, fp, fn, tp = cm_g.ravel()
    far = fn / (fn + tp)
    frr = fp / (fp + tn)

    tmp = pd.DataFrame({
        "type": type_test_g.values,
        "pred": pred_g
    })
    hw = tmp[tmp["type"] == "hardware_injection"]
    hw_far = (hw["pred"] == 0).mean()

    return {
        "ROC_AUC": roc_auc_score(y_test_g, s),
        "PR_AUC": average_precision_score(y_test_g, s),
        "EER": eer_g,
        "FAR_all_attacks": far,
        "FRR_genuine": frr,
        "FAR_hardware_injection": hw_far
    }

ablation = pd.DataFrame([
    {"model": name, **train_eval_feature_group(feats)}
    for name, feats in FEATURE_GROUPS.items()
])

ablation.round(4)


# Bar chart for ablation EER and FAR_hardware_injection
plt.figure()
plt.bar(ablation["model"], ablation["EER"])
plt.title("Ablation study: Equal Error Rate")
plt.ylabel("EER")
plt.xticks(rotation=25, ha="right")
plt.show()

plt.figure()
plt.bar(ablation["model"], ablation["FAR_hardware_injection"])
plt.title("Ablation study: FAR on hardware-injection attacks")
plt.ylabel("FAR for hardware injection")
plt.xticks(rotation=25, ha="right")
plt.show()



# ============================================
# 9B. Optional stress test: leave-one-attack-type-out
# ============================================
# This is useful for a stronger paper:
# train on software_repack + replay_attack + genuine,
# then test generalization on unseen hardware_injection.

train_mask = df["type"].isin(["genuine", "software_repack", "replay_attack"])
test_mask = df["type"].isin(["genuine", "hardware_injection"])

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

X_train_ood = train_df[FEATURES]
y_train_ood = train_df["label_attack"].astype(int)

X_test_ood = test_df[FEATURES]
y_test_ood = test_df["label_attack"].astype(int)
type_test_ood = test_df["type"]

ood_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
])

ood_model.fit(X_train_ood, y_train_ood)
ood_scores = ood_model.predict_proba(X_test_ood)[:, 1]

ood_auc = roc_auc_score(y_test_ood, ood_scores)
ood_pr_auc = average_precision_score(y_test_ood, ood_scores)
ood_eer, ood_thr, _, _, _ = compute_eer(y_test_ood, ood_scores)
ood_pred = (ood_scores >= ood_thr).astype(int)

ood_summary = []
for stype in ["genuine", "hardware_injection"]:
    sub_mask = (type_test_ood.values == stype)
    if stype == "genuine":
        ood_summary.append({
            "type": stype,
            "n": int(sub_mask.sum()),
            "mean_proba_attack": float(ood_scores[sub_mask].mean()),
            "FRR": float((ood_pred[sub_mask] == 1).mean()),
            "attack_detection_rate": np.nan,
            "FAR": np.nan,
        })
    else:
        ood_summary.append({
            "type": stype,
            "n": int(sub_mask.sum()),
            "mean_proba_attack": float(ood_scores[sub_mask].mean()),
            "FRR": np.nan,
            "attack_detection_rate": float((ood_pred[sub_mask] == 1).mean()),
            "FAR": float((ood_pred[sub_mask] == 0).mean()),
        })

ood_results = pd.DataFrame(ood_summary)

print(f"OOD ROC-AUC : {ood_auc:.4f}")
print(f"OOD PR-AUC  : {ood_pr_auc:.4f}")
print(f"OOD EER     : {ood_eer:.4f}")
ood_results.round(4)


# ============================================
# 10. Export outputs
# ============================================

df.to_csv("synthetic_ekyc_sessions.csv", index=False)
ablation.to_csv("ablation_results.csv", index=False)
per_type.to_csv("per_type_results.csv", index=False)

print("Saved:")
print("- synthetic_ekyc_sessions.csv")
print("- ablation_results.csv")
print("- per_type_results.csv")
