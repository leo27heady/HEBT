"""
Cross-Attention Ablation Analysis
Compare training with vs without parent→child cross-attention conditioning.
Generates plots and a text report.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

# ── Load CSV ──────────────────────────────────────────────────────────────── #

def load_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        data = defaultdict(list)
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)
    return {k: np.array(v) for k, v in data.items()}


WITH = load_csv("logs/with cross att.csv")
WITHOUT = load_csv("logs/without cross att.csv")

OUT_DIR = "logs/ablation_report"
os.makedirs(OUT_DIR, exist_ok=True)

STAGES = ["s0_s1", "s1_s2", "s2_s3"]
STAGE_LABELS = {"s0_s1": "Stage 0 (s1, 32×32)", "s1_s2": "Stage 1 (s2, 16×16)", "s2_s3": "Stage 2 (s3, 8×8)"}

# ── Smoothing helper ─────────────────────────────────────────────────────── #

def ema(arr, alpha=0.05):
    """Exponential moving average."""
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

# ── 1. Total loss ─────────────────────────────────────────────────────────── #

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
steps = WITH["step"]
ax.plot(steps, ema(WITH["loss_total"]), label="With cross-attn", color="#2196F3", linewidth=1.5)
ax.plot(steps, ema(WITHOUT["loss_total"]), label="Without cross-attn", color="#F44336", linewidth=1.5)
ax.set_xlabel("Step"); ax.set_ylabel("Total Loss (EMA)")
ax.set_title("Total Loss: With vs Without Cross-Attention")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_total_loss.png"), dpi=150)
plt.close(fig)
print("  [1/8] Total loss plot saved")

# ── 2. Per-stage final reconstruction ─────────────────────────────────────── #

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (stage, label) in enumerate(STAGE_LABELS.items()):
    col = f"{stage}_final_recon"
    axes[i].plot(steps, ema(WITH[col]), label="With cross-attn", color="#2196F3", linewidth=1.5)
    axes[i].plot(steps, ema(WITHOUT[col]), label="Without cross-attn", color="#F44336", linewidth=1.5)
    axes[i].set_xlabel("Step"); axes[i].set_ylabel("Final Recon (SmoothL1)")
    axes[i].set_title(f"Final Reconstruction — {label}")
    axes[i].legend(); axes[i].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_per_stage_final_recon.png"), dpi=150)
plt.close(fig)
print("  [2/8] Per-stage final recon plot saved")

# ── 3. Per-stage energy gap ───────────────────────────────────────────────── #

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (stage, label) in enumerate(STAGE_LABELS.items()):
    col = f"{stage}_energy_gap"
    axes[i].plot(steps, ema(WITH[col]), label="With cross-attn", color="#2196F3", linewidth=1.5)
    axes[i].plot(steps, ema(WITHOUT[col]), label="Without cross-attn", color="#F44336", linewidth=1.5)
    axes[i].set_xlabel("Step"); axes[i].set_ylabel("Energy Gap")
    axes[i].set_title(f"Energy Gap (init−final) — {label}")
    axes[i].legend(); axes[i].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_per_stage_energy_gap.png"), dpi=150)
plt.close(fig)
print("  [3/8] Per-stage energy gap plot saved")

# ── 4. Per-stage gradient norm ────────────────────────────────────────────── #

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (stage, label) in enumerate(STAGE_LABELS.items()):
    col = f"{stage}_grad_norm"
    axes[i].plot(steps, ema(WITH[col]), label="With cross-attn", color="#2196F3", linewidth=1.5)
    axes[i].plot(steps, ema(WITHOUT[col]), label="Without cross-attn", color="#F44336", linewidth=1.5)
    axes[i].set_xlabel("Step"); axes[i].set_ylabel("Gradient Norm")
    axes[i].set_title(f"Gradient Norm — {label}")
    axes[i].legend(); axes[i].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_per_stage_grad_norm.png"), dpi=150)
plt.close(fig)
print("  [4/8] Per-stage gradient norm plot saved")

# ── 5. Alpha (learnable step size) ────────────────────────────────────────── #

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (stage, label) in enumerate(STAGE_LABELS.items()):
    col = f"{stage}_alpha"
    axes[i].plot(steps, WITH[col], label="With cross-attn", color="#2196F3", linewidth=1.5)
    axes[i].plot(steps, WITHOUT[col], label="Without cross-attn", color="#F44336", linewidth=1.5)
    axes[i].set_xlabel("Step"); axes[i].set_ylabel("Alpha")
    axes[i].set_title(f"Learnable Step Size (α) — {label}")
    axes[i].legend(); axes[i].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_per_stage_alpha.png"), dpi=150)
plt.close(fig)
print("  [5/8] Alpha plot saved")

# ── 6. Reconstruction vs Baseline (copy-last-frame) ──────────────────────── #

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (stage, label) in enumerate(STAGE_LABELS.items()):
    recon_w = ema(WITH[f"{stage}_final_recon"])
    recon_wo = ema(WITHOUT[f"{stage}_final_recon"])
    base_w = ema(WITH[f"{stage}_baseline_copy"])
    base_wo = ema(WITHOUT[f"{stage}_baseline_copy"])
    # Relative improvement vs baseline: (baseline - recon) / baseline
    rel_w = (base_w - recon_w) / np.clip(base_w, 1e-8, None)
    rel_wo = (base_wo - recon_wo) / np.clip(base_wo, 1e-8, None)
    axes[i].plot(steps, rel_w * 100, label="With cross-attn", color="#2196F3", linewidth=1.5)
    axes[i].plot(steps, rel_wo * 100, label="Without cross-attn", color="#F44336", linewidth=1.5)
    axes[i].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[i].set_xlabel("Step"); axes[i].set_ylabel("Improvement vs copy-last (%)")
    axes[i].set_title(f"Recon vs Copy-Last Baseline — {label}")
    axes[i].legend(); axes[i].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_recon_vs_baseline.png"), dpi=150)
plt.close(fig)
print("  [6/8] Recon vs baseline plot saved")

# ── 7. Convergence speed: steps to reach threshold ───────────────────────── #

def steps_to_threshold(arr, threshold):
    smoothed = ema(arr, alpha=0.02)
    idx = np.where(smoothed <= threshold)[0]
    return int(idx[0]) if len(idx) > 0 else None

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
thresholds_s1 = np.linspace(0.02, 0.10, 20)
conv_w = []
conv_wo = []
for th in thresholds_s1:
    sw = steps_to_threshold(WITH["s0_s1_final_recon"], th)
    swo = steps_to_threshold(WITHOUT["s0_s1_final_recon"], th)
    conv_w.append(sw if sw is not None else 2000)
    conv_wo.append(swo if swo is not None else 2000)
ax.plot(thresholds_s1, conv_w, "o-", label="With cross-attn", color="#2196F3", markersize=5)
ax.plot(thresholds_s1, conv_wo, "s-", label="Without cross-attn", color="#F44336", markersize=5)
ax.set_xlabel("Final Recon Threshold (SmoothL1)"); ax.set_ylabel("Steps to Reach Threshold")
ax.set_title("Convergence Speed — Stage 0 (s1, 32×32, finest)")
ax.legend(); ax.grid(True, alpha=0.3)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "07_convergence_speed_s1.png"), dpi=150)
plt.close(fig)
print("  [7/8] Convergence speed plot saved")

# ── 8. Summary bar chart: final metrics ───────────────────────────────────── #

last_n = 50  # average last 50 steps

def avg_last(d, col, n=last_n):
    return float(np.mean(d[col][-n:]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
labels_bar = ["With\ncross-attn", "Without\ncross-attn"]
colors_bar = ["#2196F3", "#F44336"]

# Final recon per stage
recons_w = [avg_last(WITH, f"{s}_final_recon") for s in STAGES]
recons_wo = [avg_last(WITHOUT, f"{s}_final_recon") for s in STAGES]

x = np.arange(len(STAGES))
width = 0.35
axes[0].bar(x - width/2, recons_w, width, label="With cross-attn", color="#2196F3", alpha=0.85)
axes[0].bar(x + width/2, recons_wo, width, label="Without cross-attn", color="#F44336", alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels([STAGE_LABELS[s].split("(")[1].rstrip(")") for s in STAGES])
axes[0].set_ylabel("Final Recon (lower=better)"); axes[0].set_title("Final Reconstruction (avg last 50 steps)")
axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")
# Add value labels
for j, (vw, vwo) in enumerate(zip(recons_w, recons_wo)):
    axes[0].text(j - width/2, vw + 0.002, f"{vw:.4f}", ha="center", va="bottom", fontsize=8)
    axes[0].text(j + width/2, vwo + 0.002, f"{vwo:.4f}", ha="center", va="bottom", fontsize=8)

# Energy gap per stage
eg_w = [avg_last(WITH, f"{s}_energy_gap") for s in STAGES]
eg_wo = [avg_last(WITHOUT, f"{s}_energy_gap") for s in STAGES]
axes[1].bar(x - width/2, eg_w, width, label="With cross-attn", color="#2196F3", alpha=0.85)
axes[1].bar(x + width/2, eg_wo, width, label="Without cross-attn", color="#F44336", alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels([STAGE_LABELS[s].split("(")[1].rstrip(")") for s in STAGES])
axes[1].set_ylabel("Energy Gap (higher=better MCMC)"); axes[1].set_title("Energy Gap (avg last 50 steps)")
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
for j, (vw, vwo) in enumerate(zip(eg_w, eg_wo)):
    axes[1].text(j - width/2, vw + 0.002, f"{vw:.4f}", ha="center", va="bottom", fontsize=8)
    axes[1].text(j + width/2, vwo + 0.002, f"{vwo:.4f}", ha="center", va="bottom", fontsize=8)

# Grad norms
gn_w = [avg_last(WITH, f"{s}_grad_norm") for s in STAGES]
gn_wo = [avg_last(WITHOUT, f"{s}_grad_norm") for s in STAGES]
axes[2].bar(x - width/2, gn_w, width, label="With cross-attn", color="#2196F3", alpha=0.85)
axes[2].bar(x + width/2, gn_wo, width, label="Without cross-attn", color="#F44336", alpha=0.85)
axes[2].set_xticks(x); axes[2].set_xticklabels([STAGE_LABELS[s].split("(")[1].rstrip(")") for s in STAGES])
axes[2].set_ylabel("Gradient Norm"); axes[2].set_title("Gradient Norm (avg last 50 steps)")
axes[2].legend(); axes[2].grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "08_summary_bars.png"), dpi=150)
plt.close(fig)
print("  [8/8] Summary bar chart saved")

# ── Text report ───────────────────────────────────────────────────────────── #

report_lines = []
report_lines.append("=" * 80)
report_lines.append("HVEBT CROSS-ATTENTION ABLATION REPORT")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Experiment: 3-stage HVEBT (s1 32×32, s2 16×16, s3 8×8)")
report_lines.append("Steps: 2000 per run")
report_lines.append("Conditions:")
report_lines.append("  A) WITH cross-attention: top-down parent KV conditioning")
report_lines.append("  B) WITHOUT cross-attention: each stage trains independently")
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("1. FINAL RECONSTRUCTION (avg last 50 steps, lower = better)")
report_lines.append("-" * 80)
report_lines.append(f"{'Stage':<30} {'With XAttn':>12} {'Without XAttn':>14} {'Δ':>10} {'Winner':>10}")
for s, label in STAGE_LABELS.items():
    rw = avg_last(WITH, f"{s}_final_recon")
    rwo = avg_last(WITHOUT, f"{s}_final_recon")
    delta = rw - rwo
    pct = (rwo - rw) / max(rwo, 1e-8) * 100
    winner = "WITH" if rw < rwo else "WITHOUT" if rwo < rw else "TIE"
    report_lines.append(f"{label:<30} {rw:>12.6f} {rwo:>14.6f} {delta:>+10.6f} {winner:>10} ({pct:+.1f}%)")

report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("2. ENERGY GAP (avg last 50 steps, higher = MCMC working better)")
report_lines.append("-" * 80)
report_lines.append(f"{'Stage':<30} {'With XAttn':>12} {'Without XAttn':>14} {'Δ':>10}")
for s, label in STAGE_LABELS.items():
    ew = avg_last(WITH, f"{s}_energy_gap")
    ewo = avg_last(WITHOUT, f"{s}_energy_gap")
    delta = ew - ewo
    report_lines.append(f"{label:<30} {ew:>12.6f} {ewo:>14.6f} {delta:>+10.6f}")

report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("3. TOTAL LOSS (avg last 50 steps)")
report_lines.append("-" * 80)
lw = avg_last(WITH, "loss_total")
lwo = avg_last(WITHOUT, "loss_total")
report_lines.append(f"  With cross-attn:    {lw:.6f}")
report_lines.append(f"  Without cross-attn: {lwo:.6f}")
report_lines.append(f"  Δ (with - without): {lw - lwo:+.6f} ({'WITH better' if lw < lwo else 'WITHOUT better'})")

report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("4. CONVERGENCE SPEED — Stage 0 (s1, 32×32, finest)")
report_lines.append("-" * 80)
for th in [0.10, 0.08, 0.06, 0.05, 0.04, 0.035]:
    sw = steps_to_threshold(WITH["s0_s1_final_recon"], th)
    swo = steps_to_threshold(WITHOUT["s0_s1_final_recon"], th)
    sw_str = f"{sw}" if sw is not None else ">2000"
    swo_str = f"{swo}" if swo is not None else ">2000"
    speedup = ""
    if sw is not None and swo is not None and sw > 0:
        speedup = f"  ({swo/sw:.2f}x)"
    elif sw is not None and swo is None:
        speedup = "  (WITH converges, WITHOUT does NOT)"
    report_lines.append(f"  Recon ≤ {th:.3f}:  WITH @ step {sw_str:<8}  WITHOUT @ step {swo_str:<8}{speedup}")

report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("5. CONVERGENCE SPEED — Stage 1 (s2, 16×16)")
report_lines.append("-" * 80)
for th in [0.05, 0.03, 0.02, 0.015, 0.010, 0.008]:
    sw = steps_to_threshold(WITH["s1_s2_final_recon"], th)
    swo = steps_to_threshold(WITHOUT["s1_s2_final_recon"], th)
    sw_str = f"{sw}" if sw is not None else ">2000"
    swo_str = f"{swo}" if swo is not None else ">2000"
    speedup = ""
    if sw is not None and swo is not None and sw > 0:
        speedup = f"  ({swo/sw:.2f}x)"
    elif sw is not None and swo is None:
        speedup = "  (WITH converges, WITHOUT does NOT)"
    report_lines.append(f"  Recon ≤ {th:.3f}:  WITH @ step {sw_str:<8}  WITHOUT @ step {swo_str:<8}{speedup}")

report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("6. ALPHA EVOLUTION (learnable MCMC step size)")
report_lines.append("-" * 80)
for s, label in STAGE_LABELS.items():
    aw_start = float(WITH[f"{s}_alpha"][0])
    aw_end = avg_last(WITH, f"{s}_alpha")
    awo_start = float(WITHOUT[f"{s}_alpha"][0])
    awo_end = avg_last(WITHOUT, f"{s}_alpha")
    report_lines.append(f"  {label}")
    report_lines.append(f"    With:    {aw_start:.1f} → {aw_end:.1f} (Δ={aw_end - aw_start:+.1f})")
    report_lines.append(f"    Without: {awo_start:.1f} → {awo_end:.1f} (Δ={awo_end - awo_start:+.1f})")

report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("7. STAGE 2 (s3, APEX) — CONTROL CHECK")
report_lines.append("-" * 80)
report_lines.append("Stage 2 is the apex (coarsest). It NEVER has cross-attention in either")
report_lines.append("condition. Differences here reflect only statistical noise / indirect effects.")
rw3 = avg_last(WITH, "s2_s3_final_recon")
rwo3 = avg_last(WITHOUT, "s2_s3_final_recon")
report_lines.append(f"  Final recon WITH:    {rw3:.6f}")
report_lines.append(f"  Final recon WITHOUT: {rwo3:.6f}")
report_lines.append(f"  Δ: {rw3 - rwo3:+.6f}")

report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("8. TRAINING SPEED")
report_lines.append("-" * 80)
tw = avg_last(WITH, "secs")
two = avg_last(WITHOUT, "secs")
report_lines.append(f"  Avg secs/step WITH:    {tw:.3f}")
report_lines.append(f"  Avg secs/step WITHOUT: {two:.3f}")
report_lines.append(f"  Overhead from cross-attn: {(tw - two)/max(two, 1e-8)*100:+.1f}%")

report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("CONCLUSION")
report_lines.append("=" * 80)
# Determine which stages benefited
s1_better = avg_last(WITH, "s0_s1_final_recon") < avg_last(WITHOUT, "s0_s1_final_recon")
s2_better = avg_last(WITH, "s1_s2_final_recon") < avg_last(WITHOUT, "s1_s2_final_recon")
total_better = lw < lwo

if s1_better and s2_better:
    verdict = "CROSS-ATTENTION IS BENEFICIAL"
    detail = ("Both child stages (s1 and s2) achieve lower reconstruction error with\n"
              "cross-attention, confirming that top-down parent conditioning improves\n"
              "prediction quality at finer scales.")
elif s1_better or s2_better:
    better_stage = "s1 (32×32)" if s1_better else "s2 (16×16)"
    verdict = "CROSS-ATTENTION IS PARTIALLY BENEFICIAL"
    detail = f"Only {better_stage} benefits. The other stage shows no clear improvement."
elif total_better:
    verdict = "MIXED RESULTS — marginal total loss improvement"
    detail = "Total loss is lower with cross-attention but individual stage metrics are mixed."
else:
    verdict = "CROSS-ATTENTION SHOWS NO CLEAR BENEFIT"
    detail = ("Neither child stage achieves measurably better reconstruction with\n"
              "cross-attention. The architecture may need rethinking:\n"
              "  - The detached KV may strip useful gradient signal\n"
              "  - The 1-parent-per-child mask may be too restrictive\n"
              "  - The MCMC dynamics may already converge well without conditioning")

report_lines.append(f"\n  >>> {verdict} <<<\n")
report_lines.append(detail)
report_lines.append("")

report_text = "\n".join(report_lines)
report_path = os.path.join(OUT_DIR, "ablation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n{'='*60}")
print(report_text)
print(f"\nReport saved to: {report_path}")
print(f"Plots saved to:  {OUT_DIR}/")
