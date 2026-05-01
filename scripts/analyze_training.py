"""
Analyze HVEBT training results from CSV log.
Generates a markdown report with convergence analysis.

Usage:
    python scripts/analyze_training.py --log logs/hvebt_2stage_100/train_log.csv --output logs/hvebt_2stage_100/analysis.md
"""
import argparse
import os
import sys

import pandas as pd


def analyze(log_path: str, output_path: str):
    df = pd.read_csv(log_path)
    n_steps = len(df)

    # Detect stages from column names
    stage_cols = [c for c in df.columns if c.startswith("s0_") and c.endswith("_final_recon")]
    n_stages = len([c for c in df.columns if "_final_recon" in c])
    stage_names = []
    for c in df.columns:
        if "_final_recon" in c:
            # e.g. "s0_s1_final_recon" -> "s1"
            parts = c.split("_")
            stage_names.append(parts[1])

    lines = []
    lines.append("# HVEBT Training Analysis\n")
    lines.append(f"**Log**: `{log_path}`\n")
    lines.append(f"**Steps**: {n_steps}\n")
    lines.append(f"**Stages**: {stage_names}\n")

    # Overall loss
    lines.append("\n## Loss Convergence\n")
    lines.append(f"| Metric | First | Last | Reduction |")
    lines.append(f"|--------|-------|------|-----------|")
    for col in ["loss_total", "loss_energy"]:
        if col in df.columns:
            first = df[col].iloc[0]
            last = df[col].iloc[-1]
            reduction = (first - last) / first * 100
            lines.append(f"| {col} | {first:.4f} | {last:.4f} | {reduction:.1f}% |")
    if "loss_decoder" in df.columns:
        first = df["loss_decoder"].iloc[0]
        last = df["loss_decoder"].iloc[-1]
        reduction = (first - last) / first * 100
        lines.append(f"| loss_decoder | {first:.4f} | {last:.4f} | {reduction:.1f}% |")

    # Per-stage analysis
    lines.append("\n## Per-Stage Analysis\n")
    for i, name in enumerate(stage_names):
        prefix = f"s{i}_{name}"
        lines.append(f"\n### Stage {i} ({name})\n")

        fr_col = f"{prefix}_final_recon"
        ir_col = f"{prefix}_init_recon"
        bl_col = f"{prefix}_baseline_copy"
        eg_col = f"{prefix}_energy_gap"
        al_col = f"{prefix}_alpha"

        if fr_col in df.columns:
            # MCMC improvement
            init_first = df[ir_col].iloc[0]
            final_first = df[fr_col].iloc[0]
            init_last = df[ir_col].iloc[-1]
            final_last = df[fr_col].iloc[-1]
            mcmc_improvement_first = (init_first - final_first) / init_first * 100
            mcmc_improvement_last = (init_last - final_last) / init_last * 100

            lines.append(f"| Metric | First Step | Last Step |")
            lines.append(f"|--------|-----------|----------|")
            lines.append(f"| init_recon | {init_first:.4f} | {init_last:.4f} |")
            lines.append(f"| final_recon | {final_first:.4f} | {final_last:.4f} |")
            lines.append(f"| MCMC improvement | {mcmc_improvement_first:.1f}% | {mcmc_improvement_last:.1f}% |")

        if bl_col in df.columns:
            bl_first = df[bl_col].iloc[0]
            bl_last = df[bl_col].iloc[-1]
            fr_last = df[fr_col].iloc[-1]
            vs_baseline = "BETTER" if fr_last < bl_last else "WORSE"
            lines.append(f"| baseline_copy | {bl_first:.4f} | {bl_last:.4f} |")
            lines.append(f"| vs baseline | - | **{vs_baseline}** ({fr_last:.4f} vs {bl_last:.4f}) |")

        if eg_col in df.columns:
            eg_first = df[eg_col].iloc[0]
            eg_last = df[eg_col].iloc[-1]
            lines.append(f"| energy_gap | {eg_first:.4e} | {eg_last:.4e} |")

        if al_col in df.columns:
            al_first = df[al_col].iloc[0]
            al_last = df[al_col].iloc[-1]
            lines.append(f"| alpha | {al_first:.2f} | {al_last:.2f} |")

    # Convergence velocity
    lines.append("\n## Convergence Velocity\n")
    lines.append("Loss at key checkpoints:\n")
    lines.append(f"| Step | loss_total | loss_energy |")
    lines.append(f"|------|-----------|-------------|")
    checkpoints = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    for cp in checkpoints:
        if cp < n_steps:
            row = df.iloc[cp]
            lt = row.get("loss_total", "N/A")
            le = row.get("loss_energy", "N/A")
            lines.append(f"| {int(row['step'])} | {lt:.4f} | {le:.4f} |")

    # Timing
    lines.append("\n## Performance\n")
    if "secs" in df.columns:
        avg_time = df["secs"].mean()
        total_time = df["secs"].sum()
        lines.append(f"- Average step time: {avg_time:.2f}s")
        lines.append(f"- Total training time: {total_time:.0f}s ({total_time/60:.1f} min)")
        lines.append(f"- Steps/second: {1/avg_time:.3f}")

    # Key findings
    lines.append("\n## Key Findings\n")
    # Check if model beats copy-last baseline at any stage
    any_beats_baseline = False
    for i, name in enumerate(stage_names):
        fr_col = f"s{i}_{name}_final_recon"
        bl_col = f"s{i}_{name}_baseline_copy"
        if fr_col in df.columns and bl_col in df.columns:
            if df[fr_col].iloc[-1] < df[bl_col].iloc[-1]:
                any_beats_baseline = True
                lines.append(f"- Stage {i} ({name}): model BEATS copy-last baseline")
            else:
                lines.append(f"- Stage {i} ({name}): model does NOT beat copy-last baseline yet")

    # Check MCMC is improving predictions
    for i, name in enumerate(stage_names):
        ir_col = f"s{i}_{name}_init_recon"
        fr_col = f"s{i}_{name}_final_recon"
        if ir_col in df.columns and fr_col in df.columns:
            if df[fr_col].iloc[-1] < df[ir_col].iloc[-1]:
                lines.append(f"- Stage {i} ({name}): MCMC IS refining predictions (final < init)")
            else:
                lines.append(f"- Stage {i} ({name}): MCMC NOT helping (final >= init)")

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Analysis saved to: {output_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to train_log.csv")
    parser.add_argument("--output", type=str, default=None, help="Output markdown path")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.log), "analysis.md")

    analyze(args.log, args.output)


if __name__ == "__main__":
    main()
