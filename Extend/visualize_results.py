"""
visualize_results.py - Generate comparison figures for the report/presentation.

Creates:
  1. Qualitative comparison: slices showing original, GT, Exp A/B/C predictions
  2. Quantitative bar charts: Dice/clDice across experiments
  3. Training curves: loss over epochs for Exp B and C
  4. Threshold analysis: Dice vs threshold curves

Usage:
    python visualize_results.py --results_dir ./results --output_dir ./figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional


# ============================================================================
# COLOR SCHEME
# ============================================================================

COLORS = {
    'exp_a': '#E24B4A',   # Red - direct inference (worst expected)
    'exp_b': '#1D9E75',   # Teal - fine-tuned (best expected)
    'exp_c': '#378ADD',   # Blue - from scratch
    'gt':    '#EF9F27',   # Amber - ground truth overlay
}

EXP_LABELS = {
    'exp_a': 'Direct Inference\n(no fine-tune)',
    'exp_b': 'Fine-tuned\n(MiniVess → VesSAP)',
    'exp_c': 'From Scratch\n(VesSAP only)',
}


# ============================================================================
# 1. QUALITATIVE COMPARISON (SLICE VIEW)
# ============================================================================

def plot_qualitative_comparison(
    results_dir: str,
    processed_dir: str,
    strategy: str,
    fold: int,
    output_dir: str,
    slice_indices: Optional[List[int]] = None,
):
    """
    Generate side-by-side comparison of segmentation results.
    Shows: Input | Ground Truth | Exp A | Exp B | Exp C
    for selected axial slices.
    """
    # Load volume and label
    sample_dir = os.path.join(processed_dir, strategy)
    split_path = os.path.join(sample_dir, f"fold_{fold}.json")
    with open(split_path) as f:
        split = json.load(f)
    test_sample = split['test'][0]

    volume = np.load(os.path.join(sample_dir, test_sample, "volume.npy"))
    label = np.load(os.path.join(sample_dir, test_sample, "label.npy"))

    # Load predictions
    preds = {}
    for exp in ['exp_a', 'exp_b', 'exp_c']:
        pred_path = os.path.join(results_dir, exp, strategy, f"fold_{fold}", "pred_binary.npy")
        if os.path.exists(pred_path):
            preds[exp] = np.load(pred_path)

    if not preds:
        print(f"  No predictions found for {strategy}/fold_{fold}")
        return

    # Select slices
    D = volume.shape[0]
    if slice_indices is None:
        # Pick 3 evenly spaced slices
        slice_indices = [D // 4, D // 2, 3 * D // 4]

    n_slices = len(slice_indices)
    n_cols = 2 + len(preds)  # input + GT + predictions

    fig, axes = plt.subplots(n_slices, n_cols, figsize=(3.5 * n_cols, 3.5 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Input', 'Ground Truth'] + [EXP_LABELS[k].replace('\n', ' ') for k in sorted(preds.keys())]

    for i, sl in enumerate(slice_indices):
        # Input image
        axes[i, 0].imshow(volume[sl], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_ylabel(f'Slice {sl}', fontsize=11, fontweight='bold')

        # Ground truth overlay
        axes[i, 1].imshow(volume[sl], cmap='gray', vmin=0, vmax=1)
        gt_mask = np.ma.masked_where(label[sl] == 0, label[sl])
        axes[i, 1].imshow(gt_mask, cmap='autumn', alpha=0.5, vmin=0, vmax=1)

        # Predictions
        for j, exp in enumerate(sorted(preds.keys())):
            col = 2 + j
            axes[i, col].imshow(volume[sl], cmap='gray', vmin=0, vmax=1)
            pred_mask = np.ma.masked_where(preds[exp][sl] == 0, preds[exp][sl])
            color = COLORS[exp]
            cmap = matplotlib.colors.ListedColormap([color])
            axes[i, col].imshow(pred_mask, cmap=cmap, alpha=0.5, vmin=0, vmax=1)

        # Titles (first row only)
        if i == 0:
            for j, title in enumerate(col_titles):
                axes[0, j].set_title(title, fontsize=10, fontweight='bold')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(f'Qualitative Comparison — {strategy} / fold {fold} (test: {test_sample})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"qualitative_{strategy}_fold{fold}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# 2. QUANTITATIVE BAR CHART
# ============================================================================

def plot_metrics_comparison(
    all_results: Dict,
    output_dir: str,
    metrics_to_plot: List[str] = ['dice', 'precision', 'recall', 'cldice'],
):
    """
    Bar chart comparing Dice/Precision/Recall/clDice across experiments.
    Groups by experiment, with separate bars per strategy/fold.
    """
    # Aggregate results by experiment
    exp_data = {}  # exp_name -> {metric: [values across folds/strategies]}
    for key, results in all_results.items():
        exp_name = key.split('/')[0]  # exp_a, exp_b, exp_c
        if exp_name not in exp_data:
            exp_data[exp_name] = {m: [] for m in metrics_to_plot}
        for m in metrics_to_plot:
            val = results.get(m, float('nan'))
            if isinstance(val, (int, float)) and not np.isnan(val):
                exp_data[exp_name][m].append(val)

    # Compute means and stds
    exps = sorted(exp_data.keys())
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(exps))
    width = 0.6

    for ax_idx, metric in enumerate(metrics_to_plot):
        means = []
        stds = []
        colors = []
        for exp in exps:
            vals = exp_data[exp].get(metric, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)
            colors.append(COLORS.get(exp, '#888'))

        bars = axes[ax_idx].bar(x, means, width, yerr=stds, capsize=5,
                                color=colors, edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for bar, mean in zip(bars, means):
            if mean > 0:
                axes[ax_idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                                  f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([EXP_LABELS.get(e, e) for e in exps], fontsize=8)
        axes[ax_idx].set_title(metric.upper(), fontsize=12, fontweight='bold')
        axes[ax_idx].set_ylim(0, 1.1)
        axes[ax_idx].spines['top'].set_visible(False)
        axes[ax_idx].spines['right'].set_visible(False)
        axes[ax_idx].grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Domain Fine-Tuning: Quantitative Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# 3. PER-STRATEGY COMPARISON
# ============================================================================

def plot_strategy_comparison(all_results: Dict, output_dir: str):
    """
    Compare channel fusion strategies (ch0_only vs max_fusion) side by side.
    Shows which fusion strategy works best for each experiment.
    """
    # Parse results
    data = {}  # {(exp, strategy): [dice values]}
    for key, results in all_results.items():
        parts = key.split('/')
        exp = parts[0]
        strategy = parts[1]
        if (exp, strategy) not in data:
            data[(exp, strategy)] = []
        data[(exp, strategy)].append(results.get('dice', 0))

    if not data:
        return

    strategies = sorted(set(s for _, s in data.keys()))
    exps = sorted(set(e for e, _ in data.keys()))

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(exps))
    n_strats = len(strategies)
    width = 0.35
    offsets = np.linspace(-width * (n_strats - 1) / 2, width * (n_strats - 1) / 2, n_strats)

    strat_colors = {'ch0_only': '#534AB7', 'max_fusion': '#1D9E75', 'ch1_only': '#D85A30', 'mean_fusion': '#378ADD'}

    for i, strategy in enumerate(strategies):
        means = []
        stds = []
        for exp in exps:
            vals = data.get((exp, strategy), [])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        bars = ax.bar(x + offsets[i], means, width * 0.9, yerr=stds, capsize=3,
                      label=strategy.replace('_', ' '),
                      color=strat_colors.get(strategy, '#888'),
                      edgecolor='white', linewidth=0.5)

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([EXP_LABELS.get(e, e) for e in exps], fontsize=9)
    ax.set_ylabel('Dice Score', fontsize=11)
    ax.set_title('Channel Fusion Strategy Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "strategy_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# 4. TRAINING CURVES
# ============================================================================

def plot_training_curves(all_results: Dict, output_dir: str):
    """Plot training loss curves for Exp B and C."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for exp, ax, title in [('exp_b', axes[0], 'Fine-Tune (Exp B)'),
                           ('exp_c', axes[1], 'From Scratch (Exp C)')]:
        plotted = False
        for key, results in all_results.items():
            if not key.startswith(exp):
                continue
            history = results.get('training_history', {})
            losses = history.get('train_loss', [])
            if losses:
                label = '/'.join(key.split('/')[1:])  # strategy/fold
                ax.plot(losses, label=label, linewidth=1.5)
                plotted = True

        if plotted:
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Training Loss', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# 5. RESULTS TABLE (LaTeX-ready)
# ============================================================================

def generate_latex_table(all_results: Dict, output_dir: str):
    """
    Generate a LaTeX-ready results table for the report.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-domain fine-tuning results on VesSAP data (mean $\pm$ std over 2 folds).}",
        r"\label{tab:vessap_results}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Experiment & Strategy & Dice & Precision & Recall & clDice \\",
        r"\midrule",
    ]

    # Aggregate by (exp, strategy)
    data = {}
    for key, results in all_results.items():
        parts = key.split('/')
        exp = parts[0]
        strategy = parts[1]
        k = (exp, strategy)
        if k not in data:
            data[k] = []
        data[k].append(results)

    exp_names = {'exp_a': 'Direct Inference', 'exp_b': 'Fine-tuned', 'exp_c': 'From Scratch'}

    for (exp, strat) in sorted(data.keys()):
        results_list = data[(exp, strat)]
        metrics = {}
        for m in ['dice', 'precision', 'recall', 'cldice']:
            vals = [r.get(m, float('nan')) for r in results_list]
            vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
            if vals:
                metrics[m] = f"{np.mean(vals):.3f} $\\pm$ {np.std(vals):.3f}"
            else:
                metrics[m] = "---"

        strat_clean = strat.replace('_', r'\_')
        line = f"{exp_names.get(exp, exp)} & {strat_clean} & "
        line += " & ".join(metrics.get(m, '---') for m in ['dice', 'precision', 'recall', 'cldice'])
        line += r" \\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_str = '\n'.join(lines)
    save_path = os.path.join(output_dir, "results_table.tex")
    with open(save_path, 'w') as f:
        f.write(table_str)
    print(f"  Saved: {save_path}")
    print("\n  LaTeX table preview:")
    print(table_str)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--processed_dir", type=str, default="./data/vessap_processed")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    results_path = os.path.join(args.results_dir, "all_results.json")
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Run run_experiments.py first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    print("Generating figures...")

    # 1. Quantitative comparison
    plot_metrics_comparison(all_results, args.output_dir)

    # 2. Strategy comparison
    plot_strategy_comparison(all_results, args.output_dir)

    # 3. Training curves
    plot_training_curves(all_results, args.output_dir)

    # 4. Qualitative comparison (for each strategy/fold)
    for strategy in ['ch0_only', 'max_fusion']:
        for fold in [1, 2]:
            try:
                plot_qualitative_comparison(
                    args.results_dir, args.processed_dir,
                    strategy, fold, args.output_dir,
                )
            except Exception as e:
                print(f"  Skipping qualitative for {strategy}/fold_{fold}: {e}")

    # 5. LaTeX table
    generate_latex_table(all_results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
