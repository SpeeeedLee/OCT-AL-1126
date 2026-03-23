import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_best_lr_stats(rho_data):
    """Find LR with highest mean, return (mean, std, best_lr, vals) of that LR's runs."""
    best_mean = -1
    best_vals = []
    best_lr = None
    for lr, vals in rho_data.items():
        m = np.mean(vals)
        if m > best_mean:
            best_mean = m
            best_vals = vals
            best_lr = lr
    return np.mean(best_vals), np.std(best_vals), best_lr, best_vals


AUG_CONFIGS = {
    "no_aug":          ("No Aug",            "#888888"),
    "aug2_horizontal": ("Horizontal (2×)",   "#4C9BE8"),
    "aug2_vertical":   ("Vertical (2×)",     "#E87C4C"),
    "aug3":            ("H + V (3×)",        "#9B59B6"),
    "aug4":            ("H + V + HV (4×)",   "#27AE60"),
}


def print_stats(data):
    all_rhos = sorted(set(
        float(r)
        for cfg in AUG_CONFIGS
        if cfg in data
        for r in data[cfg]
    ))

    for rho in all_rhos:
        rho_str = str(float(rho))
        print(f"\n{'='*75}")
        print(f"  rho = {int(rho) if rho == int(rho) else rho}%")
        print(f"{'='*75}")
        print(f"  {'Strategy':<22} {'Best LR':<10} {'Mean':>8} {'Std':>8}  Values")
        print(f"  {'-'*73}")
        for cfg_key, (label, _) in AUG_CONFIGS.items():
            if cfg_key not in data:
                continue
            if rho_str not in data[cfg_key]:
                continue
            mean, std, best_lr, best_vals = get_best_lr_stats(data[cfg_key][rho_str])
            vals_str = '[' + ', '.join(f'{v:.4f}' for v in best_vals) + ']'
            print(f"  {label:<22} {best_lr:<10} {mean:>8.4f} {std:>8.4f}  {vals_str}")


def plot(data, save_path=None):
    all_rhos = sorted(set(
        float(r)
        for cfg in AUG_CONFIGS
        if cfg in data
        for r in data[cfg]
    ))

    fig, ax = plt.subplots(figsize=(8, 5))

    for cfg_key, (label, color) in AUG_CONFIGS.items():
        if cfg_key not in data:
            continue

        rhos, means, stds = [], [], []
        for rho in all_rhos:
            rho_str = str(float(rho))
            if rho_str not in data[cfg_key]:
                continue
            mean, std, _, _ = get_best_lr_stats(data[cfg_key][rho_str])
            rhos.append(rho / 100.0)
            means.append(mean)
            stds.append(std)

        rhos  = np.array(rhos)
        means = np.array(means)
        stds  = np.array(stds)

        ax.plot(rhos, means, marker='o', markersize=5, linewidth=2,
                label=label, color=color)
        ax.fill_between(rhos, means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.set_xlabel(r'$\rho$ (Labeled Training Data Ratio; %)', fontsize=15, labelpad=8)
    ax.set_ylabel('Accuracy', fontsize=15, labelpad=8)
    # tick 字體大小
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(round(x*100))}')
    )
    # ax.set_xticks(sorted(set(r / 100.0 for r in all_rhos)))
    ax.set_xticks([r / 100.0 for r in range(10, 101, 10)])
    ax.tick_params(axis='x', 
                #    rotation=45
                )
    ax.set_ylim(0.525, 0.95)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str,
                        default='../../exp_results/classification_hard/cold_start_imagenet/random42_bs16.json',
                        help='Path to the results JSON file')
    parser.add_argument('--save', type=str, default='./imagenet_aug.png',
                        help='Path to save the plot. If not set, show interactively.')
    args = parser.parse_args()

    data = load_data(args.json_path)
    print_stats(data)
    plot(data, save_path=args.save)


if __name__ == '__main__':
    main()