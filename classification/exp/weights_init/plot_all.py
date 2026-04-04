import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Initialization method configs: key -> (display label, color)
# ImageNet color intentionally matches aug4 (#27AE60) from the original script.
# ---------------------------------------------------------------------------
INIT_CONFIGS = {
    "random":  ("Random Init",   "#888888"),   # neutral gray
    "imagenet": ("ImageNet",     "#27AE60"),   # green  ← same as aug4
    "simclr1": ("SimCLR (v1)",   "#4C9BE8"),   # steel blue
    "simclr2": ("SimCLR (v2)",   "#E8A838"),   # amber / golden orange
}

# Legend order (best → worst, top → bottom)
LEGEND_ORDER = ["ImageNet", "SimCLR (v2)", "SimCLR (v1)", "Random Init"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def get_json_paths(base_dir, init_key, seeds, suffix):
    return [
        os.path.join(base_dir, f'cold_start_{init_key}', f'{seed}_{suffix}')
        for seed in seeds
    ]


def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LR selection
# ---------------------------------------------------------------------------

def get_best_lr_representative_acc(rho_data):
    """Pick the LR with the highest mean accuracy."""
    best_mean, best_vals, best_lr = -1, [], None
    for lr, vals in rho_data.items():
        m = np.mean(vals)
        if m > best_mean:
            best_mean, best_vals, best_lr = m, vals, lr
    return np.mean(best_vals), best_lr, best_vals


def get_fixed_lr_representative_acc(rho_data, only_lr):
    """Use a specific LR (float-compared so '1e-4' == '0.0001')."""
    target = float(only_lr)
    matched_key = None
    for k in rho_data:
        try:
            if float(k) == target:
                matched_key = k
                break
        except ValueError:
            continue
    if matched_key is None:
        return None, only_lr, []
    vals = rho_data[matched_key]
    return np.mean(vals), matched_key, vals


def get_representative_acc(rho_data, only_lr):
    if only_lr is not None:
        return get_fixed_lr_representative_acc(rho_data, only_lr)
    return get_best_lr_representative_acc(rho_data)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_all_rhos(init_data_dict, aug_key):
    """Sorted union of all rho values across all init types and seeds."""
    rhos = set()
    for data_list in init_data_dict.values():
        for data in data_list:
            if aug_key in data:
                for r in data[aug_key]:
                    rhos.add(float(r))
    return sorted(rhos)


# ---------------------------------------------------------------------------
# Terminal stats
# ---------------------------------------------------------------------------

def print_stats(init_data_dict, seeds, aug_key,
                filter_rhos=None, only_lr=None, presented_keys=None):
    all_rhos = get_all_rhos(init_data_dict, aug_key)
    if filter_rhos is not None:
        all_rhos = [r for r in all_rhos if r in filter_rhos]

    lr_mode_str = f"only_lr={only_lr}" if only_lr is not None else "best LR per run"

    print(f"\n{'='*90}")
    print(f"  Aug key: {aug_key}  |  LR mode: {lr_mode_str}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*90}")

    for rho in all_rhos:
        rho_str = str(float(rho))
        rho_display = int(rho) if rho == int(rho) else rho

        print(f"\n{'='*90}")
        print(f"  rho = {rho_display}%")
        print(f"{'='*90}")

        for init_key, (label, _) in INIT_CONFIGS.items():
            if presented_keys is not None and init_key not in presented_keys:
                continue
            if init_key not in init_data_dict:
                continue

            data_list = init_data_dict[init_key]
            per_seed_results = []
            for data in data_list:
                if aug_key not in data or rho_str not in data[aug_key]:
                    per_seed_results.append(None)
                    continue
                mean, best_lr, best_vals = get_representative_acc(
                    data[aug_key][rho_str], only_lr)
                if mean is None:
                    per_seed_results.append(None)
                    continue
                per_seed_results.append({
                    'mean': mean,
                    'std': np.std(best_vals),
                    'best_lr': best_lr,
                    'best_vals': best_vals,
                })

            valid = [r for r in per_seed_results if r is not None]
            if not valid:
                continue

            print(f"\n  [{label}]")
            print(f"  {'-'*96}")
            print(f"  {'Seed':<8} {'LR':<10} {'Mean':>8} {'Std':>8}  {'Values'}")
            print(f"  {'-'*96}")

            for i, result in enumerate(per_seed_results):
                seed_name = seeds[i] if i < len(seeds) else str(i)
                if result is None:
                    print(f"  {seed_name:<12} {'N/A':<10}")
                else:
                    vals_str = '[' + ', '.join(f'{v:.4f}' for v in result['best_vals']) + ']'
                    print(f"  {seed_name:<12} {result['best_lr']:<10} "
                          f"{result['mean']:>8.4f} {result['std']:>8.4f}  {vals_str}")

            representative_accs = [r['mean'] for r in valid]
            cross_mean = np.mean(representative_accs)
            cross_std  = np.std(representative_accs)
            accs_str   = '[' + ', '.join(f'{v:.4f}' for v in representative_accs) + ']'
            print(f"  {'-'*96}")
            print(f"  {'Cross-Seed':<18} {'Mean':>8} {'Std':>8}  Representative Accs")
            print(f"  {'':<18} {cross_mean:>8.4f} {cross_std:>8.4f}  {accs_str}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(init_data_dict, aug_key,
         save_path=None, only_lr=None,
         presented_keys=None, plot_rhos=None, plot_xticks=None):

    all_rhos = get_all_rhos(init_data_dict, aug_key)
    if plot_rhos is not None:
        plot_rhos_set = set(plot_rhos)
        all_rhos = [r for r in all_rhos if r in plot_rhos_set]

    fig, ax = plt.subplots(figsize=(8, 5))

    for init_key, (label, color) in INIT_CONFIGS.items():
        if presented_keys is not None and init_key not in presented_keys:
            continue
        if init_key not in init_data_dict:
            continue

        data_list = init_data_dict[init_key]
        rhos, means, stds = [], [], []

        for rho in all_rhos:
            rho_str = str(float(rho))
            representative_accs = []

            for data in data_list:
                if aug_key not in data or rho_str not in data[aug_key]:
                    continue
                mean, _, _ = get_representative_acc(data[aug_key][rho_str], only_lr)
                if mean is None:
                    continue
                representative_accs.append(mean)

            if not representative_accs:
                continue

            rhos.append(rho / 100.0)
            means.append(np.mean(representative_accs))
            stds.append(np.std(representative_accs))

        if not rhos:
            continue

        rhos  = np.array(rhos)
        means = np.array(means)
        stds  = np.array(stds)

        ax.plot(rhos, means * 100, marker='o', markersize=5, linewidth=2,
                label=label, color=color, linestyle='-')
        ax.fill_between(rhos,
                        (means - stds) * 100,
                        (means + stds) * 100,
                        alpha=0.15, color=color)

    ax.set_xlabel(r'Labeled Training Data Ratio $\rho$ (%)', fontsize=15, labelpad=8)
    ax.set_ylabel('Accuracy (%)', fontsize=15, labelpad=8)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x*100:.4g}')
    )

    xticks = plot_xticks if plot_xticks is not None else all_rhos
    ax.set_xticks([r / 100.0 for r in xticks])
    ax.set_ylim(38.5, 94.5)

    # Reorder legend
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in LEGEND_ORDER if l in labels]

    lr_title = f" (LR={only_lr})" if only_lr is not None else ""
    ax.legend(
        [handles[i] for i in order],
        [labels[i]  for i in order],
        fontsize=9, loc='lower right',
        title=f"LR mode: fixed{lr_title}" if only_lr else None,
    )

    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare weight-initialization strategies across data portions.')

    parser.add_argument('--base_dir', type=str,
                        default='../../exp_results/classification_hard',
                        help='Base directory containing cold_start_<init_type>/ subdirectories.')

    parser.add_argument('--seeds', type=str, nargs='+',
                        default=['random10', 'random24', 'random38',
                                 'random42', 'random57'],
                        help='Seed prefixes used in JSON filenames.')

    parser.add_argument('--suffix', type=str, default='bs16_ep20.json',
                        help='Filename suffix that follows each seed prefix '
                             '(default: bs16_ep20.json).')

    parser.add_argument('--aug_key', type=str, default='aug4',
                        help='Aug-config key to read from each JSON '
                             '(default: aug4). E.g. aug4, no_aug, aug3 …')

    parser.add_argument('--init_types', type=str, nargs='+',
                        default=['random', 'imagenet', 'simclr1', 'simclr2'],
                        help='Init types to plot.  Each maps to a '
                             'cold_start_<init_type>/ subfolder. '
                             'Choices: random imagenet simclr1 simclr2')

    parser.add_argument('--portions', type=float, nargs='*', default=None,
                        help='Only print terminal stats for these rho values '
                             '(e.g. --portions 2.5 10 20). Default: all.')

    parser.add_argument('--save', type=str, default='./init_comparison.png',
                        help='Output path for the saved figure. '
                             'If omitted the figure is shown interactively.')

    parser.add_argument('--only_lr', type=str, default=None,
                        help='Fix a single LR for all runs instead of '
                             'picking the best LR per run.')

    parser.add_argument('--plot_rhos', type=float, nargs='+',
                        default=[2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='Rho values to include in the plot.')

    parser.add_argument('--plot_xticks', type=float, nargs='+',
                        default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='Rho values shown as x-axis tick labels.')

    args = parser.parse_args()

    if args.plot_xticks is None:
        args.plot_xticks = args.plot_rhos

    # ------------------------------------------------------------------
    # Load JSON files for each requested init type
    # ------------------------------------------------------------------
    init_data_dict = {}
    for init_key in args.init_types:
        if init_key not in INIT_CONFIGS:
            print(f"Warning: '{init_key}' is not defined in INIT_CONFIGS — skipping.")
            continue
        paths = get_json_paths(args.base_dir, init_key, args.seeds, args.suffix)
        data_list = []
        for p in paths:
            try:
                data_list.append(load_data(p))
                print(f"  Loaded: {p}")
            except FileNotFoundError:
                print(f"  Warning – file not found, skipping: {p}")
        if data_list:
            init_data_dict[init_key] = data_list
        else:
            print(f"  No files loaded for init_type='{init_key}', skipping.")

    print(f"\nLoaded init types : {list(init_data_dict.keys())}")
    print(f"Aug key           : {args.aug_key}")
    print(f"LR mode           : "
          f"{'fixed LR = ' + args.only_lr if args.only_lr else 'best LR per run (default)'}")

    print_stats(init_data_dict, args.seeds, args.aug_key,
                filter_rhos=args.portions,
                only_lr=args.only_lr,
                presented_keys=args.init_types)

    plot(init_data_dict, args.aug_key,
         save_path=args.save,
         only_lr=args.only_lr,
         presented_keys=args.init_types,
         plot_rhos=args.plot_rhos,
         plot_xticks=args.plot_xticks)


if __name__ == '__main__':
    main()