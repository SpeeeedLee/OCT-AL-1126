"""
SimCLR bs=256 & bs=16 fine-tuning accuracy vs. pretraining epoch (line + shaded std).

For each (bs, ep, seed): picks best (simclr_lr, ft_lr) combination by mean accuracy.
Single seed (ρ=100) → std of the best-lr runs.

Usage (from repo root):
    python3 thesis/chapter_4/plot_simclr_bs256_line.py
"""
import os, re, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASES   = [
    "classification/exp_results/classification_hard/cold_start_simclr",
    "exp_results/classification_hard/cold_start_simclr",
]
OUT     = "thesis/chapter_4/figs/simclr_bs256_epoch_line.png"
AUG_KEY = "aug4"
RHO_KEY = "100.0"
EXCLUDE_EPS = {10}   # ep=10 excluded from plot

# viridis(linspace(0.12, 0.9, 5)): index 0=bs16, index 4=bs256
_PALETTE = cm.viridis(np.linspace(0.12, 0.9, 5))
BS_CONFIG = {
    16:  {"color": _PALETTE[0], "label": "16"},
    256: {"color": _PALETTE[4], "label": "256"},
}


def make_pattern(bs):
    return re.compile(
        rf"random(\d+)_bs16_ep20_simclr_lr([\d.e-]+)_simclr_bs{bs}_simclr_ep"
        r"(\d+(?:_wval)?(?:_best)?)\.json"
    )


def load_data(bs):
    PAT = make_pattern(bs)
    raw = {}  # ep_tag -> seed -> (best_mean, best_runs)
    seen = set()
    for base in BASES:
        if not os.path.isdir(base):
            continue
        for fname in os.listdir(base):
            if fname in seen:
                continue
            seen.add(fname)
            m = PAT.match(fname)
            if not m:
                continue
            seed, slr, ep_tag = int(m.group(1)), m.group(2), m.group(3)
            if "best" in ep_tag:
                continue
            d = json.load(open(os.path.join(base, fname)))
            lrd = d.get(AUG_KEY, {}).get(RHO_KEY, {})
            if not lrd:
                continue
            best_mean, best_runs = -1, []
            for ft_lr, runs in lrd.items():
                m_acc = float(np.mean(runs))
                if m_acc > best_mean:
                    best_mean, best_runs = m_acc, list(runs)
            prev = raw.setdefault(ep_tag, {}).get(seed)
            if prev is None or best_mean > prev[0]:
                raw[ep_tag][seed] = (best_mean, best_runs)

    out = {}
    for ep_tag, seeds in raw.items():
        seed_means = np.array([v[0] for v in seeds.values()])
        if len(seed_means) > 1:
            mean = seed_means.mean() * 100
            std  = seed_means.std(ddof=1) * 100
        else:
            runs = list(seeds.values())[0][1]
            arr  = np.array(runs, dtype=float)
            mean = arr.mean() * 100
            std  = arr.std(ddof=1) * 100 if len(arr) > 1 else 0.0
        out[ep_tag] = (mean, std)
    return out


def ep_sort_key(tag):
    base = tag.replace("_wval", "").replace("_best", "")
    return (int(base), 0 if "_wval" not in tag else 1)


def ep_to_x(tag):
    return int(tag.replace("_wval", "").replace("_best", ""))


def main():
    FONT_LABEL  = 24
    FONT_TICK   = 18
    FONT_LEGEND = 20
    FONT_ANNOT  = 17

    fig, ax = plt.subplots(figsize=(13.5, 6.5))

    all_xs = set()
    series = {}
    for bs in [16, 256]:
        data = load_data(bs)
        tags = sorted([t for t in data if ep_to_x(t) not in EXCLUDE_EPS], key=ep_sort_key)
        xs    = np.array([ep_to_x(t) for t in tags])
        means = np.array([data[t][0] for t in tags])
        stds  = np.array([data[t][1] for t in tags])
        series[bs] = (tags, xs, means, stds)
        all_xs.update(xs.tolist())

    # unified x-ticks (union of both series)
    all_xs = sorted(all_xs)

    for bs in [16, 256]:
        cfg = BS_CONFIG[bs]
        tags, xs, means, stds = series[bs]
        c = cfg["color"]
        ax.fill_between(xs, means - stds, means + stds,
                        color=c, alpha=0.22, linewidth=0)
        ax.plot(xs, means, color=c, linewidth=2.5, marker="o",
                markersize=8, markerfacecolor=c,
                markeredgecolor="white", markeredgewidth=1.2,
                label=cfg["label"])


    ax.set_xscale("log")
    ax.set_xticks(all_xs)
    ax.set_xticklabels([str(x) for x in all_xs])
    ax.set_xlim(min(all_xs) * 0.7, 2200)
    ax.set_xlabel("SimCLR Pretraining Epoch", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.set_ylim(90.5, 97)

    legend = ax.legend(fontsize=FONT_LEGEND, framealpha=0.9,
                       loc="upper left", title="Pretraining Batch Size",
                       title_fontsize=FONT_LEGEND, ncol=2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")
    for bs in [16, 256]:
        tags, xs, means, stds = series[bs]
        print(f"\n  bs={bs}:")
        for t, m, s in zip(tags, means, stds):
            print(f"    ep={t:>12}: {m:.2f} ± {s:.2f}%")


if __name__ == "__main__":
    main()
