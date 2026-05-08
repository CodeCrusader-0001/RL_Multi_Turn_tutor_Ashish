import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "figure.dpi":       130,
})

COLORS = {
    "D":        "#4472C4",
    "D+":       "#ED7D31",
    "D++":      "#9DC3E6",
    "Prompt":   "#70AD47",
    "SuperCQL": "#C00000",
}
PALETTE      = ["#2196F3","#4CAF50","#FF9800","#F44336","#9C27B0","#00BCD4","#FF5722","#607D8B"]
ACTION_NAMES = ["Instruct", "Encourage", "Refocus", "Question"]

def _c(name):
    if name == "Prompt":   return COLORS["Prompt"]
    if name == "SuperCQL": return COLORS["SuperCQL"]
    return COLORS["D+"] if "+" in name else COLORS["D"]

def _save(fig, out, fname):
    path = os.path.join(out, fname)
    fig.savefig(path, bbox_inches="tight")
    print(f"   -> {path}")

def plot_success_rates(res, out, n_eval, order=None):
    order  = order or [k for k in ["BC","BC+","Q","Q+","CQL","CQL+","SuperCQL","Prompt"] if k in res]
    means  = [res[n][0] for n in order]
    cis    = [res[n][1] for n in order]
    cols   = [_c(n)     for n in order]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(order, means, yerr=cis, color=cols, capsize=6, alpha=0.88,
                  edgecolor="black", linewidth=0.8,
                  error_kw={"linewidth": 2, "capthick": 2})
    for b, m, ci in zip(bars, means, cis):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+ci+0.012,
                f"{m:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    if "Prompt" in res:
        ax.axhline(res["Prompt"][0], color=COLORS["Prompt"], linestyle="--", alpha=0.6, linewidth=1.5)
    handles = [
        mpatches.Patch(color=COLORS["D"],        label="Original D"),
        mpatches.Patch(color=COLORS["D+"],       label="Augmented D+"),
        mpatches.Patch(color=COLORS["SuperCQL"], label="SuperCQL (D++ + tricks)"),
        mpatches.Patch(color=COLORS["Prompt"],   label="Prompt Engineering"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_ylabel("Average Success Rate", fontweight="bold")
    ax.set_xlabel("Tutor Policy",         fontweight="bold")
    ax.set_title(f"Figure 3 - All Policies ({n_eval} eval episodes)", fontweight="bold")
    ax.set_ylim(0, min(1.05, max(means)*1.55+0.05))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout(); _save(fig, out, "figure3_success_rates.png")
    plt.close(fig)

def plot_augmentation_impact(res, out):
    pairs = [(b, a) for b, a in [("BC","BC+"),("Q","Q+"),("CQL","CQL+")] if b in res and a in res]
    if not pairs: return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pairs)); w = 0.32
    bm = [res[b][0] for b, a in pairs]; bc_ci = [res[b][1] for b, a in pairs]
    am = [res[a][0] for b, a in pairs]; ac_ci = [res[a][1] for b, a in pairs]
    ax.bar(x - w/2, bm, w, yerr=bc_ci, label="Original D",   color=COLORS["D"],  alpha=0.87,
           edgecolor="black", capsize=5, error_kw={"linewidth":1.5,"capthick":1.5})
    ax.bar(x + w/2, am, w, yerr=ac_ci, label="Augmented D+", color=COLORS["D+"], alpha=0.87,
           edgecolor="black", capsize=5, error_kw={"linewidth":1.5,"capthick":1.5})
    for i, (base, aug) in enumerate(zip(bm, am)):
        pct = (aug - base) / max(base, 0.001) * 100
        col = "darkgreen" if pct >= 0 else "red"
        ax.annotate(f"{pct:+.0f}%",
                    xy=(x[i], max(base, aug) + max(bc_ci[i], ac_ci[i]) + 0.04),
                    ha="center", fontsize=11, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([a for b, a in pairs], fontsize=12)
    ax.set_ylabel("Success Rate", fontweight="bold")
    ax.set_title("Augmentation Impact: D vs D+", fontweight="bold")
    ax.set_ylim(0, min(1.15, max(am)*1.7+0.1))
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout(); _save(fig, out, "augmentation_impact.png")
    plt.close(fig)

def plot_loss_curves(data_dir, out, model_names=None):
    if model_names is None:
        model_names = ["bc","q","cql","bc_aug","q_aug","cql_aug",
                       "super_cql_s42","super_cql_s7","super_cql_s13"]
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = plt.cm.tab10.colors
    found = 0
    for i, name in enumerate(model_names):
        path = os.path.join(data_dir, f"loss_{name}.npy")
        if not os.path.exists(path): continue
        loss = np.load(path)
        if len(loss) > 0:
            ax.plot(range(1, len(loss)+1), loss, label=name,
                    color=palette[i%len(palette)], linewidth=1.8)
            found += 1
    if found == 0:
        print("  [warn] No loss files found - skipping loss curve plot")
        plt.close(fig); return
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves", fontweight="bold")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout(); _save(fig, out, "loss_curves.png")
    plt.close(fig)

def plot_dataset_stats(stats_list, out):
    labels    = [s["label"]     for s in stats_list]
    successes = [s["success"]   for s in stats_list]
    diversity = [s["diversity"] for s in stats_list]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, successes, color=PALETTE[:len(labels)], alpha=0.87, edgecolor="black")
    axes[0].set_title("Success Rate by Dataset"); axes[0].set_ylim(0, 1)
    axes[1].bar(labels, diversity, color=PALETTE[:len(labels)], alpha=0.87, edgecolor="black")
    axes[1].set_title("Action Diversity by Dataset"); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(out, "dataset_stats.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"   -> {path}")

def plot_action_heatmap(trained, S, out, order=None):
    order = order or ["BC","BC+","Q","Q+","CQL","CQL+","SuperCQL"]
    order = [n for n in order if n in trained]
    np.random.seed(42)
    idx = np.random.choice(len(S), size=min(200, len(S)), replace=False)
    S_sample = S[idx]
    action_matrix = []
    for name in order:
        model = trained[name]
        if isinstance(model, list):
            votes = np.stack([m.predict(S_sample) for m in model], axis=0)
            preds = np.array([np.bincount(votes[:,i], minlength=4).argmax() for i in range(len(S_sample))])
        else:
            preds = model.predict(S_sample)
        counts = np.bincount(preds, minlength=4).astype(float)
        action_matrix.append(counts / counts.sum())
    action_matrix = np.array(action_matrix)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(action_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(ACTION_NAMES, fontsize=11)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=11)
    ax.set_title("Action Distribution per Policy (n=200 sampled states)", fontsize=12)
    for i in range(len(order)):
        for j in range(4):
            val = action_matrix[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=10, color=color)
    plt.colorbar(im, ax=ax, label="Fraction of actions")
    fig.tight_layout(); _save(fig, out, "action_distribution.png")
    plt.close(fig)

def plot_offline_eval(trained, S, A, R, T, out, order=None):
    order = order or ["BC","BC+","Q","Q+","CQL","CQL+","SuperCQL"]
    order = [n for n in order if n in trained]
    terminal_idx = np.where(T == 1.0)[0]
    metrics = {}
    for name in order:
        model = trained[name]
        if isinstance(model, list):
            votes = np.stack([m.predict(S) for m in model], axis=1)
            preds = np.array([np.bincount(row, minlength=4).argmax() for row in votes])
        else:
            preds = model.predict(S)
        acc   = np.mean(preds == A)
        avg_r = np.mean(R[preds == A]) if np.any(preds == A) else float("nan")
        mt    = terminal_idx[preds[terminal_idx] == A[terminal_idx]]
        pol_succ = np.mean(R[mt] > 0) if len(mt) else 0.0
        dist  = np.bincount(preds, minlength=4).astype(float); dist /= dist.sum()
        metrics[name] = dict(acc=acc, avg_r=avg_r, success_rate=pol_succ, action_dist=dist)
    print(f"  {'Policy':<12} {'Action Acc':>11} {'Avg Reward':>11} {'Term Match Success':>20}")
    print("  " + "-"*58)
    for name in order:
        m = metrics[name]
        print(f"  {name:<12}: acc={m['acc']:.2%}  avg_r={m['avg_r']:+.3f}  term_success={m['success_rate']:.2%}")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(order))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(order))]
    accs   = [metrics[n]["acc"]   for n in order]
    avg_rs = [metrics[n]["avg_r"] for n in order]
    dists  = np.array([metrics[n]["action_dist"] for n in order])
    ax = axes[0]
    bars = ax.bar(x, accs, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Action Accuracy vs Expert Dataset", fontsize=11)
    ax.axhline(np.mean(accs), color="black", linestyle="--", linewidth=1, label=f"mean={np.mean(accs):.2%}")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    ax = axes[1]
    bars = ax.bar(x, avg_rs, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg Reward"); ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Avg Reward on Matching Steps", fontsize=11)
    for bar, v in zip(bars, avg_rs):
        ax.text(bar.get_x()+bar.get_width()/2, v+(0.01 if v>=0 else -0.04), f"{v:+.2f}",
                ha="center", va="bottom", fontsize=8)
    ax = axes[2]
    im = ax.imshow(dists, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(ACTION_NAMES, fontsize=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
    ax.set_title("Action Distribution per Policy", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(order)):
        for j in range(4):
            ax.text(j, i, f"{dists[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if dists[i,j] > 0.6 else "black")
    fig.suptitle("Offline Dataset Evaluation", fontsize=13, y=1.02)
    fig.tight_layout(); _save(fig, out, "offline_eval.png")
    plt.close(fig)

def plot_all(res, out, n_eval, data_dir, stats_list=None, order=None,
             trained=None, S=None, A=None, R=None, T=None):
    print("\n-- Generating figures --")
    os.makedirs(out, exist_ok=True)
    plot_success_rates(res, out, n_eval, order)
    plot_augmentation_impact(res, out)
    plot_loss_curves(data_dir, out)
    if stats_list:
        plot_dataset_stats(stats_list, out)
    if trained is not None and S is not None:
        plot_action_heatmap(trained, S, out, order)
        if A is not None and R is not None and T is not None:
            plot_offline_eval(trained, S, A, R, T, out, order)
    print("All figures saved.")
