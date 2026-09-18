"""Render descriptive fit/cal curves from completed, checked frozen artifacts."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/router/hotpotqa_bd_router_v1/runs/m6_output_adaptation_v1"
DEST = ROOT / "analysis/hotpotqa_router/figures"
STEM = "m6_output_adaptation_fit_cal_20260917"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(name):
    return hashlib.sha256((OUT / name).read_bytes()).hexdigest()


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def main():
    r, check, recovery = [read(n) for n in (
        "results.json", "separate_checks.json", "recovery_separate_checks_20260917.json")]
    assert check["status"] == "passed_independent_M6_output_adaptation_formal_checks"
    assert recovery["status"] == "passed_independent_M6_output_adaptation_recovery_provenance_checks"
    for receipt in (check, recovery):
        assert receipt["results_sha256"] == sha("results.json")
        assert receipt["predictions_sha256"] == sha("predictions.npz")
        assert receipt["protocol_sha256"] == sha("protocol.json")
    assert recovery["formal_separate_checks_sha256"] == sha("separate_checks.json")
    for key in r["primary"]:
        for metric in ("mean", "interval"):
            close(r["primary"][key][metric], check["effects"]["primary"][key][metric])
    with np.load(OUT / "predictions.npz", allow_pickle=False) as z:
        p = {k: z[k] for k in z.files}
    # Utility archive order is BM25, Dense; no new test inference is performed.
    gap = p["utility"][:, 0] - p["utility"][:, 1]
    assert p["utility"].shape == (9600, 2)
    weight = np.where(np.abs(gap) > 1e-12, np.abs(gap), 0.0)
    label = (gap > 0).astype(float)
    values = {k: np.where(p[k], p["utility"][:, 0], p["utility"][:, 1])
              for k in ("A", "C", "B", "Dense", "BM25")}
    for f in [None] + r["folds"]:
        idx = np.arange(9600) if f is None else np.flatnonzero(p["fold_id"] == f["fold"])
        policies = r["policy"] if f is None else f["policy"]
        for k, summary in policies.items():
            action, g = p[k][idx], gap[idx]
            close(values[k][idx].mean(), summary["F1"])
            assert int(action.sum()) == summary["bm25_count"]
            assert int((action & (g > 1e-12)).sum()) == summary["beneficial_count"]
            assert int((action & (g < -1e-12)).sum()) == summary["harmful_count"]
            assert int((action & (np.abs(g) <= 1e-12)).sum()) == summary["zero_count"]
            close(np.where(action, np.maximum(g, 0), 0).mean(), summary["beneficial_mass"])
            close(np.where(action, np.maximum(-g, 0), 0).mean(), summary["harmful_mass"])
        for k in r["primary"]:
            a, b = k.split("_minus_")
            expected = r["primary"][k]["mean"] if f is None else f["primary_means"][k]
            close((values[a][idx] - values[b][idx]).mean(), expected)

    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6B6B00"]
    for row, arm in enumerate(("A", "C")):
        for f in r["folds"]:
            fold, record = f["fold"], f["arms"][arm]
            candidates = record["candidates"]
            assert [c["epoch"] for c in candidates] == list(range(5))
            fit = np.array([c["fit_BCE"] for c in candidates])
            cal = np.array([c["cal_gain"] for c in candidates])
            with np.load(OUT / f"fold{fold}_{arm}_fit_cal.npz", allow_pickle=False) as z:
                for part in ("fit", "cal"):
                    idx, score = z[f"{part}_indices"], z[f"{part}_logits"].astype(float)
                    bce = np.sum(weight[idx] * (np.logaddexp(0, score) - label[idx] * score), axis=1) / weight[idx].sum()
                    gain = np.mean(gap[idx] * (score > 0), axis=1)
                    close(bce, [c[f"{part}_BCE"] for c in candidates])
                    close(gain, [c[f"{part}_gain"] for c in candidates])
            selected = int(np.flatnonzero(cal >= cal.max() - 1e-12)[0])
            assert selected == r["selected_epochs"][arm][fold]
            for col, data in enumerate((fit, cal - cal[0])):
                ax = axes[row, col]
                ax.plot(range(5), data, color=colors[fold], marker=".", lw=1.6)
                ax.scatter([selected], [data[selected]], s=85, facecolors="none",
                           edgecolors=colors[fold], linewidths=2.1, zorder=5)
            print(arm, fold, "selected", selected, "fit0/fit4", fit[[0, 4]].tolist(),
                  "selected_cal_delta", float(cal[selected] - cal[0]))
        arm_name = "A: encoder + head" if arm == "A" else "C: head only"
        axes[row, 0].set_title(f"{arm_name} | fit weighted BCE", loc="left")
        axes[row, 1].set_title(f"{arm_name} | cal gain from epoch 0", loc="left")
        axes[row, 0].set_ylabel("Weighted BCE (partition normalization)")
        axes[row, 1].set_ylabel("Answer F1 difference from own epoch 0")
        axes[row, 1].axhline(0, color="#555555", lw=.8, linestyle="--", zorder=0)
        axes[row, 1].set_ylim(-.0108, .0082)
        axes[row, 0].text(.02, .04, "Selected epochs (folds 0-4): " +
                          ", ".join(map(str, r["selected_epochs"][arm])),
                          transform=axes[row, 0].transAxes, fontsize=9)
    axes[0, 0].set_ylim(.145, .65)
    axes[1, 0].set_ylim(.613, .630)
    for ax in axes.ravel():
        ax.set_xticks(range(5))
        ax.grid(alpha=.18)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in axes[1]:
        ax.set_xlabel("Epoch (0 = original M6)")
    legend = [Line2D([0], [0], color=colors[f], lw=2, label=f"Fold {f}") for f in range(5)]
    legend.append(Line2D([0], [0], color="#444444", marker="o", mfc="none", ls="none",
                         markersize=8, label="Cal-selected endpoint"))
    fig.suptitle("M6 output adaptation: fit improves, cal utility does not improve steadily", y=.98, fontsize=15)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .935), ncol=6, frameon=False)
    fig.text(.5, .018, "Descriptive curves only. Fit panels use different y-scales; cal panels share a scale.\n"
             "Each fold is shown separately, without confidence intervals. No unselected endpoint test scores are evaluated.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(top=.83, bottom=.13, left=.085, right=.98, hspace=.36, wspace=.26)
    DEST.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST / (STEM + ".png"), dpi=180, metadata={"Description": "results_sha256=" + sha("results.json")})
    fig.savefig(DEST / (STEM + ".svg"), metadata={"Description": "results_sha256=" + sha("results.json"), "Date": "2026-09-17"})
    plt.close(fig)
    print("Checked all 30 policy summaries, 24 primary means, 50 candidate fit/cal metrics and 10 selections.")
    for suffix in (".png", ".svg"):
        path = DEST / (STEM + suffix)
        print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
