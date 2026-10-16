"""
sweep_psyset.py

Run the 3-agent system from toy_emotion_agents.py under EVERY PsySET condition
and measure which ones discriminate broken vs solvable problems, versus ones that
just flag everything (or nothing).

Conditions (20):
    emotion:<e>   for the 9 emotions in emotions_fewshot_you.csv
                  (joy, sadness, anger, fear, surprise, pride, guilt, disgust, neutral)
    <trait>+/-    for the 5 Big Five traits in attributes.json (pos / neg shots)
    no_prompt     no few-shot block at all (baseline)

Per condition (signal-detection view, "signal" = problem is broken):
    hit rate H          = P(report_problem | broken)     -> sensitivity to broken problems
    false-alarm rate F  = P(report_problem | solvable)   -> false flags
    d' = z(H) - z(F)    -> discrimination (0 = can't tell them apart)
    c  = -(z(H)+z(F))/2 -> bias (negative = flags everything, positive = flags nothing)
    (log-linear correction so 0% / 100% rates give finite z-scores)

Outputs:
    logs/sweep_<timestamp>.jsonl     every conversation (append-only; resumable)
    logs/sweep_<timestamp>_summary.csv
    logs/sweep_<timestamp>.png       the plot

Usage:
    python sweep_psyset.py --mock --n 3                 # quick pipeline test, no GPU
    python sweep_psyset.py --n 10                       # real run
    python sweep_psyset.py --resume logs/sweep_X.jsonl  # continue an interrupted run
    python sweep_psyset.py --replot logs/sweep_X.jsonl  # re-make summary + plot only
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import toy_emotion_agents as T   # also disables torch.compile on import

# ----------------------------------------------------------------------------
# All PsySET conditions
# ----------------------------------------------------------------------------
def load_all_conditions():
    """Returns ordered {condition_name: (group, [few-shot lines] or None)}."""
    csv_path = T.fetch(T.EMOTION_CSV_URL, "emotions_fewshot_you.csv")
    json_path = T.fetch(T.PERSONALITY_JSON_URL, "attributes.json")

    conds = {"no_prompt": ("baseline", None)}

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        by_emotion = defaultdict(list)
        for r in csv.DictReader(f):
            by_emotion[r["emotion"]].append(r["text-Yes"])
    for emo, lines in by_emotion.items():
        conds[f"emotion:{emo}"] = ("emotion", lines)

    shots = json.load(open(json_path, encoding="utf-8"))["big_five_shots"]
    for trait, sides in shots.items():
        conds[f"{trait}+"] = ("personality", sides["pos"])
        conds[f"{trait}-"] = ("personality", sides["neg"])
    return conds


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
Z = NormalDist().inv_cdf


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, mid - half), min(1.0, mid + half))


def summarize(records, conds):
    """records: list of log dicts -> list of per-condition summary rows."""
    tally = defaultdict(lambda: {"hit": 0, "n_broken": 0, "fa": 0, "n_solv": 0, "limit": 0, "n": 0})
    for r in records:
        t = tally[r["condition"]]
        t["n"] += 1
        t["limit"] += r["outcome"] in ("turn_limit", "token_limit")
        if r["dataset"] == "abstention":
            t["n_broken"] += 1
            t["hit"] += r["flagged"]
        else:
            t["n_solv"] += 1
            t["fa"] += r["flagged"]

    rows = []
    for name, (group, _) in conds.items():
        t = tally.get(name)
        if not t or not t["n_broken"] or not t["n_solv"]:
            continue
        H, F = t["hit"] / t["n_broken"], t["fa"] / t["n_solv"]
        # log-linear correction (Hautus 1995) keeps z finite at 0 and 1
        Hc = (t["hit"] + 0.5) / (t["n_broken"] + 1)
        Fc = (t["fa"] + 0.5) / (t["n_solv"] + 1)
        rows.append({
            "condition": name, "group": group,
            "hit_rate": H, "false_alarm_rate": F,
            "hit_ci_lo": wilson(t["hit"], t["n_broken"])[0],
            "hit_ci_hi": wilson(t["hit"], t["n_broken"])[1],
            "fa_ci_lo": wilson(t["fa"], t["n_solv"])[0],
            "fa_ci_hi": wilson(t["fa"], t["n_solv"])[1],
            "d_prime": Z(Hc) - Z(Fc),
            "criterion_c": -(Z(Hc) + Z(Fc)) / 2,
            "limit_rate": t["limit"] / t["n"],
            "n_broken": t["n_broken"], "n_solvable": t["n_solv"],
        })
    return rows


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUP_COLOR = {"emotion": "#2a78d6", "personality": "#eb6834", "baseline": "#1baf7a"}


def plot(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2,
                         "axes.edgecolor": GRID, "axes.facecolor": SURFACE})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- Left: ROC space -------------------------------------------------------
    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color=INK_2, alpha=0.6, zorder=1)
    for x, y, txt, ha, va in [(0.02, 0.98, "ideal: flags only broken", "left", "top"),
                              (0.98, 0.98, "flags everything", "right", "top"),
                              (0.02, 0.02, "flags nothing", "left", "bottom")]:
        ax1.text(x, y, txt, color=INK_2, fontsize=8, ha=ha, va=va, style="italic")

    for r in rows:
        c = GROUP_COLOR[r["group"]]
        x, y = r["false_alarm_rate"], r["hit_rate"]
        ax1.errorbar(x, y,
                     xerr=[[x - r["fa_ci_lo"]], [r["fa_ci_hi"] - x]],
                     yerr=[[y - r["hit_ci_lo"]], [r["hit_ci_hi"] - y]],
                     fmt="none", ecolor=c, alpha=0.18, lw=1, zorder=2)
        ax1.scatter(x, y, s=55, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
    # Label points with short rank numbers (#1 = best d'), keyed to the right panel;
    # names don't fit in ROC space, and with small n many conditions share a point.
    rank = {r["condition"]: i + 1 for i, r in
            enumerate(sorted(rows, key=lambda r: -r["d_prime"]))}
    at_point = defaultdict(list)
    for r in rows:
        at_point[(r["false_alarm_rate"], r["hit_rate"])].append(rank[r["condition"]])
    for (x, y), ranks in at_point.items():
        ranks.sort()
        label = "\n".join(",".join(map(str, ranks[i:i + 4])) for i in range(0, len(ranks), 4))
        ax1.annotate(label, (x, y), xytext=(5, 3), textcoords="offset points",
                     fontsize=7.5, color=INK, va="bottom", linespacing=1.0,
                     bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none", alpha=0.85),
                     zorder=4)
    ax1.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04), aspect="equal",
            xlabel="False-flag rate on solvable GSM8K  (lower is better)",
            ylabel="Flag rate on broken AbstentionBench  (higher is better)")
    ax1.set_title("Where each condition sits", loc="left", fontsize=11, color=INK)
    ax1.grid(color=GRID, lw=0.6)
    ax1.set_axisbelow(True)

    # --- Right: d' ranking -----------------------------------------------------
    srt = sorted(rows, key=lambda r: rank[r["condition"]], reverse=True)  # #1 at top
    ys = range(len(srt))
    ax2.barh(ys, [r["d_prime"] for r in srt], height=0.7,
             color=[GROUP_COLOR[r["group"]] for r in srt], edgecolor=SURFACE, linewidth=2)
    ax2.axvline(0, color=INK_2, lw=1)
    neutral = next((r["d_prime"] for r in rows if r["condition"] == "emotion:neutral"), None)
    if neutral is not None:
        ax2.axvline(neutral, color=INK_2, lw=1, ls=":")
        ax2.text(neutral, len(srt) - 0.4, " neutral", color=INK_2, fontsize=8, va="bottom")
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([f"#{rank[r['condition']]}  {r['condition'].replace('emotion:', '')}   "
                         f"(c={r['criterion_c']:+.2f})" for r in srt], fontsize=8)
    for y, r in zip(ys, srt):
        ax2.text(r["d_prime"], y, f" {r['d_prime']:.2f}" if r["d_prime"] >= 0 else f"{r['d_prime']:.2f} ",
                 va="center", ha="left" if r["d_prime"] >= 0 else "right", fontsize=7, color=INK)
    ax2.set_xlabel("d′  (discrimination: broken vs solvable; higher is better)")
    ax2.set_title("Sensitivity ranking   (c < 0 leans 'flag', c > 0 leans 'don't flag')",
                  loc="left", fontsize=11, color=INK)
    ax2.grid(axis="x", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)

    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=g)
               for g, c in GROUP_COLOR.items() if any(r["group"] == g for r in rows)]
    handles.append(Line2D([], [], ls="--", color=INK_2, label="chance (no discrimination)"))
    n_b = rows[0]["n_broken"] if rows else 0
    fig.suptitle(f"PsySET conditions: detecting broken problems  "
                 f"(n = {n_b} broken + {n_b} solvable per condition; bars = 95% Wilson CI; "
                 f"point labels = rank on the right)",
                 x=0.01, y=0.985, ha="left", fontsize=12, color=INK)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.95),
               frameon=False, ncol=len(handles), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    print(f"Plot saved to {out_png}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def read_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_outputs(log_path, conds):
    rows = summarize(read_log(log_path), conds)
    stem = Path(log_path).with_suffix("")
    with open(f"{stem}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'condition':22s} {'hit':>5s} {'FA':>5s} {'d′':>6s} {'c':>6s} {'limit':>6s}")
    for r in sorted(rows, key=lambda r: -r["d_prime"]):
        print(f"{r['condition']:22s} {r['hit_rate']:5.2f} {r['false_alarm_rate']:5.2f} "
              f"{r['d_prime']:6.2f} {r['criterion_c']:+6.2f} {r['limit_rate']:6.2f}")
    print(f"\nSummary saved to {stem}_summary.csv")
    plot(rows, f"{stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="paired problems per condition")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="use MockLLM (no GPU)")
    ap.add_argument("--resume", default=None, help="append to an existing sweep log")
    ap.add_argument("--replot", default=None, help="only rebuild summary + plot from a log")
    ap.add_argument("--only", nargs="*", default=None, help="subset of condition names")
    args = ap.parse_args()

    conds = load_all_conditions()
    if args.only:
        conds = {k: v for k, v in conds.items() if k in args.only}

    if args.replot:
        write_outputs(args.replot, conds)
        return

    T.LOG_DIR.mkdir(exist_ok=True)
    log_path = Path(args.resume) if args.resume else \
        T.LOG_DIR / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    done = {(r["condition"], r["pair_id"], r["dataset"])
            for r in read_log(log_path)} if log_path.exists() else set()

    data = T.load_paired_gsm8k(args.n, args.seed)
    llm = T.MockLLM() if args.mock else T.LlamaLLM()
    total = len(conds) * len(data)
    print(f"{len(conds)} conditions x {len(data)} problems = {total} conversations "
          f"({len(done)} already done)")

    with open(log_path, "a", encoding="utf-8") as log:
        for cond, (group, lines) in conds.items():
            emo = T.emotion_block(lines) if lines else ""
            for item in data:
                key = (cond, item["pair_id"], item["dataset"])
                if key in done:
                    continue
                res, transcript, toks = T.run_conversation(llm, item["question"], emo)
                log.write(json.dumps({
                    "condition": cond, "group": group, **item, **res, "tokens_used": toks,
                    "transcript": [{"agent": a, "text": t} for a, t in transcript],
                }) + "\n")
                log.flush()   # so --resume works after a crash / job timeout
                print(f"{cond:22s} {item['dataset']:10s} {item['pair_id']:24s} "
                      f"flagged={res['flagged']} ({res['outcome']})")

    print(f"\nConversations logged to {log_path}")
    write_outputs(log_path, conds)


if __name__ == "__main__":
    main()
