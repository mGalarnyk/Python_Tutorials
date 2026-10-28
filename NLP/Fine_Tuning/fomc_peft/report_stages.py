"""One report per pipeline stage, each built to teach one thing.

- prepare_data  -> what is in the data (class balance, tone by year, the words that tilt a label)
- train_one     -> how training went (live curves) and what the model gets wrong (errors tab)
- grid / main   -> the PEFT trade-off: accuracy vs GPU memory across models and methods
- release       -> the evidence behind the ship / don't-ship decision

Stdlib only except data_report (pandas + openpyxl, present in the CPU image).
"""

from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

import config as cfg
import report_viz as v

LABELS = ("dovish", "hawkish", "neutral")
STOP = set(
    "the a an of to and in that for on is are was were be been by with as at this its it from or which "
    "have has had their they we our would will not but also than more some over these those such other "
    "into about after there can could may might should".split()
)


# --------------------------------------------------------------------- data


def _raw_labeled(split: str, seed: int):
    """All labeled sentences for one paper seed (train + test xlsx), with year."""
    import pandas as pd

    from load_data import download_split

    frames = [pd.read_excel(download_split(kind, str(seed), split)) for kind in ("train", "test")]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["sentence"].apply(lambda s: isinstance(s, str))].copy()
    df["stance"] = df["label"].map(dict(enumerate(LABELS)))
    return df


def _tilt_words(df, k: int = 10, min_count: int = 8) -> tuple[list, list]:
    """Hawkish-vs-dovish log-odds with an informative Dirichlet prior (Monroe et al., 2008)."""
    tok = lambda s: {w for w in re.findall(r"[a-z][a-z\-']+", s.lower()) if w not in STOP and len(w) > 2}  # noqa: E731
    counts = {s: collections.Counter() for s in LABELS}
    for sent, stance in zip(df["sentence"], df["stance"]):
        counts[stance].update(tok(sent))
    every = sum(counts.values(), collections.Counter())
    total = sum(every.values())
    nh, nd = sum(counts["hawkish"].values()), sum(counts["dovish"].values())
    scored = []
    for w, c in every.items():
        if c < min_count:
            continue
        a = c / total * 1000
        yh, yd = counts["hawkish"][w], counts["dovish"][w]
        delta = math.log((yh + a) / (nh + 1000 - yh - a)) - math.log((yd + a) / (nd + 1000 - yd - a))
        scored.append((delta / math.sqrt(1 / (yh + a) + 1 / (yd + a)), w, yh, yd))
    scored.sort(reverse=True)
    hawk = [(w, z, f"'{w}': {yh} hawkish vs {yd} dovish sentences (z = {z:.1f})") for z, w, yh, yd in scored[:k]]
    dove = [(w, -z, f"'{w}': {yd} dovish vs {yh} hawkish sentences (z = {-z:.1f})") for z, w, yh, yd in scored[-k:][::-1]]
    return dove, hawk


def data_report(split: str = cfg.DEFAULT_SPLIT, seed: int = cfg.DEFAULT_SEED) -> str:
    df = _raw_labeled(split, seed)
    n = len(df)
    share = df["stance"].value_counts(normalize=True).to_dict()
    neutral = share.get("neutral", 0.0)
    # Always answering "neutral": accuracy = neutral share; weighted F1 = share x F1(neutral).
    majority_f1 = neutral * (2 * neutral / (1 + neutral))
    years = df.groupby("year")["stance"].value_counts(normalize=True).unstack().fillna(0)
    per_year = df.groupby("year").size()
    years = years[per_year >= 40]
    dove_peak = years["dovish"].idxmax()
    last = int(years.index.max())
    dove_words, hawk_words = _tilt_words(df)

    bars = v.hbars(
        [(s, share.get(s, 0) * 100, v.STANCE_COLOR[s], f"{s}: {int(share.get(s, 0) * n)} of {n} sentences")
         for s in LABELS],
        vmax=100, fmt="{:.0f}%", width=430,
    )
    tone = v.lines(
        [
            {"name": s, "color": v.STANCE_COLOR[s],
             "points": [(int(y), float(years.loc[y, s]) * 100, f"{y}: {years.loc[y, s]:.0%} {s} ({per_year[y]} sentences)")
                        for y in years.index]}
            for s in ("dovish", "hawkish")
        ],
        x_label="year", y_fmt="{:.0f}%", width=430, height=230, y_min=0,
        marks=[(2008, "2008 crisis"), (2022, "2022 hikes")],
    )
    words = v.diverging_words(dove_words, hawk_words, v.STANCE_COLOR["dovish"], v.STANCE_COLOR["hawkish"],
                              "pushes toward dovish", "pushes toward hawkish", width=900)
    # Clearest example per label: a short sentence containing that label's strongest cue
    # word (neutral: one with none of the cue words).
    cues = {"dovish": [w for w, _, _ in dove_words], "hawkish": [w for w, _, _ in hawk_words]}
    all_cues = cues["dovish"] + cues["hawkish"]
    examples = []
    for s in LABELS:
        pick = df[df["stance"] == s].copy()
        pick["len"] = pick["sentence"].str.len()
        pick = pick[(pick["len"] > 60) & (pick["len"] < 160)].sort_values("len")
        lowered = pick["sentence"].str.lower()
        if s in cues:
            hit = pick[lowered.str.contains(rf"\b{cues[s][0]}\b") & ~lowered.str.contains(
                "|".join(rf"\b{w}\b" for w in cues["hawkish" if s == "dovish" else "dovish"]))]
        else:
            hit = pick[~lowered.str.contains("|".join(rf"\b{w}\b" for w in all_cues))]
        row = (hit if len(hit) else pick).iloc[0]
        examples.append(f"<tr><td>{v.stance_pill(s)}</td><td class='quote'>{v._t(row['sentence'])}</td><td>{int(row['year'])}</td></tr>")

    body = (
        v.tiles([
            ("Labeled sentences", f"{n:,}", f"FOMC statements, minutes, speeches {int(df['year'].min())}–{int(df['year'].max())}"),
            ("Neutral", f"{neutral:.0%}", "the most common label"),
            ("Always-neutral baseline", f"{majority_f1:.2f}", "weighted F1 of a model that never commits"),
            ("Paper's best", f"{cfg.paper_f1('roberta-large', split):.3f}", "RoBERTa-large, full fine-tune"),
        ])
        + v.takeaway(
            f"<b>Half the data is neutral.</b> A model that always says \"neutral\" is right {neutral:.0%} of the time "
            f"but scores only {majority_f1:.2f} weighted F1, which is why the paper reports weighted F1 and why "
            "0.71 is a real result."
        )
        + "<div class='row2'>"
        + v.figure("Label mix", "Share of all labeled sentences", bars)
        + v.figure("Tone by year", "Share of each year's sentences labeled dovish or hawkish (years with 40+ sentences)",
                   tone, legend=[("dovish", v.STANCE_COLOR["dovish"]), ("hawkish", v.STANCE_COLOR["hawkish"])])
        + "</div>"
        + v.takeaway(
            f"<b>Stance is about language, not the rate decision.</b> Dovish language peaks in {dove_peak} "
            f"({years.loc[dove_peak, 'dovish']:.0%}), the near-zero-rate era. Yet {last}, a year of rapid hikes, is only "
            f"{years.loc[last, 'hawkish']:.0%} hawkish and {years.loc[last, 'neutral']:.0%} neutral: the model learns how "
            "the Fed talks, one sentence at a time."
        )
        + "<h3>Which words tilt a sentence</h3>"
        + v.figure("Hawkish vs dovish words",
                   "Log-odds with an informative prior (Monroe et al., 2008); bar = how strongly the word separates the two labels",
                   words)
        + v.takeaway(
            f"<b>The labels are readable.</b> Hawkish sentences talk about pressure: "
            f"{', '.join(w for w, _, _ in hawk_words[:5])}. Dovish ones talk about slack: "
            f"{', '.join(w for w, _, _ in dove_words[:5])}. A fine-tune has to learn these cues and when context overrides them."
        )
        + "<h3>One example per label</h3><table><tr><th>label</th><th>sentence</th><th>year</th></tr>"
        + "".join(examples) + "</table>"
        + f"<p class='muted'>Shah et al. (2023), Combined-S split, seed {seed}. Data is CC BY-NC 4.0. This task runs on "
        "the CPU image: no GPU is held while spreadsheets download.</p>"
    )
    return v.page("Stage 1 · data", "What is in the FOMC data",
                  "Sentences from Fed communications, hand-labeled hawkish (tightening), dovish (easing), or neutral.", body)


# ------------------------------------------------------------------ training


def load_predictions(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "test_predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _per_class(preds: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for s in LABELS:
        tp = sum(1 for r in preds if r["pred"] == s and r["label"] == s)
        fp = sum(1 for r in preds if r["pred"] == s and r["label"] != s)
        fn = sum(1 for r in preds if r["pred"] != s and r["label"] == s)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[s] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}
    return out


def weighted_f1(preds: list[dict]) -> float:
    pc = _per_class(preds)
    n = sum(c["support"] for c in pc.values()) or 1
    return sum(c["f1"] * c["support"] for c in pc.values()) / n


def errors_section(preds: list[dict], k: int = 6) -> str:
    """Confusion matrix, per-class F1, and the model's most confident mistakes."""
    if not preds:
        return "<p class='muted'>Test predictions appear here when the run finishes.</p>"
    idx = {s: i for i, s in enumerate(LABELS)}
    matrix = [[0] * 3 for _ in LABELS]
    for r in preds:
        matrix[idx[r["label"]]][idx[r["pred"]]] += 1
    pc = _per_class(preds)
    off = [(matrix[i][j], LABELS[i], LABELS[j]) for i in range(3) for j in range(3) if i != j]
    top_n, top_true, top_pred = max(off)
    polar = matrix[idx["hawkish"]][idx["dovish"]] + matrix[idx["dovish"]][idx["hawkish"]]
    wrong = sorted((r for r in preds if r["pred"] != r["label"]), key=lambda r: -r["probs"][r["pred"]])[:k]
    rows = "".join(
        f"<tr><td class='quote'>{v._t(r['text'])}</td><td>{v.stance_pill(r['label'])}</td>"
        f"<td>{v.stance_pill(r['pred'])} {r['probs'][r['pred']]:.0%}</td></tr>"
        for r in wrong
    )
    worst = min(LABELS, key=lambda s: pc[s]["f1"])
    source = preds[0].get("source", "re-scored from checkpoint")
    f1_bars = v.hbars(
        [(s, pc[s]["f1"], v.STANCE_COLOR[s],
          f"{s}: F1 {pc[s]['f1']:.3f}, precision {pc[s]['precision']:.2f}, recall {pc[s]['recall']:.2f}, {pc[s]['support']} sentences")
         for s in LABELS],
        vmax=1, fmt="{:.2f}", width=400,
    )
    return (
        v.takeaway(
            f"<b>Most mistakes are about neutral, not about direction.</b> The biggest confusion is true {top_true} "
            f"predicted {top_pred} ({top_n} sentences). Hawkish and dovish are swapped only {polar} times out of "
            f"{len(preds)}. The weakest class is {worst} (F1 {pc[worst]['f1']:.2f})."
        )
        + "<div class='row2'>"
        + v.figure("Confusion matrix", "Rows: true label. Cells: count and share of that row.", v.confusion(matrix, list(LABELS)))
        + v.figure("F1 by label", "How well each class is recovered", f1_bars)
        + "</div><h3>Most confident mistakes</h3>"
        + "<p class='muted'>Where the model was surest and still wrong. Some are label noise; most are sentences "
        "where one cue word points the other way.</p>"
        + f"<table><tr><th>sentence</th><th>true</th><th>predicted</th></tr>{rows}</table>"
        + f"<p class='muted'>Test split, {len(preds)} sentences, weighted F1 {weighted_f1(preds):.3f} ({source}).</p>"
    )


def zero_shot_report(model_key: str, seed: int, metrics: dict, preds: list[dict]) -> str:
    f1 = (metrics.get("test") or {}).get("weighted_f1", 0.0)
    how = ("the same prompt, scored by the likeliest answer word" if cfg.is_causal(model_key)
           else "an untrained classifier head (an encoder is not a classifier until one is trained)")
    body = (
        v.tiles([("Test weighted F1", f"{f1:.3f}", "no fine-tuning"),
                 ("Paper's best", f"{metrics.get('paper_weighted_f1') or 0:.3f}", "RoBERTa-large, full fine-tune"),
                 ("Trainable params", "0", "pretrained weights only")])
        + v.takeaway(f"<b>The starting point.</b> {cfg.display_name(model_key)} with {how}. Compare this tab with a "
                     "fine-tuned run of the same model to see what training buys.")
        + "<h3>What it gets wrong</h3>" + errors_section(preds)
    )
    return v.page("Baseline · no fine-tuning", f"{cfg.display_name(model_key)} · seed {seed}",
                  "The pretrained model on the same test split, before any training.", body)


def training_report(evals: list[dict], model_key: str, method: str, seed: int, *,
                    patience: int = 7, metrics: dict | None = None, preds: list[dict] | None = None) -> str:
    xs = [int(e["epoch"]) for e in evals]
    f1 = [float(e["val_f1"]) for e in evals]
    ce = [float(e["val_ce"]) for e in evals]
    best = max(range(len(f1)), key=f1.__getitem__)
    last = evals[-1]
    counter = int(last.get("early_stopping_count", 0))
    status = "finished" if metrics else ("early-stopping soon" if counter >= patience - 2 else "training")
    min_ce = min(range(len(ce)), key=ce.__getitem__)
    title = f"{cfg.display_name(model_key)} · {v.METHOD_LABEL.get(method, method)} · seed {seed}"
    tl = [
        ("Status", status, f"epoch {last['epoch']}"),
        ("Best val F1", f"{f1[best]:.3f}", f"at epoch {xs[best]}"),
        ("Early-stop counter", f"{counter} / {patience}", "epochs without a 0.01 gain"),
    ]
    if metrics:
        test = (metrics.get("test") or {}).get("weighted_f1")
        mem = metrics.get("peak_mem_gb")
        tl.append(("Test weighted F1", f"{test:.3f}" if test is not None else "—", f"paper {metrics['paper_weighted_f1']:.3f}" if metrics.get("paper_weighted_f1") else "paper —"))
        tl.append(("Peak GPU memory", f"{mem:.1f} GB" if mem else "—", f"{(metrics.get('wall_seconds') or 0) / 60:.0f} min wall"))
    if len(xs) > 2 and min_ce < len(ce) - 2:
        lesson = (f"<b>Overfitting is visible.</b> Validation loss bottomed out at epoch {xs[min_ce]} and has risen since, "
                  f"while F1 peaked at epoch {xs[best]}. The test score uses the epoch {xs[best]} weights, not the last ones.")
    else:
        lesson = "<b>Still improving.</b> Validation loss is falling; early stopping waits for 7 epochs without a 0.01 gain."
    curves = (
        "<div class='row2'>"
        + v.figure("Validation weighted F1", "Higher is better; circled point is the epoch kept",
                   v.lines([{"name": "val F1", "color": v.METHOD_COLOR.get(method, "#2a78d6"),
                             "points": [(x, y, f"epoch {x}: val F1 {y:.3f}") for x, y in zip(xs, f1)]}],
                           x_label="epoch", y_fmt="{:.2f}", best=("val F1", best)))
        + v.figure("Validation loss", "Cross-entropy; rising while F1 holds = growing overconfidence",
                   v.lines([{"name": "val loss", "color": "#7a7974",
                             "points": [(x, y, f"epoch {x}: val loss {y:.3f}") for x, y in zip(xs, ce)]}],
                           x_label="epoch", y_fmt="{:.2f}"))
        + "</div>"
    )
    errors = ("<h3>What the model gets wrong</h3>" + errors_section(preds)) if preds else ""
    return v.page("Stage 2 · train", title,
                  "Redrawn after every epoch while the task runs. Validation is 20% of the training file, as in the paper.",
                  v.tiles(tl) + v.takeaway(lesson) + curves + errors)


# ---------------------------------------------------------------------- grid


def grid_section(runs: list[dict], split: str = cfg.DEFAULT_SPLIT) -> str:
    """Accuracy vs peak GPU memory: the PEFT trade-off in one chart."""
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in runs:
        if r.get("weighted_f1") is not None:
            groups[(str(r["model"]), str(r["method"]))].append(r)
    points = []
    for (model, method), rs in groups.items():
        if method == "zeroshot":
            continue  # inference memory is not comparable to training memory
        mems = [float(r["peak_mem_gb"]) for r in rs if r.get("peak_mem_gb")]
        if not mems:
            continue
        f1 = sum(float(r["weighted_f1"]) for r in rs) / len(rs)
        mem = sorted(mems)[len(mems) // 2]
        name = f"{cfg.display_name(model).replace('Nemotron-3-Nano-4B', 'Nemotron-4B')} {v.METHOD_LABEL[method]}"
        points.append({"x": mem, "y": f1, "color": v.METHOD_COLOR[method], "label": name,
                       "tip": f"{name}: test F1 {f1:.3f} over {len(rs)} seed(s), peak {mem:.1f} GB"})
    paper = cfg.paper_f1("roberta-large", split)
    best = max(points, key=lambda p: (p["y"], -p["x"])) if points else None
    lightest_good = min((p for p in points if paper and p["y"] >= paper - 0.01), key=lambda p: p["x"], default=None)
    lesson = ""
    if lightest_good:
        lesson = v.takeaway(
            f"<b>Up and to the left wins.</b> {lightest_good['label']} reaches F1 {lightest_good['y']:.3f} at "
            f"{lightest_good['x']:.1f} GB, within 0.01 of the paper's full fine-tune ({paper:.3f}). "
            "Moving right buys almost no accuracy: LoRA and QLoRA trim gradients and optimizer state, and on "
            "short sentences the remaining memory is mostly activations."
        )
    del best
    chart = v.figure(
        "Accuracy vs GPU memory", "Each dot is a model + method on the RTX PRO 5000 laptop (24 GB); hover for seeds",
        v.scatter(points, x_label="peak GPU memory (GB)", y_label="test weighted F1", ref_y=paper,
                  ref_label=f"Shah et al. RoBERTa-large full FT {paper:.3f}", x_fmt="{:.0f} GB"),
        legend=[(v.METHOD_LABEL[m], v.METHOD_COLOR[m]) for m in ("full", "lora", "qlora")],
    )
    return why_finetune_section(groups, split) + why_finetune_section(groups, split, method="lora", follow_up=True) + lesson + chart


def why_finetune_section(groups: dict, split: str = cfg.DEFAULT_SPLIT, method: str = "qlora",
                         follow_up: bool = False) -> str:
    """Each model before (no fine-tuning) and after one method (QLoRA by default), from our runs.

    One method for every model keeps the comparison like for like. follow_up=True draws the
    same chart for a second method under the first, without repeating the takeaway.
    """
    mean = lambda rs: sum(float(r["weighted_f1"]) for r in rs) / len(rs)  # noqa: E731
    rows, lifts = [], []
    for model in ("roberta-base", "roberta-large", "nemotron-nano-4b"):
        before, after = groups.get((model, "zeroshot")), groups.get((model, method))
        if not before or not after:
            continue
        b, a = mean(before), mean(after)
        name = cfg.display_name(model).replace("Nemotron-3-Nano-4B", "Nemotron-4B")
        how = "zero-shot prompt" if cfg.is_causal(model) else "untrained head"
        rows.append({"label": name, "before": b, "after": a, "after_color": v.METHOD_COLOR[method],
                     "tip_before": f"{name}, no fine-tuning ({how}): F1 {b:.3f} over {len(before)} seed(s)",
                     "tip_after": f"{name}, {v.METHOD_LABEL[method]}: F1 {a:.3f} over {len(after)} seed(s)"})
        lifts.append((name, how, b, a))
    if not rows:
        return ""
    label = v.METHOD_LABEL[method]
    llm = next((x for x in lifts if "Nemotron" in x[0]), None)
    enc = [x for x in lifts if "RoBERTa" in x[0]]
    msg = "<b>Fine-tuning is what makes these models useful here.</b> "
    if enc:
        def span(vals):  # "0.173 to 0.188", or "about 0.690" when both round the same
            lo, hi = f"{min(vals):.3f}", f"{max(vals):.3f}"
            return f"about {lo}" if lo == hi else f"{lo} to {hi}"

        msg += (f"A pretrained encoder is not a classifier until you train one: with an untrained head, RoBERTa scores "
                f"{span([x[2] for x in enc])}, and {label} takes it to {span([x[3] for x in enc])}. ")
    if llm:
        msg += (f"An LLM gets part of the way on its own: {llm[0]} answers the prompt at {llm[2]:.3f} zero-shot, and "
                f"{label} adds {llm[3] - llm[2]:+.3f} to reach {llm[3]:.3f}.")
    if follow_up:
        return v.figure(f"Why fine-tune? ({label})",
                        f"Gray: the pretrained model with no fine-tuning. Orange: the same model after {label}. Mean test F1 over seeds.",
                        v.dumbbell(rows, fmt="{:.3f}"),
                        legend=[("No fine-tuning", v.BEFORE_COLOR), (label, v.METHOD_COLOR[method])])
    return (
        v.takeaway(msg)
        + v.figure(f"Why fine-tune? ({label})", f"Gray: the pretrained model with no fine-tuning. Green: the same model after {label}. Mean test F1 over seeds.",
                   v.dumbbell(rows, fmt="{:.3f}"),  # same precision as the results table
                   legend=[("No fine-tuning", v.BEFORE_COLOR), (label, v.METHOD_COLOR[method])])
    )


# ------------------------------------------------------------------- release


def release_report(rows: list[dict], model_key: str, method: str, mean: float, paper: float,
                   status: str, preds: list[dict] | None = None, shown_seed: int | None = None) -> str:
    f1s = [float(r["weighted_f1"]) for r in rows]
    spread = max(f1s) - min(f1s)
    gap = mean - paper
    if abs(gap) <= max(spread, 0.005):
        verdict = (f"<b>The candidate ties the baseline.</b> Mean F1 {mean:.3f} vs {paper:.3f}: a gap of {gap:+.3f}, "
                   f"smaller than the {spread:.3f} spread between seeds. Matching a full fine-tune with a 4-bit adapter "
                   "on a laptop GPU is the win; calling it better would overread the noise.")
    elif gap > 0:
        verdict = (f"<b>The candidate beats the baseline.</b> Mean F1 {mean:.3f} vs {paper:.3f}, a gap of {gap:+.3f}, "
                   f"larger than the {spread:.3f} spread between seeds.")
    else:
        verdict = (f"<b>The candidate trails the baseline.</b> Mean F1 {mean:.3f} vs {paper:.3f} ({gap:+.3f}), beyond the "
                   f"{spread:.3f} seed spread. Shipping it would be a regression.")
    decision = (f"ship {shown_seed}" if status.startswith("Approved") else
                "nothing shipped" if status.startswith(("Not approved", "'")) else "pending")
    seeds = v.strip([(f"seed {r['seed']}", float(r["weighted_f1"]), f"seed {r['seed']}: test F1 {float(r['weighted_f1']):.3f}")
                     for r in rows], ref=paper, ref_label=f"baseline {paper:.3f}", color=v.METHOD_COLOR.get(method, "#2a78d6"))
    body = (
        v.tiles([
            ("Test weighted F1", f"{mean:.3f}", f"mean of {len(rows)} seed(s)"),
            ("Baseline", f"{paper:.3f}", "Shah et al. RoBERTa-large, full fine-tune"),
            ("Seed spread", f"{spread:.3f}", "max minus min test F1"),
            ("Decision", decision, "a person picks the seed"),
        ])
        + v.takeaway(verdict)
        + v.figure("Test F1 by seed", "Zoomed axis; dashed line is the paper's 3-seed mean", seeds)
        + ((f"<h3>What seed {shown_seed} gets wrong</h3>" + errors_section(preds, k=4)) if preds else "")
        + f"<p><b>{v._t(status)}</b></p>"
    )
    return v.page("Stage 4 · release", f"Which {cfg.display_name(model_key)} {v.METHOD_LABEL.get(method, method)} adapter ships?",
                  "A person picks a seed, or none, before anything is merged for serving.", body)


def grid_report(runs: list[dict], split: str = cfg.DEFAULT_SPLIT) -> str:
    """Stage 3: every finished cell, before vs after fine-tuning, and accuracy vs memory."""
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in runs:
        if r.get("weighted_f1") is not None:
            groups[(str(r["model"]), str(r["method"]))].append(r)
    order = {"zeroshot": 0, "full": 1, "lora": 2, "qlora": 3}
    paper = cfg.paper_f1("roberta-large", split)
    body_rows = []
    for (model, method), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        f1s = [float(r["weighted_f1"]) for r in rs]
        mean = sum(f1s) / len(f1s)
        sd = (sum((x - mean) ** 2 for x in f1s) / (len(f1s) - 1)) ** .5 if len(f1s) > 1 else None
        mems = [float(r["peak_mem_gb"]) for r in rs if r.get("peak_mem_gb")]
        mem = f"{sorted(mems)[len(mems) // 2]:.1f} GB" if mems and method != "zeroshot" else "—"
        body_rows.append(
            f"<tr><td>{v._t(cfg.display_name(model))}</td><td>{v._t(v.METHOD_LABEL.get(method, method))}</td>"
            f"<td><b>{mean:.3f}</b>{f' ± {sd:.3f}' if sd is not None else ''}</td><td>{len(f1s)}</td>"
            f"<td>{mean - paper:+.3f}</td><td>{mem}</td></tr>"
        )
    table = ("<h3>All finished cells</h3><table><tr><th>model</th><th>method</th><th>test weighted F1</th>"
             "<th>seeds</th><th>vs paper best (0.711)</th><th>peak GPU memory (training)</th></tr>" + "".join(body_rows) + "</table>"
             f"<p class='muted'>Paper: Shah et al. (2023) RoBERTa-large full fine-tune, {paper:.3f} (Table 5, Combined-S). "
             "All our runs on the RTX PRO 5000 laptop GPU (24 GB).</p>")
    return v.page("Stage 3 · compare", "Did fine-tuning help, and what did it cost?",
                  "Every finished cell: the pretrained model, then full fine-tuning, LoRA, and QLoRA.",
                  grid_section(runs, split) + table)
