"""Flyte / talk report for the FOMC hawkish–dovish tutorial (stdlib only)."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import config as cfg

try:
    import report_snapshot as _report_snapshot_mod
except ImportError:  # first run, before refresh_report_snapshot writes the file
    _report_snapshot_mod = None

TALK_SENTENCES = (
    "The Committee judged that a further increase in the target range would be appropriate.",
    "The Committee decided to lower the target range for the federal funds rate.",
    "Incoming data suggested that economic activity was expanding at a moderate pace.",
    "Inflation remains elevated, and the Committee is strongly committed to returning it to 2 percent.",
)

HOUSEHOLD = {
    "hawkish": "Rates likely go up. A new mortgage or a refinance gets more expensive.",
    "dovish": "Rates likely go down. Cheaper to borrow; savings yields usually follow.",
    "neutral": "No clear lean. Markets wait — your payment does not move on this sentence alone.",
}

# Fallback if a checkpoint is not on disk (e.g. a slim UI republish).
# Seed 944601 RoBERTa-large LoRA on this Mac. Third line is the teaching beat:
# a bland activity sentence still comes out hawkish.
EXAMPLE_READS = (
    {
        "text": "The Committee judged that a further increase in the target range would be appropriate.",
        "pred": "hawkish",
        "p": 0.963,
        "for_you": "Rates likely go up. A new mortgage or a refinance gets more expensive.",
    },
    {
        "text": "The Committee decided to lower the target range for the federal funds rate.",
        "pred": "dovish",
        "p": 0.868,
        "for_you": "Rates likely go down. Cheaper to borrow; savings yields usually follow.",
    },
    {
        "text": "Incoming data suggested that economic activity was expanding at a moderate pace.",
        "pred": "hawkish",
        "p": 0.685,
        "for_you": "Sounds bland. The model still leans hawkish — growth without easing can mean no cut coming.",
    },
    {
        "text": "Inflation remains elevated, and the Committee is strongly committed to returning it to 2 percent.",
        "pred": "hawkish",
        "p": 0.747,
        "for_you": "Fighting inflation first. That is the sentence that keeps a car loan expensive.",
    },
)

STANCES = (
    ("dovish", "#027a48", "#e6f4ea", "Easing", "Lower rates. Cheaper mortgage and car loan. More hiring risk later if prices run."),
    ("hawkish", "#b42318", "#fde8e6", "Tightening", "Higher rates. Cooler inflation. Borrowing hurts; a savings account pays more."),
    ("neutral", "#475467", "#eef0f3", "No clear lean", "Markets wait. Your payment does not move on this sentence alone."),
)

SNAPSHOT_JSON = cfg.OUTPUTS_DIR / "report_snapshot.json"
SNAPSHOT_PY = Path(__file__).resolve().parent / "report_snapshot.py"
_MAC_JOBS = (("roberta-base", "full"), ("roberta-base", "lora"), ("roberta-large", "full"), ("roberta-large", "lora"))


REPORT_CSS = """
<style>
  .fomc { font-family: system-ui, -apple-system, sans-serif; color: #16213e; max-width: 960px; margin: 0 auto; }
  .fomc h2 { color: #0f3460; border-bottom: 2px solid #0f3460; padding-bottom: 8px; }
  .fomc h3 { color: #0f3460; margin-top: 22px; }
  .fomc .lede { font-size: 1.05em; line-height: 1.45; }
  .fomc .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 14px 0; }
  .fomc .card { border-radius: 10px; padding: 14px 16px; border: 1px solid #dee2e6; }
  .fomc .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78em; font-weight: 700; letter-spacing: .02em; text-transform: uppercase; }
  .fomc table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .fomc th { background: #0f3460; color: #fff; padding: 8px 10px; text-align: left; }
  .fomc td { padding: 7px 10px; border-bottom: 1px solid #dee2e6; }
  .fomc .quote { font-style: italic; margin: 0 0 8px; }
  .fomc .note { background: #fff8e6; border-left: 4px solid #e6b800; padding: 10px 12px; border-radius: 4px; font-size: 0.92em; }
  .fomc .credit { font-size: 0.82em; color: #667085; }
  .fomc .hero { background: linear-gradient(135deg,#0f3460,#1a1a2e); color: #fff; border-radius: 12px; padding: 22px 24px; margin: 12px 0 18px; }
  .fomc .hero .kicker { font-size: 0.78em; letter-spacing: .08em; text-transform: uppercase; color: #f2c94c; font-weight: 700; }
  .fomc .hero .big { font-size: 2.4em; font-weight: 800; line-height: 1.05; margin: 6px 0 8px; }
  .fomc .hero p { margin: 0; color: #e2e8f0; line-height: 1.45; }
  .fomc .live { font-size: 0.82em; color: #667085; margin: 0 0 8px; }
</style>
"""


def _machine(device: str | None, who: str = "") -> str:
    if "shah" in (who or "").lower():
        return "RTX A6000"
    d = (device or "").lower()
    if d == "cuda" or "rtx" in d or "nvidia" in d or "h200" in d:
        return "RTX PRO 6000 Blackwell, 96 GB, CUDA"
    if d == "mps" or "mps" in d:
        return "MacBook Pro M4 Max, 128 GB, MPS"
    return device or "—"


def _slim_run(data: dict) -> dict:
    test = data.get("test") or {}
    return {
        "model": data.get("model"),
        "method": data.get("method"),
        "seed": data.get("seed"),
        "split": data.get("split", cfg.DEFAULT_SPLIT),
        "device": data.get("device"),
        "status": data.get("status", "complete"),
        "weighted_f1": test.get("weighted_f1", data.get("weighted_f1")),
        "epochs_trained": data.get("epochs_trained"),
        "trainable_pct": data.get("trainable_pct"),
        "wall_seconds": data.get("wall_seconds"),
        "delta_vs_paper": data.get("delta_vs_paper"),
        "paper_weighted_f1": data.get("paper_weighted_f1"),
    }


def load_completed_runs() -> list[dict]:
    """Live `metrics.json` first, then outputs, then the last bundled snapshot."""
    runs: list[dict] = []
    if cfg.CHECKPOINTS_DIR.exists():
        for path in sorted(cfg.CHECKPOINTS_DIR.glob("*/metrics.json")):
            if not cfg.is_tracked_run_dir(path.parent.name):
                continue
            data = json.loads(path.read_text())
            if data.get("status") != "complete":
                continue
            runs.append(_slim_run(data))
    if runs:
        return runs
    for path in (SNAPSHOT_JSON, cfg.OUTPUTS_DIR / "grid_summary.json"):
        if path.exists():
            raw = json.loads(path.read_text())
            items = raw.get("runs", raw) if isinstance(raw, dict) else raw
            return [_slim_run(r) for r in items if r.get("status") == "complete"]
    if _report_snapshot_mod is not None:
        return [_slim_run(r) for r in _report_snapshot_mod.SNAPSHOT.get("runs", [])]
    return []


def summarize_grid(runs: list[dict] | None = None) -> dict:
    """Shah Table 5 rows + ours 3-seed mean ± std. Stdlib; same math as run_grid."""
    runs = list(runs if runs is not None else load_completed_runs())
    paper = [
        {
            "who": "Shah et al.",
            "model": "roberta-base",
            "method": "full",
            "n_seeds": 3,
            "weighted_f1": cfg.PAPER_WEIGHTED_F1[("roberta-base", cfg.DEFAULT_SPLIT)],
            "weighted_f1_std": cfg.PAPER_WEIGHTED_F1_STD[("roberta-base", cfg.DEFAULT_SPLIT)],
            "delta_vs_paper": None,
            "machine": "RTX A6000",
            "status": "published",
            "f1_by_seed": [],
        },
        {
            "who": "Shah et al.",
            "model": "roberta-large",
            "method": "full",
            "n_seeds": 3,
            "weighted_f1": cfg.PAPER_WEIGHTED_F1[("roberta-large", cfg.DEFAULT_SPLIT)],
            "weighted_f1_std": cfg.PAPER_WEIGHTED_F1_STD[("roberta-large", cfg.DEFAULT_SPLIT)],
            "delta_vs_paper": None,
            "machine": "RTX A6000",
            "status": "published",
            "f1_by_seed": [],
        },
    ]
    ours: list[dict] = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in runs:
        grouped.setdefault((str(row["model"]), str(row["method"])), []).append(row)
    for (model, method), group in grouped.items():
        f1s = [float(r["weighted_f1"]) for r in group if r.get("weighted_f1") is not None]
        n = len(f1s)
        if not n:
            continue
        mean = statistics.fmean(f1s)
        std = statistics.stdev(f1s) if n > 1 else 0.0
        paper_f1 = cfg.paper_f1(model)
        ours.append(
            {
                "who": "ours",
                "model": model,
                "method": method,
                "n_seeds": n,
                "weighted_f1": mean,
                "weighted_f1_std": std,
                "delta_vs_paper": None if paper_f1 is None else mean - paper_f1,
                "machine": _machine(str(group[0].get("device") or ""), "ours"),
                "status": "complete" if n >= len(cfg.PAPER_SEEDS) else f"partial ({n}/{len(cfg.PAPER_SEEDS)})",
                "f1_by_seed": [round(v, 4) for v in f1s],
                "trainable_pct": group[0].get("trainable_pct"),
            }
        )
    cuda_n = sum(1 for r in runs if str(r.get("device") or "").lower() == "cuda")
    mac_n = sum(1 for r in runs if str(r.get("device") or "").lower() == "mps")
    return {
        "runs": runs,
        "rows": paper + ours,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_complete": len(runs),
        "n_mac": mac_n,
        "n_cuda": cuda_n,
        "n_mac_expected": len(_MAC_JOBS) * len(cfg.PAPER_SEEDS),
    }


def _class_only(reads: list[dict] | None) -> list[dict]:
    out = []
    for row in reads or []:
        pred = str(row.get("pred") or row.get("pred_name") or "neutral")
        text = str(row.get("text", ""))
        out.append(
            {
                "text": text,
                "pred": pred,
                "for_you": row.get("for_you") or HOUSEHOLD.get(pred, ""),
            }
        )
    return out


def persist_talk_reads(result: dict, reads: list[dict]) -> None:
    """Write class-only labels next to this cell so later reports can reuse them."""
    ckpt = result.get("checkpoint_dir")
    if not ckpt or not reads:
        return
    path = Path(ckpt) / "talk_reads.json"
    path.write_text(
        json.dumps(
            {"source": source_label(result), "reads": _class_only(reads)},
            indent=2,
        )
    )


def load_talk_reads(reads: list[dict] | None = None) -> tuple[list[dict], str]:
    """Prefer this-run reads, then a checkpoint talk_reads.json, then the snapshot."""
    if reads:
        return _class_only(reads), ""
    best: tuple[float, dict] | None = None
    if cfg.CHECKPOINTS_DIR.exists():
        for path in cfg.CHECKPOINTS_DIR.glob("*/talk_reads.json"):
            if not cfg.is_tracked_run_dir(path.parent.name):
                continue
            metrics = path.parent / "metrics.json"
            f1 = -1.0
            if metrics.exists():
                test = json.loads(metrics.read_text()).get("test") or {}
                if test.get("weighted_f1") is not None:
                    f1 = float(test["weighted_f1"])
            payload = json.loads(path.read_text())
            if best is None or f1 > best[0]:
                best = (f1, payload)
    if best:
        return _class_only(best[1].get("reads")), str(best[1].get("source") or "")
    snap = {}
    if SNAPSHOT_JSON.exists():
        snap = json.loads(SNAPSHOT_JSON.read_text())
    elif _report_snapshot_mod is not None:
        snap = _report_snapshot_mod.SNAPSHOT
    return _class_only(snap.get("talk_reads")), str(snap.get("talk_source") or "")


def _previous_snapshot_runs() -> list[dict]:
    """Runs recorded in report_snapshot.py (read as text: the module may be stale in sys.modules)."""
    if not SNAPSHOT_PY.exists():
        return []
    text = SNAPSHOT_PY.read_text()
    marker = "SNAPSHOT = "
    if marker not in text:
        return []
    return list(json.loads(text.split(marker, 1)[1]).get("runs", []))


def refresh_report_snapshot(
    runs: list[dict] | None = None,
    reads: list[dict] | None = None,
    talk_source: str | None = None,
) -> dict:
    """Rewrite snapshot files so the slim devbox image has last-known cells + labels.

    Local checkpoints win; runs from other machines already in the snapshot are
    kept, so refreshing on a laptop does not drop the GPU server's cells.
    """
    grid = summarize_grid(runs)
    prev_reads, prev_src = load_talk_reads()
    talk_reads = _class_only(reads) if reads else prev_reads
    source = talk_source or prev_src

    def _key(rec: dict) -> tuple:
        return (rec.get("model"), rec.get("method"), rec.get("seed"), rec.get("split"), rec.get("device"))

    merged_runs = list(grid["runs"])
    local = {_key(r) for r in merged_runs}
    merged_runs += [r for r in _previous_snapshot_runs() if _key(r) not in local]
    payload = {
        "updated": grid["updated"],
        "runs": merged_runs,
        "talk_reads": talk_reads,
        "talk_source": source,
    }
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_JSON.write_text(json.dumps(payload, indent=2))
    SNAPSHOT_PY.write_text(
        "# Auto-written by refresh_report_snapshot(). Do not edit by hand.\n"
        f"SNAPSHOT = {json.dumps(payload, indent=2)}\n"
    )
    grid["talk_reads"] = talk_reads
    grid["talk_source"] = source
    return grid


def dump_grid_json() -> str:
    return json.dumps(refresh_report_snapshot())


def _f1_cell(row: dict) -> str:
    mean = row.get("weighted_f1")
    if mean is None:
        return "—"
    std = row.get("weighted_f1_std")
    n = int(row.get("n_seeds") or 0)
    if n > 1 and std is not None:
        return f"{float(mean):.3f} ± {float(std):.3f}"
    return f"{float(mean):.3f}"


def _delta_cell(row: dict) -> str:
    if row.get("who") != "ours" or row.get("delta_vs_paper") is None:
        return "—"
    return f"{float(row['delta_vs_paper']):+.3f}"


def machine_row_color(machine: str) -> str:
    """Row background by Machine. Check 1 GPU/job and 4 GPUs/model before CUDA."""
    m = (machine or "").lower()
    if "a6000" in m:
        return "#eef0f3"
    if "macbook" in m or m.endswith("mps") or ", mps" in m:
        return "#dbeafe"
    if "1 gpu/job" in m:
        return "#fde8e6"
    if "gpus/model" in m:
        return "#ede9fe"
    if "rtx pro 5000" in m:
        return "#dcfce7"
    if "h200" in m or "blackwell" in m or "cuda" in m:
        return "#fef3c7"
    return "#ffffff"


def machine_color_legend_html() -> str:
    chips = (
        ("#eef0f3", "Shah et al. · RTX A6000"),
        ("#dbeafe", "This Mac · M4 Max, MPS"),
        ("#fef3c7", "1× RTX PRO 6000 Blackwell"),
        ("#dcfce7", "RTX PRO 5000 Blackwell laptop, 24 GB"),
    )
    bits = "".join(
        f'<span style="display:inline-block;margin:0 10px 6px 0;padding:2px 8px;'
        f'border-radius:999px;background:{bg};font-size:0.82em">{label}</span>'
        for bg, label in chips
    )
    return f'<p class="credit">{bits}</p>'


def _short_model(name: str) -> str:
    return cfg.display_name(str(name))


def _results_table_html(grid: dict, highlight: tuple[str, str] | None = None) -> str:
    head = (
        "<table><tr><th>Who</th><th>Model</th><th>Method</th>"
        "<th>Weighted F1</th><th>vs paper</th><th>Seeds</th><th>Status</th><th>Machine</th></tr>"
    )
    body = []
    for row in grid["rows"]:
        bg = machine_row_color(str(row.get("machine") or ""))
        extra = ""
        if highlight and row.get("who") == "ours" and (row["model"], row["method"]) == highlight:
            extra = ";outline:2px solid #0d9488"
        body.append(
            f'<tr style="background:{bg}{extra}"><td>{row["who"]}</td>'
            f'<td>{_short_model(row["model"])}</td>'
            f"<td>{row['method']}</td><td><b>{_f1_cell(row)}</b></td>"
            f"<td>{_delta_cell(row)}</td><td>{row['n_seeds']}</td>"
            f"<td>{row['status']}</td><td>{row['machine']}</td></tr>"
        )
    pending = [
        job
        for job in cfg.TABLE_JOBS
        if not any(r["model"] == job[0] and r["method"] == job[1] and r["who"] == "ours" for r in grid["rows"])
    ]
    for model, method in pending:
        machine = "RTX PRO 6000 Blackwell, 96 GB (not run)"
        bg = machine_row_color(machine)
        body.append(
            f'<tr style="background:{bg}"><td>ours</td><td>{_short_model(model)}</td>'
            f"<td>{method}</td><td>—</td><td>—</td><td>0</td><td>pending</td>"
            f"<td>{machine}</td></tr>"
        )
    return head + "".join(body) + "</table>"


def _seed_table_html(runs: list[dict]) -> str:
    if not runs:
        return "<p class='note'>No finished <code>metrics.json</code> yet.</p>"
    rows = [
        "<table><tr><th>Model</th><th>Method</th><th>Seed</th><th>Weighted F1</th><th>Epochs</th><th>Machine</th></tr>"
    ]
    for r in sorted(runs, key=lambda x: (str(x.get("model")), str(x.get("method")), int(x.get("seed") or 0))):
        f1 = r.get("weighted_f1")
        f1_s = f"{float(f1):.3f}" if f1 is not None else "—"
        ep = r.get("epochs_trained")
        ep_s = f"{float(ep):.0f}" if ep is not None else "—"
        machine = _machine(str(r.get("device") or ""), "ours")
        bg = machine_row_color(machine)
        rows.append(
            f'<tr style="background:{bg}"><td>{_short_model(r["model"])}</td>'
            f"<td>{r['method']}</td><td>{r['seed']}</td><td><b>{f1_s}</b></td>"
            f"<td>{ep_s}</td><td>{machine}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _status_board(grid: dict) -> str:
    n = grid["n_complete"]
    mac = grid["n_mac"]
    expect = grid["n_mac_expected"]
    cuda = grid["n_cuda"]
    return (
        f"<div class='grid3'>"
        f"<div class='card'><div style='font-weight:700'>Finished cells</div>"
        f"<p style='margin:6px 0 0;font-size:1.4em'><b>{n}</b></p>"
        f"<p class='credit'>from metrics.json / snapshot · {grid['updated']}</p></div>"
        f"<div class='card'><div style='font-weight:700'>This Mac (MPS)</div>"
        f"<p style='margin:6px 0 0;font-size:1.4em'><b>{mac} / {expect}</b></p>"
        f"<p class='credit'>base+large × full+LoRA × 3 seeds</p></div>"
        f"<div class='card'><div style='font-weight:700'>CUDA (not started)</div>"
        f"<p style='margin:6px 0 0;font-size:1.4em'><b>{cuda} / 18</b></p>"
        f"<p class='credit'>RTX PRO 6000 Blackwell · QLoRA lives here</p></div>"
        f"</div>"
    )


def _badge(name: str) -> str:
    color = {"dovish": "#027a48", "hawkish": "#b42318", "neutral": "#475467"}[name]
    bg = {"dovish": "#e6f4ea", "hawkish": "#fde8e6", "neutral": "#eef0f3"}[name]
    return f'<span class="badge" style="color:{color};background:{bg}">{name}</span>'


def _read_card(ex: dict, src: str) -> str:
    return (
        f'<div class="card">'
        f'<p class="quote">“{ex["text"]}”</p>'
        f'{_badge(ex["pred"])} &nbsp; {src}'
        f'<p style="margin:8px 0 0;font-size:0.92em">{ex["for_you"]}</p></div>'
    )


def _hero(reads: tuple | list) -> str:
    beat = next((r for r in reads if "moderate pace" in r["text"].lower()), None)
    if not beat:
        return ""
    return (
        '<div class="hero">'
        '<div class="kicker">The teaching beat · not keyword search</div>'
        f'<div class="big">{beat["pred"]}</div>'
        "<p>“Incoming data suggested that economic activity was expanding at a "
        "moderate pace.” Sounds like nothing. The model still labels it "
        "<b>hawkish</b> — growth without easing. That is the sentence "
        "that keeps a mortgage expensive.</p></div>"
    )


def household_tab_html() -> str:
    rows = (
        ("Mortgage / refinance", "Hawkish → payment up. Dovish → cheaper to borrow."),
        ("Car loan / credit card", "The same overnight rate shows up in a 60-month loan."),
        ("Job / hiring", "Tight policy cools inflation and, later, the labor market."),
        ("Savings yield", "Hawkish is the one time a savings account pays more."),
    )
    cards = "".join(
        f'<div class="card"><div style="font-weight:700">{title}</div>'
        f'<p style="margin:6px 0 0;font-size:0.92em">{body}</p></div>'
        for title, body in rows
    )
    return (
        f"{REPORT_CSS}<div class='fomc'><h2>What this means at home</h2>"
        "<p class='lede'>The federal funds rate is overnight bank money. "
        "Households meet it as a payment.</p>"
        f"<div class='grid3'>{cards}</div></div>"
    )


def reads_tab_html(reads: tuple | list | None = None, src: str | None = None) -> str:
    shown, loaded_src = load_talk_reads(list(reads) if reads else None)
    src = src or loaded_src or "no checkpoint labels yet"
    if not shown:
        body = "<p class='note'>No sentence labels yet. They appear after a <code>train_one</code> cell writes <code>talk_reads.json</code>.</p>"
    else:
        body = "".join(_read_card(ex, src) for ex in shown)
    return (
        f"{REPORT_CSS}<div class='fomc'><h2>Four sentences</h2>"
        f"<p class='lede'>Predicted class only (argmax), from <b>{src}</b>. "
        f"Not a probability.</p>{body}</div>"
    )


def table5_tab_html(grid: dict | None = None, highlight: tuple[str, str] | None = None) -> str:
    grid = grid or summarize_grid()
    return (
        f"{REPORT_CSS}<div class='fomc'><h2>Did we match Table 5?</h2>"
        f"{_status_board(grid)}"
        f"{_f1_bar_chart(grid)}"
        f"{_results_table_html(grid, highlight=highlight)}"
        f"{machine_color_legend_html()}"
        "<h3>Per seed (updates when a cell writes metrics.json)</h3>"
        f"{_seed_table_html(grid['runs'])}"
        "<p class='credit'>3-seed mean ± std, Combined-S, weighted F1. "
        "Same files as the notebook Results cell. Pending rows appear when that "
        "(model, method) has no finished run yet.</p></div>"
    )


def classify_step_html(step: int, total: int, ex: dict) -> str:
    return (
        f"{REPORT_CSS}<div class='fomc'>"
        f"<p class='live'>Classifying sentence {step} of {total}…</p>"
        f"{_read_card(ex, source_label(None))}</div>"
    )


def _f1_bar_chart(grid: dict | None = None) -> str:
    grid = grid or summarize_grid()
    labels = ("RoBERTa-base", "RoBERTa-large")
    keys = ("roberta-base", "roberta-large")

    def _val(who: str, method: str, model: str) -> float | None:
        for row in grid["rows"]:
            if row["who"] == who and row["method"] == method and row["model"] == model:
                return row.get("weighted_f1")
        return None

    series = {
        "Shah et al. full": tuple(_val("Shah et al.", "full", m) for m in keys),
        "Ours full": tuple(_val("ours", "full", m) for m in keys),
        "Ours LoRA": tuple(_val("ours", "lora", m) for m in keys),
    }
    colors = ["#94a3b8", "#0f3460", "#0d9488"]
    width, height = 720, 280
    ml, mr, mt, mb = 48, 16, 36, 56
    cw, ch = width - ml - mr, height - mt - mb
    y_max = 0.85
    n_groups, n_series = len(labels), len(series)
    group_w = cw / n_groups
    bar_w = group_w * 0.72 / n_series
    gap = group_w * 0.14

    def sy(v: float) -> float:
        return mt + ch - (v / y_max) * ch

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="600" fill="#16213e">Weighted F1 · Combined-S · 3-seed mean</text>',
    ]
    for i in range(5):
        yv = y_max * i / 4
        py = sy(yv)
        out.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml+cw}" y2="{py:.1f}" stroke="#e9ecef"/>'
        )
        out.append(
            f'<text x="{ml-8}" y="{py+4:.1f}" text-anchor="end" font-size="11" '
            f'fill="#667085">{yv:.2f}</text>'
        )
    for gi, label in enumerate(labels):
        gx = ml + gi * group_w + gap
        for si, (name, vals) in enumerate(series.items()):
            val = vals[gi]
            if val is None:
                continue
            bx = gx + si * bar_w
            by = sy(val)
            out.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w-2:.1f}" '
                f'height="{mt+ch-by:.1f}" fill="{colors[si]}" rx="2"/>'
            )
            out.append(
                f'<text x="{bx+bar_w/2:.1f}" y="{by-5:.1f}" text-anchor="middle" '
                f'font-size="11" fill="#16213e">{val:.3f}</text>'
            )
        out.append(
            f'<text x="{gx + n_series*bar_w/2:.1f}" y="{mt+ch+18}" '
            f'text-anchor="middle" font-size="12" fill="#475467">{label}</text>'
        )
    lx = ml
    for si, name in enumerate(series):
        out.append(
            f'<rect x="{lx+si*150}" y="{height-18}" width="10" height="10" '
            f'rx="2" fill="{colors[si]}"/>'
        )
        out.append(
            f'<text x="{lx+si*150+14}" y="{height-9}" font-size="11" fill="#16213e">{name}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def reads_from_predict(rows: list[dict]) -> list[dict]:
    """Turn `predict_texts` rows into report cards (this cell's checkpoint)."""
    out = []
    for row in rows:
        pred = str(row.get("pred_name") or row.get("pred") or "neutral")
        text = str(row.get("text", ""))
        probs = row.get("probs") or {}
        p = float(probs.get(pred, row.get("p") or 0.0))
        for_you = HOUSEHOLD.get(pred, "")
        if "moderate pace" in text.lower() and pred == "hawkish":
            for_you = (
                "Sounds bland. The model still leans hawkish — "
                "growth without easing can mean no cut coming."
            )
        out.append({"text": text, "pred": pred, "p": p, "for_you": for_you})
    return out


def source_label(result: dict | None) -> str:
    if not result:
        return "RoBERTa-large LoRA, seed 944601"
    return (
        f"{result.get('model', '?')} {result.get('method', '?')}, "
        f"seed {result.get('seed', '?')}"
    )


def fomc_talk_report(
    result: dict | None = None,
    reads: list[dict] | None = None,
    grid: dict | None = None,
) -> str:
    """Talk HTML from live metrics (or the last snapshot if this worker has none)."""
    grid = grid or summarize_grid()
    highlight = None
    if result and result.get("model") and result.get("method"):
        highlight = (str(result["model"]), str(result["method"]))
    cards = []
    for name, fg, bg, verb, body in STANCES:
        cards.append(
            f'<div class="card" style="background:{bg}">'
            f'{_badge(name)} <div style="margin-top:8px;font-weight:700">{verb}</div>'
            f'<p style="margin:6px 0 0;font-size:0.92em">{body}</p></div>'
        )
    src = source_label(result)
    this_cell = ""
    if result:
        test = result.get("test") or {}
        f1 = test.get("weighted_f1")
        delta = result.get("delta_vs_paper")
        if f1 is not None:
            delta_txt = "—" if delta is None else f"{float(delta):+.3f} vs Table 5"
            this_cell = (
                f'<p class="note"><b>This cell</b> {src}: test weighted F1 '
                f"<b>{float(f1):.3f}</b> ({delta_txt}), "
                f"device={result.get('device')}, epochs={result.get('epochs_trained')}.</p>"
            )
    shown, loaded_src = load_talk_reads(reads)
    src = source_label(result) if result else (loaded_src or src)
    if shown:
        labels_block = (
            f"<p>Argmax class from <b>{src}</b>. Rewritten when a cell writes "
            "<code>talk_reads.json</code>.</p>"
            + "".join(_read_card(ex, src) for ex in shown)
        )
    else:
        labels_block = (
            "<p class='note'>No sentence labels yet. Finish a <code>train_one</code> "
            "cell (or classify a checkpoint) and this block fills in.</p>"
        )
    return f"""
    {REPORT_CSS}
    <div class="fomc">
      <h2>FOMC hawkish–dovish</h2>
      <p class="lede">Three labels. Households meet them as a mortgage, a car loan,
      or a savings yield. The table below is the live grid.</p>
      <div class="grid3">{"".join(cards)}</div>
      {this_cell}
      <h3>What this checkpoint labels</h3>
      {labels_block}
      <h3>Live grid</h3>
      {_status_board(grid)}
      {_f1_bar_chart(grid)}
      {_results_table_html(grid, highlight=highlight)}
      {machine_color_legend_html()}
    </div>
    """


def prepare_data_report(n_files: int, split: str, seeds: list[int]) -> str:
    return (
        f"{REPORT_CSS}<div class='fomc'>"
        "<h2>FOMC Combined-S data</h2>"
        f"<p>Downloaded <b>{n_files}</b> train/test xlsx files "
        f"(split <code>{split}</code>, seeds {seeds}).</p>"
        "<p>Shah et al. (2023) hawkish / dovish / neutral labels. "
        "This is the CPU task — the GPU is not sitting idle on an Excel fetch.</p>"
        "</div>"
    )


def market_data_report(path: str) -> str:
    return (
        f"{REPORT_CSS}<div class='fomc'>"
        "<h2>Market data — not built</h2>"
        "<p>This CPU task is a placeholder for later market files. "
        "It is not part of the talk report.</p>"
        f"<p class='credit'>Wrote <code>{path}</code></p>"
        "</div>"
    )
