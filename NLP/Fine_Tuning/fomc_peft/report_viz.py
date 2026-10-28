"""Small inline-SVG charts for Flyte reports (stdlib only: runs in the slim UI image).

Colors follow a validated palette. Methods are categorical (full / LoRA / QLoRA:
blue, orange, aqua; the first three slots stay distinct for color-blind readers
in a scatter). Stance is polar (dovish blue, neutral gray, hawkish red).
Magnitude uses one blue ramp. Text is always ink, never the series color, and
every mark carries a hover tooltip (<title>).
"""

from __future__ import annotations

import math
from html import escape

METHOD_COLOR = {"full": "#2a78d6", "lora": "#eb6834", "qlora": "#1baf7a"}
METHOD_LABEL = {"full": "Full fine-tune", "lora": "LoRA", "qlora": "QLoRA", "zeroshot": "No fine-tuning"}
BEFORE_COLOR = "#9a9893"  # "no fine-tuning" is a reference state, drawn neutral gray
STANCE_COLOR = {"dovish": "#2a78d6", "neutral": "#9a9893", "hawkish": "#e34948"}
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281")
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#7a7974", "#e6e5e1", "#fcfcfb"

VIZ_CSS = """
<style>
  .rv { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
        background: #fcfcfb; max-width: 980px; margin: 0 auto; padding: 8px 4px 24px; line-height: 1.45; }
  .rv h2 { font-size: 1.35em; margin: 6px 0 2px; }
  .rv h3 { font-size: 1.05em; margin: 26px 0 4px; }
  .rv .kicker { font-size: .74em; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #52514e; }
  .rv .lede { color: #52514e; margin: 4px 0 14px; max-width: 70ch; }
  .rv .takeaway { border-left: 4px solid #2a78d6; background: #f3f7fd; padding: 10px 14px;
                  border-radius: 6px; margin: 10px 0 4px; max-width: 78ch; }
  .rv .takeaway b { font-weight: 700; }
  .rv .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 12px 0; }
  .rv .tile { border: 1px solid #e6e5e1; border-radius: 10px; padding: 10px 12px; background: #fff; }
  .rv .tile .k { font-size: .78em; color: #52514e; }
  .rv .tile .v { font-size: 1.6em; font-weight: 700; margin: 2px 0; font-variant-numeric: tabular-nums; }
  .rv .tile .s { font-size: .78em; color: #7a7974; }
  .rv .row2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; align-items: start; }
  .rv .fig { border: 1px solid #e6e5e1; border-radius: 10px; background: #fff; padding: 10px 12px; overflow-x: auto; }
  .rv .fig .t { font-weight: 600; font-size: .92em; margin: 0 0 2px; }
  .rv .fig .st { font-size: .8em; color: #52514e; margin: 0 0 6px; }
  .rv .legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: .8em; color: #52514e; margin: 2px 0 6px; }
  .rv .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
  .rv table { border-collapse: collapse; width: 100%; font-size: .88em; margin: 8px 0; }
  .rv th { text-align: left; color: #52514e; font-weight: 600; border-bottom: 1px solid #c9c7c0; padding: 6px 8px; }
  .rv td { border-bottom: 1px solid #eeede9; padding: 6px 8px; font-variant-numeric: tabular-nums; }
  .rv .pill { display: inline-block; font-size: .74em; font-weight: 700; padding: 1px 8px; border-radius: 999px;
              border: 1px solid #c9c7c0; color: #0b0b0b; background: #fff; white-space: nowrap; }
  .rv .pill::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                      margin-right: 5px; background: var(--c, #9a9893); }
  .rv .quote { font-style: italic; }
  .rv .muted { color: #7a7974; font-size: .82em; }
  .rv svg text { font-family: inherit; }
  .rv svg .hit:hover { opacity: .78; }
</style>
"""


def _t(text: object) -> str:
    return escape(str(text))


def page(kicker: str, title: str, lede: str, body: str) -> str:
    return (
        f"{VIZ_CSS}<div class='rv'><div class='kicker'>{_t(kicker)}</div>"
        f"<h2>{_t(title)}</h2><p class='lede'>{lede}</p>{body}</div>"
    )


def takeaway(html: str) -> str:
    return f"<div class='takeaway'>{html}</div>"


def tiles(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f"<div class='tile'><div class='k'>{_t(k)}</div><div class='v'>{_t(v)}</div><div class='s'>{_t(s)}</div></div>"
        for k, v, s in items
    )
    return f"<div class='tiles'>{cells}</div>"


def figure(title: str, subtitle: str, svg: str, legend: list[tuple[str, str]] | None = None) -> str:
    leg = ""
    if legend and len(legend) >= 2:
        leg = "<div class='legend'>" + "".join(
            f"<span><i style='background:{c}'></i>{_t(n)}</span>" for n, c in legend
        ) + "</div>"
    return f"<div class='fig'><p class='t'>{_t(title)}</p><p class='st'>{_t(subtitle)}</p>{leg}{svg}</div>"


def stance_pill(name: str) -> str:
    return f"<span class='pill' style='--c:{STANCE_COLOR.get(name, '#9a9893')}'>{_t(name)}</span>"


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """Round tick values (1, 2, 2.5, 5 × 10^k steps) covering [lo, hi]."""
    raw = ((hi - lo) or 1.0) / n
    exp = math.floor(math.log10(raw))
    base = raw / 10 ** exp
    step = (1 if base <= 1 else 2 if base <= 2 else 5 if base <= 5 else 10) * 10 ** exp
    v = math.floor(lo / step) * step
    ticks = []
    while True:
        ticks.append(round(v, 10))
        if v >= hi - step * 1e-9:
            return ticks
        v += step


def hbars(rows: list[tuple[str, float, str, str]], *, vmax: float | None = None, fmt: str = "{:.0f}",
          width: int = 440, label_w: int = 150, bar_h: int = 18, gap: int = 8) -> str:
    """Horizontal bars. rows = (label, value, color, tooltip). Value labels at bar ends."""
    vmax = vmax or max((r[1] for r in rows), default=1) or 1
    plot_w = width - label_w - 56
    h = len(rows) * (bar_h + gap) + 4
    out = [f"<svg width='{width}' height='{h}' viewBox='0 0 {width} {h}' role='img'>"]
    for i, (label, val, color, tip) in enumerate(rows):
        y = i * (bar_h + gap) + 2
        w = max(2.0, val / vmax * plot_w)
        out.append(
            f"<text x='{label_w - 8}' y='{y + bar_h * .72:.1f}' font-size='12' text-anchor='end' fill='{INK}'>{_t(label)}</text>"
            f"<g class='hit'><title>{_t(tip)}</title>"
            f"<rect x='{label_w}' y='{y}' width='{w:.1f}' height='{bar_h}' rx='4' fill='{color}'/></g>"
            f"<text x='{label_w + w + 6:.1f}' y='{y + bar_h * .72:.1f}' font-size='12' fill='{INK_2}'>{_t(fmt.format(val))}</text>"
        )
    out.append("</svg>")
    return "".join(out)


def diverging_words(left: list[tuple[str, float, str]], right: list[tuple[str, float, str]],
                    left_color: str, right_color: str, left_title: str, right_title: str,
                    width: int = 900) -> str:
    """Two facing bar lists around a center axis: left grows left, right grows right."""
    n = max(len(left), len(right))
    bar_h, gap, top = 18, 7, 26
    h = top + n * (bar_h + gap)
    mid = width / 2
    half = mid - 120
    vmax = max([v for _, v, _ in left] + [v for _, v, _ in right] + [1e-9])
    out = [f"<svg width='{width}' height='{h}' viewBox='0 0 {width} {h}' role='img'>",
           f"<text x='{mid - 8}' y='14' font-size='12' font-weight='600' text-anchor='end' fill='{INK}'>{_t(left_title)}</text>",
           f"<text x='{mid + 8}' y='14' font-size='12' font-weight='600' fill='{INK}'>{_t(right_title)}</text>",
           f"<line x1='{mid}' y1='{top - 6}' x2='{mid}' y2='{h}' stroke='{GRID}'/>"]
    for i, (word, v, tip) in enumerate(left):
        y = top + i * (bar_h + gap)
        w = v / vmax * half
        out.append(
            f"<g class='hit'><title>{_t(tip)}</title><rect x='{mid - 2 - w:.1f}' y='{y}' width='{w:.1f}' height='{bar_h}' rx='4' fill='{left_color}'/></g>"
            f"<text x='{mid - 8 - w:.1f}' y='{y + 13}' font-size='12' text-anchor='end' fill='{INK}'>{_t(word)}</text>"
        )
    for i, (word, v, tip) in enumerate(right):
        y = top + i * (bar_h + gap)
        w = v / vmax * half
        out.append(
            f"<g class='hit'><title>{_t(tip)}</title><rect x='{mid + 2}' y='{y}' width='{w:.1f}' height='{bar_h}' rx='4' fill='{right_color}'/></g>"
            f"<text x='{mid + 8 + w:.1f}' y='{y + 13}' font-size='12' fill='{INK}'>{_t(word)}</text>"
        )
    out.append("</svg>")
    return "".join(out)


def lines(series: list[dict], *, x_label: str, y_fmt: str = "{:.2f}", width: int = 440, height: int = 230,
          y_min: float | None = None, y_max: float | None = None, marks: list[tuple[float, str]] | None = None,
          best: tuple[str, int] | None = None) -> str:
    """One-axis line chart. series = {name, color, points: [(x, y, tip)]}. End-of-line direct labels."""
    pts = [p for s in series for p in s["points"]]
    if not pts:
        return "<p class='muted'>No data yet.</p>"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo = min(ys) if y_min is None else y_min
    hi = max(ys) if y_max is None else y_max
    if hi - lo < 1e-9:
        lo, hi = lo - .05, hi + .05
    ticks = _nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    L, R, T, B = 44, 70 if len(series) > 1 else 18, 10, 30
    pw, ph = width - L - R, height - T - B

    def X(x: float) -> float:
        return L + (x - x0) / (x1 - x0) * pw

    def Y(y: float) -> float:
        return T + (1 - (y - lo) / (hi - lo)) * ph

    out = [f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img'>"]
    for t in ticks:
        out.append(f"<line x1='{L}' x2='{L + pw}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='{GRID}'/>"
                   f"<text x='{L - 6}' y='{Y(t) + 4:.1f}' font-size='11' text-anchor='end' fill='{INK_3}'>{_t(y_fmt.format(t))}</text>")
    for xv, lab in marks or []:
        if x0 <= xv <= x1:
            out.append(f"<line x1='{X(xv):.1f}' x2='{X(xv):.1f}' y1='{T}' y2='{T + ph}' stroke='#c9c7c0' stroke-dasharray='3 3'/>"
                       f"<text x='{X(xv) + 4:.1f}' y='{T + 10}' font-size='10.5' fill='{INK_2}'>{_t(lab)}</text>")
    xt = sorted(set(int(v) for v in _nice_ticks(x0, x1, 5) if x0 <= v <= x1)) or [int(x0), int(x1)]
    for v in xt:
        out.append(f"<text x='{X(v):.1f}' y='{T + ph + 16}' font-size='11' text-anchor='middle' fill='{INK_3}'>{v}</text>")
    out.append(f"<text x='{L + pw / 2:.1f}' y='{height - 2}' font-size='11' text-anchor='middle' fill='{INK_2}'>{_t(x_label)}</text>")
    end_labels: list[list] = []
    for s in series:
        p = sorted(s["points"])
        path = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y, _ in p)
        out.append(f"<polyline points='{path}' fill='none' stroke='{s['color']}' stroke-width='2' stroke-linejoin='round'/>")
        for x, y, tip in p:
            out.append(f"<g class='hit'><title>{_t(tip)}</title><circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='9' fill='transparent'/>"
                       f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='3.2' fill='{s['color']}' stroke='{SURFACE}' stroke-width='1.5'/></g>")
        if len(series) > 1:
            lx, ly, _ = p[-1]
            end_labels.append([Y(ly) + 4, X(lx) + 6, s["name"]])
    end_labels.sort()
    for a, b in zip(end_labels, end_labels[1:]):
        if b[0] - a[0] < 12:
            b[0] = a[0] + 12
    for y, x, name in end_labels:
        out.append(f"<text x='{x:.1f}' y='{y:.1f}' font-size='11' fill='{INK}'>{_t(name)}</text>")
    if best:
        name, idx = best
        for s in series:
            if s["name"] == name and 0 <= idx < len(s["points"]):
                bx, by, _ = sorted(s["points"])[idx]
                out.append(f"<circle cx='{X(bx):.1f}' cy='{Y(by):.1f}' r='7' fill='none' stroke='{INK}' stroke-width='1.5'/>"
                           f"<text x='{X(bx):.1f}' y='{Y(by) - 11:.1f}' font-size='11' text-anchor='middle' fill='{INK}'>best {_t(y_fmt.format(by))}</text>")
    out.append("</svg>")
    return "".join(out)


def scatter(points: list[dict], *, x_label: str, y_label: str, ref_y: float | None = None, ref_label: str = "",
            width: int = 900, height: int = 340, x_fmt: str = "{:.0f}", y_fmt: str = "{:.2f}") -> str:
    """Scatter with direct point labels. points = {x, y, color, label, tip}."""
    if not points:
        return "<p class='muted'>No finished runs with memory measurements yet.</p>"
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points] + ([ref_y] if ref_y is not None else [])
    xt = _nice_ticks(0, max(xs) * 1.08)
    yt = _nice_ticks(min(ys) - .01, max(ys) + .015)
    x0, x1, y0, y1 = 0, xt[-1], yt[0], yt[-1]
    L, R, T, B = 52, 20, 14, 40
    pw, ph = width - L - R, height - T - B

    def X(x: float) -> float:
        return L + (x - x0) / (x1 - x0) * pw

    def Y(y: float) -> float:
        return T + (1 - (y - y0) / (y1 - y0)) * ph

    out = [f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img'>"]
    for t in yt:
        out.append(f"<line x1='{L}' x2='{L + pw}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='{GRID}'/>"
                   f"<text x='{L - 6}' y='{Y(t) + 4:.1f}' font-size='11' text-anchor='end' fill='{INK_3}'>{_t(y_fmt.format(t))}</text>")
    for t in xt:
        out.append(f"<text x='{X(t):.1f}' y='{T + ph + 16}' font-size='11' text-anchor='middle' fill='{INK_3}'>{_t(x_fmt.format(t))}</text>")
    out.append(f"<text x='{L + pw / 2:.1f}' y='{height - 4}' font-size='11' text-anchor='middle' fill='{INK_2}'>{_t(x_label)}</text>"
               f"<text x='12' y='{T + ph / 2:.1f}' font-size='11' text-anchor='middle' fill='{INK_2}' transform='rotate(-90 12 {T + ph / 2:.1f})'>{_t(y_label)}</text>")
    if ref_y is not None:
        out.append(f"<line x1='{L}' x2='{L + pw}' y1='{Y(ref_y):.1f}' y2='{Y(ref_y):.1f}' stroke='#7a7974' stroke-dasharray='5 4'/>"
                   f"<text x='{L + 6}' y='{Y(ref_y) - 5:.1f}' font-size='11' fill='{INK_2}'>{_t(ref_label)}</text>")
    dots = [(X(p["x"]), Y(p["y"])) for p in points]
    boxes: list[tuple[float, float, float, float]] = []
    if ref_y is not None:
        boxes.append((L + 6, Y(ref_y) - 16, L + 6 + 6.2 * len(ref_label), Y(ref_y) - 2))
        boxes.append((L, Y(ref_y) - 2, L + pw, Y(ref_y) + 2))  # the dashed line itself

    def clear(b: tuple[float, float, float, float]) -> bool:
        x0b, y0b, x1b, y1b = b
        if x0b < L or x1b > L + pw or y0b < T - 4 or y1b > T + ph:
            return False
        if any(not (x1b < a or x0b > c or y1b < bb or y0b > d) for a, bb, c, d in boxes):
            return False
        return not any(x0b - 4 < dx < x1b + 4 and y0b - 4 < dy < y1b + 4 for dx, dy in dots)

    for p in sorted(points, key=lambda q: -q["y"]):
        cx, cy = X(p["x"]), Y(p["y"])
        tw = 6.2 * len(p["label"])
        cands = [(cx + 10, cy - 14), (cx + 10, cy + 2), (cx - 10 - tw, cy - 14), (cx - 10 - tw, cy + 2),
                 (cx - tw / 2, cy - 22), (cx - tw / 2, cy + 10)]
        bx, by = next(((x, y) for x, y in cands if clear((x, y, x + tw, y + 13))), cands[0])
        boxes.append((bx, by, bx + tw, by + 13))
        out.append(f"<g class='hit'><title>{_t(p['tip'])}</title><circle cx='{cx:.1f}' cy='{cy:.1f}' r='12' fill='transparent'/>"
                   f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='{p['color']}' stroke='{SURFACE}' stroke-width='2'/></g>"
                   f"<text x='{bx:.1f}' y='{by + 10:.1f}' font-size='11' fill='{INK}'>{_t(p['label'])}</text>")
    out.append("</svg>")
    return "".join(out)


def strip(points: list[tuple[str, float, str]], *, ref: float, ref_label: str, color: str,
          width: int = 520, fmt: str = "{:.3f}") -> str:
    """Values on one zoomed axis with a reference line. points = (label, value, tip)."""
    vals = [p[1] for p in points] + [ref]
    pad = max(0.01, (max(vals) - min(vals)) * 0.25)
    ticks = _nice_ticks(min(vals) - pad, max(vals) + pad, 4)
    lo, hi = ticks[0], ticks[-1]
    L, R, T = 20, 20, 30
    h = T + 46
    X = lambda x: L + (x - lo) / (hi - lo) * (width - L - R)  # noqa: E731
    out = [f"<svg width='{width}' height='{h}' viewBox='0 0 {width} {h}' role='img'>",
           f"<line x1='{L}' x2='{width - R}' y1='{T}' y2='{T}' stroke='{GRID}'/>"]
    for t in ticks:
        out.append(f"<text x='{X(t):.1f}' y='{T + 30}' font-size='11' text-anchor='middle' fill='{INK_3}'>{_t(fmt.format(t))}</text>")
    out.append(f"<line x1='{X(ref):.1f}' x2='{X(ref):.1f}' y1='{T - 22}' y2='{T + 14}' stroke='#7a7974' stroke-dasharray='4 3'/>"
               f"<text x='{X(ref) + 5:.1f}' y='{T - 14}' font-size='11' fill='{INK_2}'>{_t(ref_label)}</text>")
    for label, val, tip in points:
        out.append(f"<g class='hit'><title>{_t(tip)}</title><circle cx='{X(val):.1f}' cy='{T}' r='12' fill='transparent'/>"
                   f"<circle cx='{X(val):.1f}' cy='{T}' r='7' fill='{color}' stroke='{SURFACE}' stroke-width='2'/></g>"
                   f"<text x='{X(val):.1f}' y='{T + 18}' font-size='10.5' text-anchor='middle' fill='{INK}'>{_t(label)}</text>")
    out.append("</svg>")
    return "".join(out)


def dumbbell(rows: list[dict], *, ref: float | None = None, ref_label: str = "", width: int = 900, fmt: str = "{:.2f}") -> str:
    """Before -> after per row on one 0-1 axis. rows = {label, before, after, after_color, tip_before, tip_after}."""
    L, R, T, row_h = 150, 30, 28, 46
    h = T + len(rows) * row_h + 26
    X = lambda x: L + x * (width - L - R)  # noqa: E731
    out = [f"<svg width='{width}' height='{h}' viewBox='0 0 {width} {h}' role='img'>"]
    for t in (0, .2, .4, .6, .8, 1.0):
        out.append(f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{T - 8}' y2='{h - 22}' stroke='{GRID}'/>"
                   f"<text x='{X(t):.1f}' y='{h - 6}' font-size='11' text-anchor='middle' fill='{INK_3}'>{t:.1f}</text>")
    if ref is not None:
        out.append(f"<line x1='{X(ref):.1f}' x2='{X(ref):.1f}' y1='{T - 14}' y2='{h - 22}' stroke='#7a7974' stroke-dasharray='4 3'/>"
                   f"<text x='{X(ref) + 5:.1f}' y='{T - 16}' font-size='11' fill='{INK_2}'>{_t(ref_label)}</text>")
    for i, r in enumerate(rows):
        y = T + i * row_h + row_h / 2
        b, a = r["before"], r["after"]
        out.append(f"<text x='{L - 12}' y='{y + 4:.1f}' font-size='12' text-anchor='end' fill='{INK}'>{_t(r['label'])}</text>"
                   f"<line x1='{X(b):.1f}' x2='{X(a):.1f}' y1='{y:.1f}' y2='{y:.1f}' stroke='#c9c7c0' stroke-width='3'/>"
                   f"<g class='hit'><title>{_t(r['tip_before'])}</title><circle cx='{X(b):.1f}' cy='{y:.1f}' r='7' fill='{BEFORE_COLOR}' stroke='{SURFACE}' stroke-width='2'/></g>"
                   f"<g class='hit'><title>{_t(r['tip_after'])}</title><circle cx='{X(a):.1f}' cy='{y:.1f}' r='7' fill='{r['after_color']}' stroke='{SURFACE}' stroke-width='2'/></g>"
                   f"<text x='{X(b):.1f}' y='{y - 12:.1f}' font-size='11' text-anchor='middle' fill='{INK_2}'>{_t(fmt.format(b))}</text>"
                   f"<text x='{X(a):.1f}' y='{y - 12:.1f}' font-size='11' font-weight='700' text-anchor='middle' fill='{INK}'>{_t(fmt.format(a))}</text>")
    out.append("</svg>")
    return "".join(out)


def confusion(matrix: list[list[int]], labels: list[str], *, cell: int = 74) -> str:
    """Confusion matrix (rows = true, cols = predicted), one blue ramp by row share."""
    n = len(labels)
    L, T = 96, 40
    w, h = L + n * cell + 10, T + n * cell + 26
    out = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' role='img'>",
           f"<text x='{L + n * cell / 2}' y='12' font-size='11' text-anchor='middle' fill='{INK_2}'>predicted</text>"]
    for j, lab in enumerate(labels):
        out.append(f"<text x='{L + j * cell + cell / 2}' y='{T - 8}' font-size='12' text-anchor='middle' fill='{INK}'>{_t(lab)}</text>")
    for i, lab in enumerate(labels):
        total = sum(matrix[i]) or 1
        out.append(f"<text x='{L - 8}' y='{T + i * cell + cell / 2 + 4}' font-size='12' text-anchor='end' fill='{INK}'>{_t(lab)}</text>")
        for j in range(n):
            v = matrix[i][j]
            share = v / total
            color = BLUE_RAMP[min(len(BLUE_RAMP) - 1, int(share * (len(BLUE_RAMP) - 1) + .5))]
            ink = "#ffffff" if share > .55 else INK
            x, y = L + j * cell, T + i * cell
            out.append(f"<g class='hit'><title>true {_t(lab)} → predicted {_t(labels[j])}: {v} of {total} ({share:.0%})</title>"
                       f"<rect x='{x + 1}' y='{y + 1}' width='{cell - 2}' height='{cell - 2}' rx='4' fill='{color}'/></g>"
                       f"<text x='{x + cell / 2}' y='{y + cell / 2}' font-size='15' font-weight='700' text-anchor='middle' fill='{ink}'>{v}</text>"
                       f"<text x='{x + cell / 2}' y='{y + cell / 2 + 16}' font-size='10.5' text-anchor='middle' fill='{ink}'>{share:.0%}</text>")
    out.append(f"<text x='14' y='{T + n * cell / 2}' font-size='11' text-anchor='middle' fill='{INK_2}' transform='rotate(-90 14 {T + n * cell / 2})'>true label</text>")
    out.append("</svg>")
    return "".join(out)
