"""Synthetic charts with exact element boxes for nemotron-graphic-elements-v1 (SPEC §6.3).

Charts are rendered with matplotlib (Agg). Every text element's box comes from `Artist.get_window_extent()`
after drawing, so the labels are exact, reproducible from a seed, and license-clean (no download).

Class semantics follow NVIDIA's model card (PMC chart taxonomy), in `Exp.labels` order:
  chart_title  title of the chart           x_title / y_title  axis titles
  xlabel / ylabel  tick labels of each axis  legend_title / legend_label  legend title / entry texts
  mark_label   labels attached to markers    value_label  numbers attached to bars/points
  other        miscellaneous text (source notes, annotations)
Pie charts are left out: NVIDIA's pie data was auto-labelled, and which class its wedge labels belong to
isn't documented, so generating them would mean guessing the semantics.
"""

from __future__ import annotations

import io
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

LABELS = ["chart_title", "x_title", "y_title", "xlabel", "ylabel", "other", "legend_label", "legend_title",
          "mark_label", "value_label"]
L = {name: i for i, name in enumerate(LABELS)}
KINDS = ("bar", "barh", "grouped_bar", "stacked_bar", "line", "scatter")

_WORDS = ["Revenue", "Costs", "Margin", "Europe", "Asia", "Americas", "Retail", "Corporate", "Online", "Branch",
          "Loans", "Deposits", "Equity", "Bonds", "Cash", "Paris", "Lyon", "London", "Milan", "Madrid", "Q1", "Q2",
          "Q3", "Q4", "North", "South", "Clients", "Assets", "Risk", "Fees", "Card", "Mortgage", "SME", "Savings"]
_TITLES = ["Quarterly revenue by region", "Net interest income", "Customer growth", "Loan portfolio mix",
           "Operating expenses", "Évolution du chiffre d'affaires", "Répartition des actifs", "Cost-income ratio",
           "Deposits by segment", "Default rate over time", "Market share", "Résultat net par métier"]
_UNITS = ["(€m)", "(%)", "(bn)", "(k€)", "(units)", "", "", "(M EUR)"]
_SOURCES = ["Source: annual report", "Source : rapport annuel", "Note: unaudited figures", "* estimated",
            "Data as of 31/12", "Source: internal data"]
_CAPTION_WORDS = ("the of and in by for over with across compared to per from shows observed during period growth "
                  "segment revenue costs ratio quarter year business clients portfolio share market trend values "
                  "remain stable increased decreased models training data results figure total average risk").split()
_CAPTION_WORDS_FR = ("le la les des du en par pour sur avec selon entre évolution du chiffre d'affaires résultat "
                     "segment clients portefeuille part marché tendance valeurs période hausse baisse total moyenne "
                     "risque données graphique trimestre année").split()
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_STYLES = ["default", "ggplot", "bmh", "seaborn-v0_8-whitegrid", "seaborn-v0_8-darkgrid", "fivethirtyeight",
           "classic", "grayscale"]
_FONTS = ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]


@dataclass
class ChartSubset:
    train: int = 3000
    val: int = 300
    test: int = 300
    seed: int = 0

    def key(self) -> str:
        return f"synthetic-charts-v2_tr{self.train}_va{self.val}_te{self.test}_s{self.seed}"


def _fmt(v: float, fr: bool, decimals: int, pct: bool) -> str:
    s = f"{v:,.{decimals}f}"
    if fr:
        s = s.replace(",", " ").replace(".", ",")  # 1 234,5 (narrow no-break space)
    return s + (" %" if fr and pct else "%" if pct else "")


def _categories(rng: random.Random, n: int) -> list[str]:
    mode = rng.random()
    if mode < 0.3:
        start = rng.randint(2005, 2020)
        return [str(start + i) for i in range(n)]
    if mode < 0.5 and n <= 12:
        return _MONTHS[:n]
    words = rng.sample(_WORDS, k=min(n, len(_WORDS)))
    return [w if rng.random() < 0.7 else f"{w} {rng.choice(_WORDS)}" for w in words][:n]


def _caption(rng: random.Random, fr: bool, w_in: float, size: float) -> list[str]:
    """1-3 caption lines ("Figure 3: ..."), each short enough to fit the figure width."""
    words = _CAPTION_WORDS_FR if fr else _CAPTION_WORDS
    max_chars = int(w_in * 72 / (size * 0.55))
    head = f"{'Graphique' if fr else rng.choice(['Figure', 'Fig.', 'Chart'])} {rng.randint(1, 12)}: "
    lines, cur = [], head + rng.choice(_TITLES) + "."
    for _ in range(rng.randint(0, 40)):
        word = rng.choice(words)
        if len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
            if len(lines) == 3:
                break
        else:
            cur = f"{cur} {word}"
    if len(lines) < 3 and cur:
        lines.append(cur)
    return [line[:max_chars] for line in lines[:3]]


def _visible_ticks(axis):
    lo, hi = sorted(axis.get_view_interval())
    for tick in axis.get_major_ticks():
        lab = tick.label1
        if not (lab.get_visible() and lab.get_text().strip()):
            continue
        if not (lo - 1e-9 <= tick.get_loc() <= hi + 1e-9):
            continue
        yield lab


def render_chart(seed: int):
    """One random chart -> (png bytes, boxes [N,4] xyxy px, labels [N], meta)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    kind = rng.choice(KINDS)
    fr = rng.random() < 0.35
    pct = rng.random() < 0.2
    decimals = rng.choice([0, 0, 1, 2])
    n_cat = rng.randint(3, 12 if kind in ("line", "scatter") else 9)
    n_series = 1 if kind in ("bar", "barh") else rng.randint(2, 4)
    w_in, h_in, dpi = rng.uniform(4.5, 9.0), rng.uniform(3.2, 6.5), rng.choice([80, 100, 120, 150])
    base_size = rng.uniform(8, 13)
    style = rng.choice(_STYLES)
    items: list[tuple[object, int]] = []  # (text artist, class)

    with plt.style.context(style), plt.rc_context({"font.family": rng.choice(_FONTS), "font.size": base_size}):
        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
        cats = _categories(rng, n_cat)
        scale = rng.choice([10, 25]) if pct else 10 ** rng.randint(0, 4)  # percentages stay within 0-100
        data = np.abs(nrng.normal(1.0, 0.45, size=(n_series, n_cat))) * scale
        if pct and kind == "stacked_bar":
            data = data / data.sum(axis=0, keepdims=True) * 100
        series = rng.sample(_WORDS, k=n_series)
        x = np.arange(n_cat)
        artists = []
        if kind == "bar":
            artists = [ax.bar(x, data[0], color=f"C{rng.randint(0, 9)}")]
        elif kind == "barh":
            artists = [ax.barh(x, data[0], color=f"C{rng.randint(0, 9)}")]
        elif kind == "grouped_bar":
            width = 0.8 / n_series
            artists = [ax.bar(x + (i - (n_series - 1) / 2) * width, data[i], width, label=series[i])
                       for i in range(n_series)]
        elif kind == "stacked_bar":
            bottom = np.zeros(n_cat)
            for i in range(n_series):
                artists.append(ax.bar(x, data[i], 0.7, bottom=bottom, label=series[i]))
                bottom = bottom + data[i]
        elif kind == "line":
            for i in range(n_series):
                ax.plot(x, data[i], marker=rng.choice(["o", "s", "^", "", ""]), label=series[i])
        else:  # scatter
            for i in range(n_series):
                ax.scatter(x + nrng.uniform(-0.3, 0.3, n_cat), data[i], label=series[i], s=rng.uniform(15, 60))

        cat_axis = ax.yaxis if kind == "barh" else ax.xaxis
        val_axis = ax.xaxis if kind == "barh" else ax.yaxis
        cat_axis.set_ticks(x)
        cat_axis.set_ticklabels(cats)
        if kind != "barh" and rng.random() < 0.35:
            plt.setp(ax.get_xticklabels(), rotation=rng.choice([30, 45, 90]), ha="right")
        val_axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _p: _fmt(v, fr, 0 if scale >= 10 else 1, pct)))

        if rng.random() < 0.85:
            items.append((ax.set_title(rng.choice(_TITLES), fontsize=base_size * rng.uniform(1.0, 1.5)),
                          L["chart_title"]))
        cat_title = rng.choice(["Year", "Région", "Segment", "Month", "Business line", "Product"])
        val_title = rng.choice(["Amount", "Montant", "Share", "Volume", "Value"]) + " " + rng.choice(_UNITS)
        x_t, y_t = (val_title, cat_title) if kind == "barh" else (cat_title, val_title)
        if rng.random() < 0.75:
            items.append((ax.set_xlabel(x_t), L["x_title"]))
        if rng.random() < 0.8:
            items.append((ax.set_ylabel(y_t.strip()), L["y_title"]))
        legend = None
        if n_series > 1 and rng.random() < 0.9:
            legend = ax.legend(title=rng.choice(["Segment", "Légende", "Series"]) if rng.random() < 0.35 else None,
                               loc=rng.choice(["best", "upper left", "upper right", "lower right"]),
                               fontsize=base_size * 0.9)
        if kind in ("bar", "barh", "grouped_bar") and rng.random() < 0.4:
            for bars in artists:
                for t in ax.bar_label(bars, labels=[_fmt(v, fr, decimals, pct) for v in bars.datavalues],
                                      fontsize=base_size * 0.8, padding=2):
                    items.append((t, L["value_label"]))
        if kind in ("line", "scatter") and rng.random() < 0.35:
            for j in rng.sample(range(n_cat), k=min(3, n_cat)):
                i = rng.randrange(n_series)
                if rng.random() < 0.5:
                    t = ax.annotate(_fmt(data[i, j], fr, decimals, pct), (j, data[i, j]), textcoords="offset points",
                                    xytext=(0, 6), ha="center", fontsize=base_size * 0.8)
                    items.append((t, L["value_label"]))
                else:
                    t = ax.annotate(rng.choice(_WORDS), (j, data[i, j]), textcoords="offset points", xytext=(5, 5),
                                    fontsize=base_size * 0.8)
                    items.append((t, L["mark_label"]))
        # Figure-style captions under the chart (real chart crops often include them). Without them, the model
        # learns that any text below the axis is a tick label or axis title, and mislabels real captions.
        caption_lines = _caption(rng, fr, w_in, base_size) if rng.random() < 0.4 else []
        if caption_lines:
            line_h = base_size * 1.45 / 72 / h_in  # in figure coordinates
            frac = min(0.35, line_h * (len(caption_lines) + 0.6))
            fig.tight_layout(rect=(0, frac, 1, 1))
            x0 = rng.choice([0.01, 0.03])
            weight = "bold" if rng.random() < 0.3 else "normal"
            for k, line in enumerate(caption_lines):
                t = fig.text(x0, frac - (k + 1) * line_h, line, fontsize=base_size * rng.uniform(0.85, 1.0),
                             fontweight=weight if k == 0 else "normal")
                items.append((t, L["other"]))
        else:
            if rng.random() < 0.3:
                items.append((fig.text(0.01, 0.01, rng.choice(_SOURCES), fontsize=base_size * 0.75), L["other"]))
            fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        W, H = fig.canvas.get_width_height()
        for lab in _visible_ticks(ax.xaxis):
            items.append((lab, L["xlabel"]))
        for lab in _visible_ticks(ax.yaxis):
            items.append((lab, L["ylabel"]))
        if legend is not None:
            items.extend((t, L["legend_label"]) for t in legend.get_texts() if t.get_text().strip())
            if legend.get_title().get_text().strip():
                items.append((legend.get_title(), L["legend_title"]))

        boxes, labels = [], []
        for artist, cls in items:
            if not artist.get_visible() or not str(artist.get_text()).strip():
                continue
            bb = artist.get_window_extent(renderer)
            x0, x1 = max(0.0, bb.x0), min(float(W), bb.x1)
            y0, y1 = max(0.0, H - bb.y1), min(float(H), H - bb.y0)  # display origin is bottom-left
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                boxes.append([x0, y0, x1, y1])
                labels.append(cls)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)  # no bbox_inches: it would shift the coordinates
        plt.close(fig)
    meta = {"seed": seed, "kind": kind, "fr": fr, "style": style, "size": [W, H], "caption_lines": len(caption_lines)}
    return buf.getvalue(), np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, np.int64), meta


def _render_many(seeds: list[int]) -> list[tuple]:
    return [render_chart(s) for s in seeds]


def prepare(cfg: ChartSubset, out_dir: Path, workers: int = 8) -> dict:
    """Write `<out_dir>/{train,validation,test}.parquet` + manifest (idempotent). Disjoint seeds per split."""
    import multiprocessing as mp

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file() and all((out_dir / f"{s}.parquet").is_file() for s in ("train", "validation", "test")):
        return json.loads(manifest_path.read_text())
    manifest = {"generator": "nemotron_dbx.detection.charts", "config": asdict(cfg), "labels": LABELS, "splits": {}}
    offsets = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
    t0 = time.monotonic()
    for split, n in (("train", cfg.train), ("validation", cfg.val), ("test", cfg.test)):
        seeds = [cfg.seed * 10_000_000 + offsets[split] + i for i in range(n)]
        chunks = [seeds[i:i + 50] for i in range(0, len(seeds), 50)]
        with mp.get_context("spawn").Pool(workers) as pool:
            rendered = [r for chunk in pool.map(_render_many, chunks) for r in chunk]
        cols = {"image": [r[0] for r in rendered], "image_id": [f"{split}-{r[3]['seed']}" for r in rendered],
                "boxes": [r[1].reshape(-1).tolist() for r in rendered], "labels": [r[2].tolist() for r in rendered],
                "meta": [json.dumps(r[3]) for r in rendered]}
        tmp = out_dir / f"{split}.parquet.tmp"
        pq.write_table(pa.table(cols), tmp)
        tmp.replace(out_dir / f"{split}.parquet")
        counts = np.bincount(np.concatenate([r[2] for r in rendered]), minlength=len(LABELS))
        manifest["splits"][split] = {"charts": len(rendered), "boxes_per_class": dict(zip(LABELS, counts.tolist(),
                                                                                         strict=True))}
        print(f"[data] {split}: {len(rendered)} charts at {time.monotonic() - t0:.0f}s: "
              f"{manifest['splits'][split]['boxes_per_class']}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
