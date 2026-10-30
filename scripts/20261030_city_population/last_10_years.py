# ruff: noqa: E501  (long lines are Czech prose inside the HTML report and post texts)
"""How the population of Czech cities changed over the last 10 years (1 Jan 2015 → 1 Jan 2025).

Reads the latest download from data/raw/city_population/ (see download.py: ČSÚ population data
and ČÚZK RÚIAN boundaries for the map) and writes processed data, one social-media image per variant and a Czech HTML report to
outputs/city_population/<yymmdd_hhmmss>/.

ČSÚ publishes population at 31 Dec; 31 Dec 2024 (= 1 Jan 2025) is the newest year available, so
the 10-year window is 31 Dec 2014 → 31 Dec 2024.

Usage: uv run scripts/20261030_city_population/last_10_years.py
"""

import base64
import html
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import monaco

ANALYSIS = "city_population"
RAW_ROOT = monaco.DATA_DIR / "raw" / ANALYSIS
START, END = 2014, 2024  # values at 31 Dec, i.e. 1 Jan 2015 → 1 Jan 2025
PRE_END, POST_START = 2020, 2021  # the 2021 census resets the register between these two years
CITY_MIN_POP = 40_000  # "city" = municipality with at least this many residents at END
PRAHA = "554782"
SIZE_BINS = [0, 500, 2_000, 5_000, 20_000, 50_000, 100_000, 2_000_000]
SIZE_LABELS = [
    "< 500",
    "500–2 tis.",
    "2–5 tis.",
    "5–20 tis.",
    "20–50 tis.",
    "50–100 tis.",
    "100 tis.+",
]

URL_130149 = "https://csu.gov.cz/produkty/obyvatelstvo-k-3112-podle-pohlavi-v-obcich"
URL_170240 = "https://csu.gov.cz/obyvatelstvo-a-domy-v-obcich-podle-vysledku-scitani-od-roku-1869"
URL_RUIAN = (
    "https://ags.cuzk.gov.cz/arcgis/rest/services/RUIAN/Prohlizeci_sluzba_nad_daty_RUIAN/MapServer"
)
PERIOD = "1. 1. 2015 → 1. 1. 2025"
SOURCE = (
    "Zdroj: Český statistický úřad (ČSÚ), datová sada 130149 — obyvatelé s trvalým nebo dlouhodobým\n"
    "pobytem k 31. 12. 2014 a 31. 12. 2024. Data k 1. 1. 2026 zatím nebyla zveřejněna."
)

# Chart styling (reference dataviz palette).
GROW = "#2a78d6"
SHRINK = "#e34948"
PRE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": INK2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


# --------------------------------------------------------------------------------------------
# Czech number formatting
# --------------------------------------------------------------------------------------------


def pct(x: float, signed: bool = True, digits: int = 0) -> str:
    """Czech percent: '−12 %', '+3,5 %'."""
    if signed and abs(x) < 0.5 * 10 ** -(digits + 2):
        return "±0 %"
    s = f"{x * 100:{'+' if signed else ''}.{digits}f}"
    return s.replace("-", "−").replace(".", ",") + " %"


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def tis(x: float) -> str:
    """Thousands: '371 tis.'."""
    return f"{x / 1e3:,.0f}".replace(",", " ").replace("-", "−") + " tis."


# --------------------------------------------------------------------------------------------
# Load & process
# --------------------------------------------------------------------------------------------


def latest_raw() -> Path:
    return sorted(p for p in RAW_ROOT.iterdir() if p.is_dir())[-1]


def load_yearly(raw: Path) -> pd.DataFrame:
    """130149: residents (permanent + long-term) at 31 Dec 2000-2024, all territorial levels."""
    path = next((raw / "csu_130149").glob("OD_DEM05_*.CSV"))
    df = pd.read_csv(path, dtype={"uzemi_kod": str, "pohlavi_txt": str, "vek_txt": str})
    df = df[df["pohlavi_txt"].isna()]  # both sexes
    df["year"] = df["obdobi"].str[:4].astype(int)
    df["age"] = df["vek_txt"].fillna("total")
    return df[["uzemi_typ", "uzemi_kod", "uzemi_txt", "year", "age", "hodnota"]].rename(
        columns={"uzemi_typ": "level", "uzemi_kod": "code", "uzemi_txt": "name", "hodnota": "pop"}
    )


def municipalities(yearly: pd.DataFrame) -> pd.DataFrame:
    """Municipality x (age, year) population; only municipalities existing in both START and END."""
    m = yearly[yearly["level"] == "obec"].pivot_table(
        index=["code", "name"], columns=["age", "year"], values="pop"
    )
    return m.dropna(subset=[("total", START), ("total", END)]).reset_index(level="name")


def load_regions(raw: Path) -> pd.Series:
    """Municipality code -> region (kraj) name, from the census dataset 170240."""
    df = pd.read_csv(raw / "csu_170240" / "OD_SLDB_HD_at.csv", dtype={"vuzemi_kod": str})
    return df.drop_duplicates("vuzemi_kod").set_index("vuzemi_kod")["kraj_text"]


def _rings(path: Path) -> dict[str, list[np.ndarray]]:
    """RÚIAN Esri JSON -> {code: [ring as (n, 2) array in S-JTSK]}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(f["attributes"]["kod"]): [np.asarray(r, dtype=float) for r in f["geometry"]["rings"]]
        for f in data["features"]
    }


def _centroid(rings: list[np.ndarray]) -> tuple[float, float]:
    """Area centroid of the largest ring (shoelace formula)."""

    def parts(r: np.ndarray) -> tuple[float, float, float]:
        x, y = r[:, 0], r[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = cross.sum() / 2
        return (
            a,
            ((x + np.roll(x, -1)) * cross).sum() / (6 * a),
            ((y + np.roll(y, -1)) * cross).sum() / (6 * a),
        )

    a, cx, cy = max((parts(r) for r in rings), key=lambda t: abs(t[0]))
    return cx, cy


def national(yearly: pd.DataFrame) -> pd.DataFrame:
    return yearly[yearly["level"] == "stát"].pivot_table(index="year", columns="age", values="pop")


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


def _changes(df: pd.DataFrame, s: Callable[[int], pd.Series], prefix: str) -> None:
    """Add 10-year, pre-census, census-reset, post-census and chain-linked change columns."""
    df[f"{prefix}ch10"] = s(END) / s(START) - 1
    df[f"{prefix}pre"] = s(PRE_END) / s(START) - 1
    df[f"{prefix}reset"] = s(POST_START) / s(PRE_END) - 1
    df[f"{prefix}post"] = s(END) / s(POST_START) - 1
    df[f"{prefix}chain"] = (1 + df[f"{prefix}pre"]) * (1 + df[f"{prefix}post"]) - 1


def city_table(muni: pd.DataFrame, regions: pd.Series) -> pd.DataFrame:
    m = muni[muni[("total", END)] >= CITY_MIN_POP]
    c = pd.DataFrame({"name": m["name"]})
    c["kraj"] = regions.reindex(c.index)
    for y in range(START, END + 1):
        c[f"pop{y}"] = m[("total", y)]
    _changes(c, lambda y: m[("total", y)], "")
    c["wa2014"], c["wa2024"] = m[("15 až 64", START)], m[("15 až 64", END)]
    _changes(c, lambda y: m[("15 až 64", y)], "wa_")
    c["s65_2014"] = m[("65 a více", START)] / m[("total", START)]
    c["s65_2024"] = m[("65 a více", END)] / m[("total", END)]
    c["s014_2014"] = m[("0 až 14", START)] / m[("total", START)]
    c["s014_2024"] = m[("0 až 14", END)] / m[("total", END)]
    return c.sort_values("ch10")


def size_class_table(muni: pd.DataFrame) -> pd.DataFrame:
    t = muni["total"]
    size = pd.cut(t[START], SIZE_BINS, labels=SIZE_LABELS, right=False)
    g = t[[START, PRE_END, POST_START, END]].groupby(size, observed=True).sum()
    g.insert(0, "obci", t.groupby(size, observed=True).size())
    g.columns = ["obci"] + [f"pop{y}" for y in (START, PRE_END, POST_START, END)]
    _changes(g, lambda y: g[f"pop{y}"], "")
    return g.drop(columns=["reset"])


# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------


def _clean(ax, grid: str = "x") -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _pct_axis(v, _) -> str:
    return f"{v:+.0f} %".replace("-", "−") if v else "0"


def _hbar_cities(ax, c: pd.DataFrame, col: str) -> None:
    """Horizontal bars of a change column, one per city, sorted, red below zero, blue above."""
    c = c.sort_values(col)
    y = np.arange(len(c))
    ax.barh(y, c[col] * 100, color=np.where(c[col] < 0, SHRINK, GROW), height=0.72)
    ax.set_yticks(y, c["name"], fontsize=10.5)
    ax.axvline(0, color=AXIS, lw=1)
    _clean(ax)
    ax.xaxis.set_major_formatter(_pct_axis)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_ylim(-0.6, len(c) - 0.4)
    lo, hi = c[col].min() * 100, c[col].max() * 100
    pad = (hi - lo) * 0.13
    ax.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
    for i, v in enumerate(c[col]):
        off = pad * 0.08 if v >= 0 else -pad * 0.08
        ax.text(v * 100 + off, i, pct(v), va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK2)  # fmt: skip


def social_image(
    path: Path,
    title: tuple[str, str],
    subtitle: list[str],
    draw: Callable,
    axes_rect: tuple[float, float, float, float] = (0.2, 0.17, 0.72, 0.6),
    legend: list | None = None,
    source: str = SOURCE,
) -> None:
    """1080x1350 px (4:5) post image: two-line title, subtitle, one chart, legend, source."""
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    ax = fig.add_axes(axes_rect)
    draw(ax)
    fig.text(0.06, 0.945, title[0], fontsize=27, weight="bold")
    fig.text(0.06, 0.912, title[1], fontsize=27, weight="bold")
    for i, line in enumerate(subtitle):
        fig.text(0.06, 0.872 - 0.022 * i, line, fontsize=13.5, color=INK2)
    if legend:
        fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.2, 0.115), ncol=len(legend),
                   frameon=False, fontsize=10.5)  # fmt: skip
    fig.text(0.06, 0.05, source, fontsize=9.5, color=MUTED, linespacing=1.5)
    fig.savefig(path, dpi=100)
    plt.close(fig)


GROW_SHRINK_LEGEND = [
    Patch(color=SHRINK, label="obyvatel ubylo"),
    Patch(color=GROW, label="obyvatel přibylo"),
]


def chart_a_total(c: pd.DataFrame, nat: pd.DataFrame, path: Path) -> dict:
    n_below = int((c["ch10"] < 0).sum())
    nat_ch = nat.loc[END, "total"] / nat.loc[START, "total"] - 1
    social_image(
        path,
        (f"{n_below} z {len(c)} největších českých měst má", "méně obyvatel než před 10 lety"),
        [
            f"Změna počtu obyvatel {PERIOD}, všechna města nad 40 000 obyvatel.",
            f"Česko za tu dobu vyrostlo o {pct(nat_ch, signed=False, digits=1)}, z velké části díky uprchlíkům z Ukrajiny.",
        ],
        lambda ax: _hbar_cities(ax, c, "ch10"),
        legend=GROW_SHRINK_LEGEND,
    )
    return {"n_below": n_below, "nat_ch": nat_ch}


def chart_b_working_age(c: pd.DataFrame, nat: pd.DataFrame, path: Path) -> dict:
    n_below = int((c["wa_ch10"] < 0).sum())
    nat_ch = nat.loc[END, "15 až 64"] / nat.loc[START, "15 až 64"] - 1
    social_image(
        path,
        (f"Ve {n_below} z {len(c)} největších měst ubylo", "lidí v produktivním věku"),
        [
            f"Změna počtu obyvatel ve věku 15–64 let, {PERIOD}, města nad 40 000 obyvatel.",
            "Právě z této skupiny jsou nájemníci. Celé Česko: "
            f"{pct(nat_ch, digits=1)}, i když celkový počet obyvatel rostl.",
        ],
        lambda ax: _hbar_cities(ax, c, "wa_ch10"),
        legend=GROW_SHRINK_LEGEND,
    )
    return {"n_below": n_below, "nat_ch": nat_ch}


def chart_c_two_periods(c: pd.DataFrame, path: Path) -> dict:
    n_pre = int((c["pre"] < 0).sum())
    n_post = int((c["post"] < 0).sum())

    def draw(ax) -> None:
        d = c.sort_values("pre")
        y = np.arange(len(d))
        ax.barh(y + 0.19, d["pre"] * 100, height=0.36, color=PRE)
        ax.barh(y - 0.19, d["post"] * 100, height=0.36, color=GROW)
        ax.set_yticks(y, d["name"], fontsize=10.5)
        ax.axvline(0, color=AXIS, lw=1)
        _clean(ax)
        ax.xaxis.set_major_formatter(_pct_axis)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_ylim(-0.6, len(d) - 0.4)

    social_image(
        path,
        (
            f"Do roku 2020 se vylidňovalo {n_pre} z {len(c)} měst.",
            "Od roku 2022 roste většina z nich",
        ),
        [
            "Změna počtu obyvatel ve dvou obdobích, která jsou každé spočítané stejnou metodikou.",
            f"Po roce 2022 ubývá už jen {n_post} měst — hlavně kvůli uprchlíkům z Ukrajiny a pracovníkům ze zahraničí.",
        ],
        draw,
        legend=[
            Patch(color=PRE, label="1. 1. 2015 → 1. 1. 2021 (6 let)"),
            Patch(color=GROW, label="1. 1. 2022 → 1. 1. 2025 (3 roky)"),
        ],
    )
    return {"n_pre": n_pre, "n_post": n_post}


def chart_d_sizes(sizes: pd.DataFrame, path: Path) -> dict:
    def draw(ax) -> None:
        v = sizes["ch10"] * 100
        x = np.arange(len(v))
        ax.bar(x, v, color=np.where(v < 0, SHRINK, GROW), width=0.7)
        ax.set_xticks(x, [f"{s}\n({num(n)} obcí)" for s, n in zip(sizes.index, sizes["obci"], strict=True)],
                      fontsize=10)  # fmt: skip
        span = v.max() - min(v.min(), 0)
        for i, val in enumerate(v):
            ax.text(i, val + (span * 0.015 if val >= 0 else -span * 0.015), pct(val / 100),
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=11.5, color=INK2)  # fmt: skip
        ax.axhline(0, color=AXIS, lw=1)
        _clean(ax, "y")
        ax.yaxis.set_major_formatter(_pct_axis)
        ax.set_xlabel("velikost obce (počet obyvatel k 1. 1. 2015)", fontsize=11, labelpad=12)

    social_image(
        path,
        ("Rostou vesnice a velkoměsta.", "Střední města ztrácejí"),
        [
            f"Změna počtu obyvatel {PERIOD} podle velikosti obce na začátku období.",
            "Každá obec zůstává ve své skupině, i když mezitím vyrostla nebo se zmenšila.",
        ],
        draw,
        axes_rect=(0.08, 0.2, 0.86, 0.58),
    )
    return {}


def chart_e_ageing(c: pd.DataFrame, path: Path) -> dict:
    n_up = int((c["s65_2024"] > c["s65_2014"]).sum())
    n_over20 = int((c["s65_2024"] >= 0.2).sum())
    n_over20_2014 = int((c["s65_2014"] >= 0.2).sum())

    def draw(ax) -> None:
        d = c.sort_values("s65_2024")
        y = np.arange(len(d))
        ax.hlines(y, d["s65_2014"] * 100, d["s65_2024"] * 100, color=AXIS, lw=2)
        ax.scatter(d["s65_2014"] * 100, y, color=MUTED, s=40, zorder=3)
        ax.scatter(d["s65_2024"] * 100, y, color=SHRINK, s=55, zorder=3)
        for i, v in enumerate(d["s65_2024"]):
            ax.text(v * 100 + 0.25, i, pct(v, signed=False), va="center", fontsize=9, color=INK2)
        ax.set_yticks(y, d["name"], fontsize=10.5)
        _clean(ax)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f} %")
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_ylim(-0.6, len(d) - 0.4)

    social_image(
        path,
        (f"Podíl seniorů vzrostl v {n_up} z {len(c)}", "největších českých měst"),
        [
            f"Podíl obyvatel ve věku 65+ let, {PERIOD}, města nad 40 000 obyvatel.",
            f"Víc než pětinu seniorů má dnes {n_over20} měst, před 10 lety jen {n_over20_2014}.",
        ],
        draw,
        legend=[
            Line2D([], [], marker="o", lw=0, color=MUTED, label="1. 1. 2015"),
            Line2D([], [], marker="o", lw=0, color=SHRINK, label="1. 1. 2025"),
        ],
    )
    return {"n_up": n_up, "n_over20": n_over20, "n_over20_2014": n_over20_2014}


def chart_national(nat: pd.DataFrame, path: Path) -> None:
    """Methodology chart, not a post: the two series breaks in the national total."""
    s = nat.loc[START:END, "total"] / 1e6
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(s.index, s, color=GROW, lw=2, marker="o", ms=4)
    ax.annotate("2021: sčítání opravilo registr\n(−185 tis. jen na papíře)", (2021, s[2021]),
                xytext=(2015.2, 10.47), fontsize=9, color=INK2,
                arrowprops={"arrowstyle": "-", "color": MUTED})  # fmt: skip
    ax.annotate("2022: uprchlíci z Ukrajiny\nzapočteni mezi obyvatele", (2022, s[2022]),
                xytext=(2015.2, 10.8), fontsize=9, color=INK2,
                arrowprops={"arrowstyle": "-", "color": MUTED})  # fmt: skip
    _clean(ax, "y")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.set_ylabel("obyvatelé (mil.)")
    ax.set_title("Česko, obyvatelé k 31. 12. — dva zlomy v řadě", loc="left", fontsize=14,
                 weight="bold", pad=12)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Label position relative to each city's circle (compass direction), hand-tuned to avoid overlaps;
# cities not listed get "r".
LABEL_POS = {
    "554782": "b",  # Praha
    "532053": "tl",  # Kladno
    "554804": "br",  # Ústí nad Labem
    "567442": "tl",  # Teplice
    "562335": "t",  # Děčín
    "567027": "br",  # Most
    "562971": "l",  # Chomutov
    "554961": "b",  # Karlovy Vary
    "563889": "t",  # Liberec
    "500496": "t",  # Olomouc
    "589250": "l",  # Prostějov
    "505927": "t",  # Opava
    "554821": "l",  # Ostrava
    "555088": "br",  # Havířov
    "598003": "b",  # Frýdek-Místek
    "598917": "tr",  # Karviná
}
DIRECTIONS = {"r": (1, 0), "l": (-1, 0), "t": (0, 1), "b": (0, -1)}
DIRECTIONS |= {a + b: (0.7 * DIRECTIONS[b][0], 0.7 * DIRECTIONS[a][1]) for a in "tb" for b in "rl"}
DIVERGING = LinearSegmentedColormap.from_list("shrink_grow", [SHRINK, "#f0efec", GROW])


def chart_f_map(c: pd.DataFrame, raw: Path, path: Path) -> dict:
    """Variant A drawn on a map: one circle per city, area ~ population, colour ~ change."""
    obce = _rings(raw / "cuzk_ruian" / "obce.json")
    kraje = _rings(raw / "cuzk_ruian" / "kraje.json")
    xy = pd.DataFrame(
        [_centroid(obce[code]) for code in c.index], index=c.index, columns=["x", "y"]
    )
    m = max(abs(c["ch10"].min()), c["ch10"].max()) * 100
    norm = TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)

    def draw(ax) -> None:
        for rings in kraje.values():
            for r in rings:
                ax.fill(r[:, 0], r[:, 1], facecolor="#efeee9", edgecolor=AXIS, lw=0.8)
        d = c.join(xy).sort_values("pop2024", ascending=False)  # big circles below small ones
        size = 40 + d["pop2024"] / 1e6 * 1500  # marker area in pt²
        ax.scatter(d["x"], d["y"], s=size, c=d["ch10"] * 100, cmap=DIVERGING, norm=norm,
                   edgecolors=INK2, linewidths=0.6, zorder=3)  # fmt: skip
        for code, row in d.iterrows():
            ux, uy = DIRECTIONS[LABEL_POS.get(code, "r")]
            pad = np.sqrt(size[code]) / 2 + 3  # circle radius + gap, in points
            ax.annotate(f"{row['name']} {pct(row['ch10'])}", (row["x"], row["y"]),
                        xytext=(ux * pad, uy * pad), textcoords="offset points",
                        ha="left" if ux > 0 else "right" if ux < 0 else "center",
                        va="bottom" if uy > 0 else "top" if uy < 0 else "center",
                        fontsize=9, color=INK, zorder=4)  # fmt: skip
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig = ax.figure
        cax = fig.add_axes((0.2, 0.278, 0.6, 0.014))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax,
                          orientation="horizontal")  # fmt: skip
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=9.5, colors=INK2)
        cb.ax.xaxis.set_major_formatter(_pct_axis)
        fig.text(0.5, 0.305, "změna počtu obyvatel 1. 1. 2015 → 1. 1. 2025", ha="center",
                 fontsize=10.5, color=INK2)  # fmt: skip
        fig.text(0.5, 0.24, "velikost kruhu = počet obyvatel; hranice krajů", ha="center",
                 fontsize=9.5, color=MUTED)  # fmt: skip
        for y, label, df in ((0.175, "Nejvíc ubylo", c.nsmallest(3, "ch10")),
                             (0.15, "Nejvíc přibylo", c.nlargest(3, "ch10"))):  # fmt: skip
            fig.text(0.06, y, f"{label}:", fontsize=12.5, weight="bold")
            fig.text(0.23, y, "   ·   ".join(f"{n} {pct(v)}" for n, v in zip(df["name"], df["ch10"], strict=True)),
                     fontsize=12.5, color=INK2)  # fmt: skip

    k = {"n_below": int((c["ch10"] < 0).sum())}
    social_image(
        path,
        (f"{k['n_below']} z {len(c)} největších českých měst má", "méně obyvatel než před 10 lety"),
        [
            f"Změna počtu obyvatel {PERIOD}, všechna města nad 40 000 obyvatel.",
            "Úbytek se soustřeďuje na severozápadě a severovýchodě, růst kolem Prahy, Plzně a Brna.",
        ],
        draw,
        axes_rect=(0.02, 0.34, 0.96, 0.5),
        source=SOURCE.replace("\npobytem ", " pobytem\n").replace(
            "zatím nebyla zveřejněna.", "zatím nebyla zveřejněna. Hranice: ČÚZK, RÚIAN."
        ),
    )
    return k


# --------------------------------------------------------------------------------------------
# Post texts (Czech)
# --------------------------------------------------------------------------------------------


def post_texts(c: pd.DataFrame, sizes: pd.DataFrame, nat: pd.DataFrame, k: dict) -> dict:
    by = c.set_index("name")
    shrink_by_kraj = c.groupby("kraj")["ch10"].agg(n="size", below=lambda v: int((v < 0).sum()))
    msk, ustk = shrink_by_kraj.loc["Moravskoslezský kraj"], shrink_by_kraj.loc["Ústecký kraj"]
    worst = c.nsmallest(3, "ch10")
    best = c.nlargest(3, "ch10")
    wa_worst = c.nsmallest(3, "wa_ch10")

    def lst(df: pd.DataFrame, col: str) -> str:
        return ", ".join(f"{n} {pct(v)}" for n, v in zip(df["name"], df[col], strict=True))

    young = c.nsmallest(1, "s65_2024").iloc[0]
    old = c.nlargest(1, "s65_2024").iloc[0]
    return {
        "A": (
            f"Česko za posledních 10 let vyrostlo o {pct(k['A']['nat_ch'], signed=False, digits=1)}. Přesto má {k['A']['n_below']} z {len(c)} největších měst méně obyvatel než v roce 2015.\n\n"
            f"Nejvíc ubylo: {lst(worst, 'ch10')}. Nejvíc přibylo: {lst(best, 'ch10')}.\n\n"
            "A to čísla po roce 2022 ještě zahrnují uprchlíky z Ukrajiny — bez nich by byl úbytek ještě výraznější.\n\n"
            "Pokud kupujete byt na pronájem kvůli vysokému výnosu, podívejte se nejdřív, jestli ve městě vůbec přibývají lidé. Vysoký výnos bývá vysoký z nějakého důvodu."
        ),
        "B": (
            f"Nájemníci jsou hlavně lidé v produktivním věku (15–64). A těch za posledních 10 let ubylo ve {k['B']['n_below']} z {len(c)} největších měst.\n\n"
            f"Nejhůř jsou na tom {lst(wa_worst, 'wa_ch10')}. V celém Česku jich ubylo {pct(-k['B']['nat_ch'], signed=False, digits=1)}, přestože celkový počet obyvatel rostl.\n\n"
            f"Přibývá jich jen tam, kde je práce: Plzeň {pct(by.loc['Plzeň', 'wa_ch10'])}, Praha {pct(by.loc['Praha', 'wa_ch10'])}, Jihlava {pct(by.loc['Jihlava', 'wa_ch10'])}, Brno {pct(by.loc['Brno', 'wa_ch10'])}.\n\n"
            "Při výběru města pro investiční byt nesledujte jen počet obyvatel — důležité je, kolik z nich bude reálně hledat nájem."
        ),
        "C": (
            f"Mezi lety 2015 a 2020 se vylidňovalo {k['C']['n_pre']} z {len(c)} největších českých měst. Od roku 2022 ubývá obyvatel už jen v {k['C']['n_post']} z nich.\n\n"
            f"Co se změnilo? Hlavně příchod uprchlíků z Ukrajiny a zahraničních pracovníků. Třeba Karlovy Vary: {pct(by.loc['Karlovy Vary', 'pre'])} do roku 2020, {pct(by.loc['Karlovy Vary', 'post'])} po roce 2022. Mladá Boleslav: {pct(by.loc['Mladá Boleslav', 'pre'])} a pak {pct(by.loc['Mladá Boleslav', 'post'])}.\n\n"
            "Pro pronajímatele je to dobrá i špatná zpráva. Poptávka po nájmech se v mnoha městech oživila, ale stojí na lidech, jejichž pobyt závisí na válce, na dočasné ochraně a na velkých zaměstnavatelích. Počítejte s tím, že se to může zase otočit."
        ),
        "D": (
            f"Kde v Česku za posledních 10 let přibylo lidí? Na obou koncích. Obce do 500 obyvatel vyrostly o {pct(sizes.loc['< 500', 'ch10'], signed=False)}, obce s 500–2 000 obyvateli o {pct(sizes.loc['500–2 tis.', 'ch10'], signed=False)} a pět největších měst o {pct(sizes.loc['100 tis.+', 'ch10'], signed=False)} (hlavně díky Praze).\n\n"
            f"Uprostřed se ztrácí: města s 20–50 tisíci obyvateli {pct(sizes.loc['20–50 tis.', 'ch10'])}, města s 50–100 tisíci {pct(sizes.loc['50–100 tis.', 'ch10'])}.\n\n"
            "Lidé buď míří za prací do velkých měst, nebo se stěhují do vesnic v dojezdové vzdálenosti. Středně velká okresní města ztrácejí obojí. Pro investora je to signál: byt ve středním městě potřebuje silný místní důvod (zaměstnavatel, škola, nemocnice), jinak poptávka spíš slábne."
        ),
        "E": (
            f"Ve {k['E']['n_over20']} z {len(c)} největších českých měst je dnes víc než pětina obyvatel starší 65 let. Před 10 lety to platilo jen pro {k['E']['n_over20_2014']} z nich.\n\n"
            f"Nejstarší je {old['name']} ({pct(old['s65_2024'], signed=False)} seniorů), nejmladší {young['name']} ({pct(young['s65_2024'], signed=False)}).\n\n"
            "Co to znamená pro pronajímatele? Ubývá klasických nájemníků v produktivním věku a přibývá lidí, kteří potřebují menší, bezbariérové byty, výtah a blízko lékaře. Kdo na to připraví svoje byty, může mít v takovém městě výhodu."
        ),
        "F": (
            f"{k['F']['n_below']} z {len(c)} největších českých měst má méně obyvatel než před 10 lety. Na mapě je vidět, že to není náhoda, ale geografie.\n\n"
            f"V Moravskoslezském kraji ztratilo obyvatele {msk['below']} z {msk['n']} velkých měst (Karviná {pct(by.loc['Karviná', 'ch10'])}, Havířov {pct(by.loc['Havířov', 'ch10'])}), v Ústeckém kraji {ustk['below']} z {ustk['n']}. "
            f"Naopak rostou Praha ({pct(by.loc['Praha', 'ch10'])}), Plzeň ({pct(by.loc['Plzeň', 'ch10'])}), Brno ({pct(by.loc['Brno', 'ch10'])}), Jihlava ({pct(by.loc['Jihlava', 'ch10'])}) a Mladá Boleslav ({pct(by.loc['Mladá Boleslav', 'ch10'])}).\n\n"
            "Pro investory do nájemních bytů: levné byty v bývalých průmyslových regionech lákají vysokým výnosem, ale nájemníků tam dlouhodobě ubývá. Rozhoduje, kde se lidé usazují — a to je dnes hlavně osa Plzeň – Praha – Brno."
        ),
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _img(path: Path) -> str:
    return (
        f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="">'
    )


def _table(df: pd.DataFrame, pct_cols=(), int_cols=(), share_cols=(), names=None) -> str:
    d = df.copy()
    for col in pct_cols:
        d[col] = d[col].map(lambda v: "" if pd.isna(v) else pct(v, digits=1))
    for col in share_cols:
        d[col] = d[col].map(lambda v: "" if pd.isna(v) else pct(v, signed=False, digits=1))
    for col in int_cols:
        d[col] = d[col].map(lambda v: "" if pd.isna(v) else num(v))
    if names:
        d = d.rename(columns=names)
    return d.to_html(escape=True, border=0, classes="t")


def _post(text: str) -> str:
    paras = "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n"))
    return f'<div class="post"><div class="post-label">Text příspěvku</div>{paras}</div>'


def write_report(out: Path, ctx: dict) -> None:
    c, sizes, nat, yearly, posts, k, imgs = (
        ctx[x] for x in ("cities", "sizes", "nat", "yearly", "posts", "k", "imgs")
    )
    n = len(c)
    nat_tot = nat["total"]
    rest = c.drop(index=PRAHA)
    rest_gain = rest["pop2024"].sum() - rest["pop2014"].sum()
    chain_below = int((c["chain"] < 0).sum())
    wa_chain_below = int((c["wa_chain"] < 0).sum())
    reset_min, reset_max = c["reset"].min(), c["reset"].max()

    city_view = c[
        ["name", "pop2014", "pop2020", "pop2021", "pop2024", "ch10", "pre", "reset", "post", "chain",
         "wa_ch10", "wa_chain", "s65_2014", "s65_2024"]
    ].iloc[::-1]  # fmt: skip
    sample = yearly[(yearly["level"] == "obec") & (yearly["year"] == END)].head(6)
    esc = html.escape

    body = f"""
<h1>Česká města za posledních 10 let</h1>
<p class="lead">Hlavní závěr: <b>{k["A"]["n_below"]} z {
        n
    } největších českých měst má méně obyvatel než před
10 lety</b> ({PERIOD}), přestože Česko vyrostlo o {
        pct(k["A"]["nat_ch"], signed=False, digits=1)
    }. Ještě výraznější je úbytek lidí
v produktivním věku: <b>ubylo jich ve {k["B"]["n_below"]} z {
        n
    } měst</b>. Mimo Prahu mají velká města dohromady
{
        "o " + tis(rest_gain) + " obyvatel víc"
        if rest_gain >= 0
        else tis(-rest_gain) + " obyvatel méně"
    } než před 10 lety.</p>

<p class="note">Proč ne 2016–2026: ČSÚ zveřejňuje počty obyvatel obcí k 31. 12.; nejnovější jsou k 31. 12. 2024
(vydáno 20. 5. 2025, data k 1. 1. 2026 zatím na webu ČSÚ nejsou). Posledních 10 let, které lze spočítat,
je tedy 31. 12. 2014 → 31. 12. 2024, tj. 1. 1. 2015 → 1. 1. 2025.</p>

<h2>Hlavní obrázek</h2>
<div class="main">{_img(imgs["A"])}</div>
<p>Jako hlavní doporučuji <b>variantu A</b>: je nejsrozumitelnější (konkrétní města, jedno číslo), překvapivá
(polovina velkých měst se zmenšuje, i když země roste) a závěr platí i po očištění o metodický zlom
v roce 2021 (viz varianta C a sekce Předpoklady). Variantu B bych použil jako navazující příspěvek.</p>

<h2>Varianta A — celkový počet obyvatel ve městech</h2>
{_img(imgs["A"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Data:</b> ČSÚ 130149, úroveň obec, obě pohlaví, všechny věkové skupiny dohromady (<code>vek_txt</code> prázdné),
stav k 31. 12. 2014 a 31. 12. 2024.</li>
<li><b>Výběr měst:</b> všechny obce s alespoň {num(CITY_MIN_POP)} obyvateli k 31. 12. 2024 → {
        n
    } měst.
Praha je jedna obec.</li>
<li><b>Výpočet:</b> změna = obyvatelé<sub>2024</sub> / obyvatelé<sub>2014</sub> − 1 pro každé město zvlášť.</li>
<li><b>Graf:</b> vodorovný sloupcový graf, města seřazená od největšího úbytku po největší růst; červeně
města s úbytkem, modře s růstem; popisky zaokrouhlené na celá procenta. Formát 1080 × 1350 px (4:5)
pro sociální sítě.</li>
<li><b>Kontrola:</b> řetězená změna bez roku sčítání (2014→2020 × 2021→2024) dává {chain_below} měst
s úbytkem místo {
        k["A"]["n_below"]
    } — závěr „zhruba polovina velkých měst se zmenšuje“ na zlomu nezávisí.</li>
</ul>
{_post(posts["A"])}

<h2>Varianta B — lidé v produktivním věku (15–64 let)</h2>
{_img(imgs["B"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Data:</b> stejná sada, věková skupina „15 až 64“ (ČSÚ v této sadě nabízí jen skupiny 0–14, 15–64, 65+).</li>
<li><b>Výběr měst:</b> stejných {n} měst jako ve variantě A.</li>
<li><b>Výpočet:</b> změna = obyvatelé 15–64<sub>2024</sub> / obyvatelé 15–64<sub>2014</sub> − 1.</li>
<li><b>Graf:</b> stejný typ jako A, aby šly obrázky postovat jako série.</li>
<li><b>Kontrola:</b> řetězená změna bez roku sčítání dává {
        wa_chain_below
    } měst s úbytkem. Sčítání 2021 ubralo
z registru hlavně lidi v produktivním věku (vystěhovalci, kteří se neodhlásili), proto je tato
kontrola u varianty B důležitější než u A.</li>
<li><b>Pozor:</b> skupina 15–64 obsahuje i studenty a lidi před důchodem; hlavní nájemníci (20–35 let) by vyžadovali
podrobnější věkovou sadu ČSÚ (tip na další analýzu).</li>
</ul>
{_post(posts["B"])}

<h2>Varianta C — dvě období bez metodického zlomu</h2>
{_img(imgs["C"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Proč:</b> řada obsahuje zlom mezi 31. 12. 2020 a 31. 12. 2021 (sčítání opravilo registr, změna měst
{pct(reset_max, digits=1)} až {
        pct(reset_min, digits=1)
    } jen za tento rok). Proto porovnáváme dvě období, každé uvnitř jedné metodiky.</li>
<li><b>Výpočet:</b> období 1 = 31. 12. 2014 → 31. 12. 2020 (6 let, starý registr); období 2 = 31. 12. 2021 → 31. 12. 2024
(3 roky, po sčítání, včetně uprchlíků). Rok 2021 vynecháváme.</li>
<li><b>Graf:</b> dvojice vodorovných sloupců pro každé město (oranžová = 2015–2020, modrá = 2022–2024),
seřazeno podle prvního období. Barvy jsou kategorické (období), ne „dobře/špatně“.</li>
<li><b>Pozor:</b> období mají různou délku (6 vs. 3 roky); graf ukazuje celkovou změnu za období, ne roční tempo.</li>
</ul>
{_post(posts["C"])}

<h2>Varianta D — podle velikosti obce</h2>
{_img(imgs["D"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Data:</b> všechny obce, které existovaly k 31. 12. 2014 i k 31. 12. 2024 ({
        num(sizes["obci"].sum())
    } obcí).</li>
<li><b>Výpočet:</b> obce rozdělené do skupin podle počtu obyvatel <b>na začátku</b> období (2014), aby obec nepřecházela
mezi skupinami; za každou skupinu sečteme obyvatele 2014 a 2024 a spočítáme změnu součtu.</li>
<li><b>Graf:</b> svislý sloupcový graf, skupiny od nejmenších po největší, pod popiskem počet obcí ve skupině.</li>
<li><b>Pozor:</b> skupina 100 tis.+ je dominována Prahou (a Brnem); růst v ní neznamená, že rostou všechna velká města.</li>
</ul>
{_post(posts["D"])}

<h2>Varianta E — stárnutí měst</h2>
{_img(imgs["E"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Data:</b> věková skupina „65 a více“ a celkový počet obyvatel za stejných {n} měst.</li>
<li><b>Výpočet:</b> podíl seniorů = obyvatelé 65+ / všichni obyvatelé, k 31. 12. 2014 a 31. 12. 2024.</li>
<li><b>Graf:</b> „činkový“ graf — šedá tečka 2015, červená 2025, spojnice ukazuje posun; seřazeno podle podílu v roce 2025.</li>
<li><b>Pozor:</b> podíl ovlivňují i uprchlíci po roce 2022 (jsou spíš mladší, podíl seniorů tedy mírně snižují).</li>
</ul>
{_post(posts["E"])}

<h2>Varianta F — mapa (varianta A na mapě)</h2>
{_img(imgs["F"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Data:</b> stejná čísla jako varianta A (změna počtu obyvatel 31. 12. 2014 → 31. 12. 2024, {
        n
    } měst nad
{
        num(CITY_MIN_POP)
    } obyvatel). Hranice obcí a krajů z ČÚZK (RÚIAN, prohlížecí služba ArcGIS, vrstvy 12 a 17),
v souřadnicích S-JTSK (EPSG:5514), zobecněné na 200 m.</li>
<li><b>Poloha města:</b> těžiště největšího polygonu obce (vzorec pro těžiště mnohoúhelníku).</li>
<li><b>Graf:</b> jeden kruh na město, plocha kruhu úměrná počtu obyvatel k 31. 12. 2024; barva podle změny —
divergentní škála červená (úbytek) → světle šedá (0) → modrá (růst), symetrická kolem nuly
(±{
        pct(max(abs(c["ch10"].min()), c["ch10"].max()), signed=False)
    }). Podklad: hranice krajů. Popisek = název + změna;
polohy popisků ručně posunuté, aby se nepřekrývaly.</li>
<li><b>Proti variantě A:</b> mapa ukazuje geografický vzor (úbytek na severu Čech a v Moravskoslezském kraji),
ale hůř se z ní čte pořadí a přesné hodnoty. Hodí se jako druhý obrázek ke stejnému příspěvku nebo samostatně.</li>
<li><b>Kraj</b> města (do textu příspěvku) je z číselníku v datové sadě ČSÚ 170240.</li>
</ul>
{_post(posts["F"])}

<h2>Proč by to mělo zajímat pronajímatele</h2>
<ul>
<li><b>Počet obyvatel je nejpomalejší, ale nejspolehlivější tahoun poptávky.</b> Města, kde za 10 let ubylo
6–13 % lidí a až 17 % lidí v produktivním věku (Karviná, Havířov, Přerov, Děčín, Frýdek-Místek), mají
zmenšující se okruh nájemníků. Vysoké hrubé výnosy tam částečně platí za toto riziko.</li>
<li><b>Krajské město ≠ rostoucí město.</b> Ostrava i Ústí nad Labem za 10 let lidi ztratily, a to i se započtením uprchlíků a i po očištění o zlom v roce 2021.</li>
<li><b>Růst po roce 2022 není organický.</b> Velká část nedávného růstu (Plzeň, Mladá Boleslav, Karlovy Vary,
Praha) jsou uprchlíci a zahraniční pracovníci. Jsou to skuteční nájemníci, ale poptávka závisí na
politice (dočasná ochrana) a na velkých zaměstnavatelích.</li>
<li><b>Stárnutí mění typ poptávky.</b> Víc seniorů = poptávka po menších, bezbariérových bytech s výtahem.</li>
</ul>

<h2>Metodický zlom v datech</h2>
{_img(imgs["N"])}
<p>Roky 2000–2020 jsou počítány z registru obyvatel, který <b>nebyl zpětně opraven</b> podle sčítání 2021 (registr
nadhodnocoval počet obyvatel, hlavně o cizince, kteří odešli bez odhlášení). Od 31. 12. 2021 navazují čísla na sčítání.
V roce 2022 přibyli uprchlíci z Ukrajiny s dočasnou ochranou (~{
        tis(nat_tot[2022] - nat_tot[2021])
    } skok v celém Česku).
Proto ukazujeme celkovou změnu (A, B, D, E), ale ověřujeme ji řetězenou změnou bez roku 2021 a ve variantě C obě období zvlášť.</p>

<h2>Zdroje dat</h2>
<table class="t">
<tr><th>Datová sada</th><th>Obsah</th><th>URL</th><th>Velikost</th></tr>
<tr><td>ČSÚ 130149-25 (DEM05)</td><td>Obyvatelé s trvalým nebo dlouhodobým pobytem k 31. 12. 2000–2024
podle pohlaví a věkových skupin (0–14, 15–64, 65+) za stát, kraje, okresy, ORP a všech {
        yearly.query("level=='obec'").code.nunique():,} obcí. Vydáno 20. 5. 2025.</td>
<td><a href="{URL_130149}">{URL_130149}</a></td><td>{
        num(len(yearly))
    } řádků (po výběru obou pohlaví)</td></tr>
<tr><td>ČSÚ 170240-17 (SLDB_HD)</td><td>Obyvatelé a domy v obcích podle sčítání; zde použit jen číselník
obec → kraj.</td><td><a href="{URL_170240}">{URL_170240}</a></td><td>—</td></tr>
<tr><td>ČÚZK RÚIAN</td><td>Hranice obcí a krajů (platné k datu stažení), S-JTSK, zobecněno na 200 m; jen pro mapu (F).</td>
<td><a href="{URL_RUIAN}">{URL_RUIAN}</a></td><td>6 258 obcí, 14 krajů</td></tr>
</table>
<p>Surová data: <code>{esc(str(ctx["raw"]))}</code></p>
<h3>Ukázka dat (zpracováno)</h3>{sample.to_html(border=0, classes="t", index=False)}
<h3>Základní statistiky</h3>
<ul>
<li>Česko: {num(nat_tot[START])} (31. 12. 2014) → {num(nat_tot[PRE_END])} (2020) → {
        num(nat_tot[POST_START])
    } (2021) → {num(nat_tot[END])} (2024).</li>
<li>Lidé 15–64 let v Česku: {num(nat.loc[START, "15 až 64"])} → {
        num(nat.loc[END, "15 až 64"])
    }; 65+: {num(nat.loc[START, "65 a více"])} → {num(nat.loc[END, "65 a více"])}.</li>
<li>Obce existující v obou letech: {num(sizes["obci"].sum())}; měst nad 40 tis.: {n}, žije v nich
{pct(c["pop2024"].sum() / nat_tot[END], signed=False)} obyvatel (2014: {
        pct(c["pop2014"].sum() / nat_tot[START], signed=False)
    }).</li>
</ul>

<h2>Předpoklady</h2>
<ul>
<li><b>Město</b> = obec s alespoň {
        num(CITY_MIN_POP)
    } obyvateli k 31. 12. 2024. Praha je jedna obec.</li>
<li><b>Obyvatelé</b> = osoby s trvalým nebo dlouhodobým pobytem včetně cizinců (definice ČSÚ), od 2022 včetně
uprchlíků s dočasnou ochranou.</li>
<li>Hranice obcí: obce se za 10 let neslučovaly; nově vzniklé obce (několik malých) jsou ve variantě D vynechány.</li>
<li>Stav k 31. 12. roku X = stav k 1. 1. roku X+1; v textech pro veřejnost používáme 1. 1.</li>
</ul>

<h2>Výpočty</h2>
<ul>
<li>Změna = hodnota<sub>konec</sub> / hodnota<sub>začátek</sub> − 1 (za město nebo za součet skupiny).</li>
<li>Řetězená změna = (1 + změna 2014→2020) × (1 + změna 2021→2024) − 1 — ignoruje rok 2021 se zlomem
(a tedy i skutečný vývoj v tom roce).</li>
<li>Podíl seniorů = obyvatelé 65+ / všichni obyvatelé.</li>
</ul>

<h2>Postup krok za krokem</h2>
<ol>
<li>Ověřil jsem, že nejnovější data za obce jsou k 31. 12. 2024 (stránka ČSÚ, vydání 20. 5. 2025) → okno 2014–2024.</li>
<li>Použil jsem dříve stažená surová data (<code>download.py</code>), z nich obě pohlaví, všechny úrovně a věkové skupiny.</li>
<li>Přeskupil jsem data na obec × (věk, rok); ponechal obce existující v obou krajních letech.</li>
<li>Spočítal jsem změny za 10 let, za obě období a řetězenou změnu, pro všechny obyvatele a pro 15–64 let.</li>
<li>Zkontroloval jsem velikost zlomu 2020→2021 ve městech ({pct(reset_max, digits=1)} až {
        pct(reset_min, digits=1)
    }) a ověřil, že
počet zmenšujících se měst po jeho odstranění zůstává podobný ({k["A"]["n_below"]} vs. {
        chain_below
    }).</li>
<li>Vytvořil jsem šest variant grafu (A–F, F je mapa varianty A) ve formátu pro sociální sítě a ke každé text příspěvku; jako hlavní zvolil A.</li>
</ol>

<h2>Města — tabulka</h2>
{
        _table(
            city_view,
            pct_cols=["ch10", "pre", "reset", "post", "chain", "wa_ch10", "wa_chain"],
            int_cols=["pop2014", "pop2020", "pop2021", "pop2024"],
            share_cols=["s65_2014", "s65_2024"],
            names={
                "name": "město",
                "pop2014": "2014",
                "pop2020": "2020",
                "pop2021": "2021",
                "pop2024": "2024",
                "ch10": "změna 10 let",
                "pre": "2014→20",
                "reset": "2020→21 (zlom)",
                "post": "2021→24",
                "chain": "řetězená",
                "wa_ch10": "15–64: 10 let",
                "wa_chain": "15–64: řetězená",
                "s65_2014": "65+ 2014",
                "s65_2024": "65+ 2024",
            },
        )
    }
<p class="note">Hodnoty k 31. 12. daného roku.</p>

<h2>Podle velikosti obce — tabulka</h2>
{
        _table(
            sizes,
            pct_cols=["ch10", "pre", "post", "chain"],
            int_cols=["obci", "pop2014", "pop2020", "pop2021", "pop2024"],
            names={
                "ch10": "změna 10 let",
                "pre": "2014→20",
                "post": "2021→24",
                "chain": "řetězená",
                "obci": "obcí",
            },
        )
    }

<h2>Moje poznámky</h2>
<ul>
<li><b>Je to důvěryhodné?</b> Směr a řád ano: oficiální data ČSÚ, stejné hranice obcí a závěr drží jak v prosté
změně, tak v řetězené změně bez zlomu. Přesná procenta mají šum ±1–2 p. b. kvůli zlomu v roce 2021.</li>
<li><b>Lidé ≠ domácnosti.</b> Průměrná domácnost se zmenšuje, takže počet domácností (= poptávka po bytech)
klesá pomaleji než počet lidí, nebo dokonce roste. Příspěvky proto říkají „méně lidí“, ne „méně bytů“.</li>
<li><b>Suburbanizace.</b> Část úbytku ve městech jsou lidé, kteří se odstěhovali hned za hranici města (viz D).
Pro majitele bytu ve městě se okruh nájemníků zmenšil, ekonomika regionu ale může být v pořádku (Brno, Plzeň).</li>
<li><b>Krátkodobý pohled je zkreslený uprchlíky.</b> Za 10 let je trend jasný; poslední 3 roky by sami o sobě
ukazovaly mnohem optimističtější obrázek (C).</li>
<li><b>Další kroky:</b> (1) věk 20–34 let po městech z podrobné věkové sady ČSÚ; (2) spojit s cenami nájmů a bytů za m²
a ukázat výnos vs. vývoj populace; (3) aktualizovat, až ČSÚ zveřejní data k 1. 1. 2026.</li>
</ul>
"""
    style = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;--post:#f2f1ec}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5;--post:#242422}}
body{background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
max-width:960px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:30px;margin-bottom:4px}h2{margin-top:48px;border-bottom:1px solid var(--line);padding-bottom:6px}
.lead{font-size:18px;color:var(--ink2)} a{color:var(--accent)}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fcfcfb}
h2+img{max-width:540px;display:block}.main img{max-width:540px}
.post{background:var(--post);border-left:4px solid var(--accent);border-radius:6px;padding:12px 18px;margin:16px 0}
.post-label{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
table.t{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}
.t th,.t td{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.t th{color:var(--ink2)} .note{color:var(--muted);font-size:13px}
code{font-size:13px}
"""
    (out / "report.html").write_text(
        f'<!doctype html><html lang="cs"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Města za 10 let</title><style>{style}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------------


def main() -> None:
    out = monaco.output_dir(ANALYSIS)
    raw = latest_raw()
    yearly = load_yearly(raw)
    muni = municipalities(yearly)
    nat = national(yearly)
    cities = city_table(muni, load_regions(raw))
    sizes = size_class_table(muni)

    (out / "data").mkdir()
    cities.to_csv(out / "data" / "mesta.csv")
    sizes.to_csv(out / "data" / "velikost_obci.csv")
    nat.to_csv(out / "data" / "cesko.csv")

    imgs = {
        "A": out / "A_mesta_obyvatele_10_let.png",
        "B": out / "B_mesta_produktivni_vek.png",
        "C": out / "C_dve_obdobi.png",
        "D": out / "D_velikost_obci.png",
        "E": out / "E_podil_senioru.png",
        "F": out / "F_mapa_mesta_10_let.png",
        "N": out / "metodika_cesko_zlomy.png",
    }
    k = {
        "A": chart_a_total(cities, nat, imgs["A"]),
        "B": chart_b_working_age(cities, nat, imgs["B"]),
        "C": chart_c_two_periods(cities, imgs["C"]),
        "D": chart_d_sizes(sizes, imgs["D"]),
        "E": chart_e_ageing(cities, imgs["E"]),
        "F": chart_f_map(cities, raw, imgs["F"]),
    }
    chart_national(nat, imgs["N"])
    posts = post_texts(cities, sizes, nat, k)
    (out / "texty_prispevku.md").write_text(
        "\n\n".join(f"## Varianta {v}\n\n{t}" for v, t in posts.items()) + "\n", encoding="utf-8"
    )
    write_report(
        out,
        {
            "cities": cities,
            "sizes": sizes,
            "nat": nat,
            "yearly": yearly,
            "posts": posts,
            "k": k,
            "imgs": imgs,
            "raw": raw,
        },  # fmt: skip
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
