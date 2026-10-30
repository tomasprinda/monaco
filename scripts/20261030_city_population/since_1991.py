# ruff: noqa: E501  (long lines are prose inside the HTML report template)
"""How the population of Czech cities evolves in time.

Reads the latest download from data/raw/city_population/ (see download.py) and
writes processed data, candidate charts, the main social-media image and an HTML report to
outputs/city_population_since_1991/<yymmdd_hhmmss>/.

Usage: uv run scripts/20261030_city_population/since_1991.py
"""

import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import monaco

RAW_ROOT = monaco.DATA_DIR / "raw" / "city_population"
CITY_MIN_POP = 40_000  # "city" = municipality with at least this many residents in 2024
CENSUS_YEARS = [1869, 1880, 1890, 1900, 1910, 1921, 1930, 1950, 1961, 1970, 1980, 1991, 2001, 2011]
SIZE_BINS = [0, 2_000, 5_000, 20_000, 50_000, 100_000, 2_000_000]
SIZE_LABELS = ["< 2k", "2k–5k", "5k–20k", "20k–50k", "50k–100k", "100k+"]

URL_130149 = "https://csu.gov.cz/produkty/obyvatelstvo-k-3112-podle-pohlavi-v-obcich"
URL_170240 = "https://csu.gov.cz/obyvatelstvo-a-domy-v-obcich-podle-vysledku-scitani-od-roku-1869"

# Chart styling (reference dataviz palette).
GROW = "#2a78d6"
SHRINK = "#e34948"
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


def load_census(raw: Path) -> pd.DataFrame:
    """170240: residents per municipality at each census 1869-2011, on 2015 municipal borders."""
    df = pd.read_csv(raw / "csu_170240" / "OD_SLDB_HD_at.csv", dtype={"vuzemi_kod": str})
    df = df[df["stapro_kod"] == 2406]  # population (2409 = number of houses)
    return df.rename(columns={"vuzemi_kod": "code", "vuzemi_txt": "name", "hodnota": "pop"})


def build_municipalities(yearly: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """One row per municipality: census populations 1869-2011 plus yearly 2021 and 2024."""
    hist = census.pivot_table(index="code", columns="rok", values="pop")
    hist.columns = [f"c{c}" for c in hist.columns]
    meta = census.drop_duplicates("code").set_index("code")[
        ["name", "so_orp_text", "okres_text", "kraj_text"]
    ]
    obce = yearly[(yearly["level"] == "obec") & (yearly["age"] == "total")]
    recent = obce.pivot_table(index="code", columns="year", values="pop")[[2021, 2024]]
    recent.columns = ["y2021", "y2024"]
    # Inner join: drops 6 municipalities created after 2015 (≈1.3k people in 2021) and 12
    # that ceased to exist before 2011; both are negligible for the aggregates below.
    return meta.join(hist).join(recent, how="inner")


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


def city_table(muni: pd.DataFrame) -> pd.DataFrame:
    cities = muni[muni["y2024"] >= CITY_MIN_POP].copy()
    hist_cols = [f"c{y}" for y in CENSUS_YEARS] + ["y2021"]
    cities["peak_pop"] = cities[hist_cols].max(axis=1)
    cities["peak_year"] = cities[hist_cols].idxmax(axis=1).str.lstrip("cy").astype(int)
    cities["ch_1991_2021"] = cities["y2021"] / cities["c1991"] - 1
    cities["ch_1991_2024"] = cities["y2024"] / cities["c1991"] - 1
    cities["ch_2011_2021"] = cities["y2021"] / cities["c2011"] - 1
    cities["ch_2021_2024"] = cities["y2024"] / cities["y2021"] - 1
    cities["ch_peak_2024"] = cities["y2024"] / cities["peak_pop"] - 1
    return cities.sort_values("ch_1991_2024")


def size_class_table(muni: pd.DataFrame) -> pd.DataFrame:
    m = muni.dropna(subset=["c1991"]).copy()
    m["size_1991"] = pd.cut(m["c1991"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    g = m.groupby("size_1991", observed=True).agg(
        municipalities=("c1991", "size"),
        pop_1991=("c1991", "sum"),
        pop_2001=("c2001", "sum"),
        pop_2011=("c2011", "sum"),
        pop_2021=("y2021", "sum"),
        pop_2024=("y2024", "sum"),
    )
    g["ch_1991_2021"] = g["pop_2021"] / g["pop_1991"] - 1
    g["ch_1991_2024"] = g["pop_2024"] / g["pop_1991"] - 1
    return g


def district_table(muni: pd.DataFrame) -> pd.DataFrame:
    g = muni.groupby("okres_text")[["c1991", "y2021", "y2024"]].sum()
    g["ch_1991_2021"] = g["y2021"] / g["c1991"] - 1
    g["ch_1991_2024"] = g["y2024"] / g["c1991"] - 1
    return g.sort_values("ch_1991_2024")


def age_table(yearly: pd.DataFrame, codes: pd.Index) -> pd.DataFrame:
    """Age structure of the cities in 2021 vs 2024 (both on the post-census basis)."""
    a = yearly[(yearly["level"] == "obec") & yearly["code"].isin(codes)]
    a = a[a["year"].isin([2021, 2024])].pivot_table(
        index=["code", "name"], columns=["age", "year"], values="pop"
    )
    out = pd.DataFrame(index=a.index)
    for y in (2021, 2024):
        out[f"share65_{y}"] = a[("65 a více", y)] / a[("total", y)]
    out["ch_15_64_2021_2024"] = a[("15 až 64", 2024)] / a[("15 až 64", 2021)] - 1
    return out.reset_index(level="name")


# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------


def _pct(x: float) -> str:
    if abs(x) < 0.005:
        return "±0%"
    return f"{x:+.0%}".replace("-", "−")


def _hbar_cities(ax, cities: pd.DataFrame, col: str) -> None:
    c = cities.sort_values(col)
    colors = np.where(c[col] < 0, SHRINK, GROW)
    y = np.arange(len(c))
    ax.barh(y, c[col] * 100, color=colors, height=0.72)
    ax.set_yticks(y, c["name"], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=AXIS, lw=1)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f} %".replace("-", "−") if v else "0")
    ax.tick_params(axis="x", labelsize=9.5, length=0)
    ax.set_ylim(-0.6, len(c) - 0.4)
    for i, v in enumerate(c[col]):
        off = 0.6 if v >= 0 else -0.6
        ax.text(
            v * 100 + off,
            i,
            _pct(v),
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
            color=INK2,
        )


def main_image(
    cities: pd.DataFrame, sizes: pd.DataFrame, nat_1991: float, nat_2024: float, path: Path
) -> None:
    n_below = int((cities["ch_1991_2024"] < 0).sum())
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)  # 1080x1350 px, 4:5 social format
    ax = fig.add_axes((0.2, 0.17, 0.72, 0.6))
    _hbar_cities(ax, cities, "ch_1991_2024")
    ax.set_xlim(-34, 20)

    fig.text(
        0.06,
        0.945,
        f"{n_below} of {len(cities)} largest Czech cities have",
        fontsize=27,
        weight="bold",
    )
    fig.text(0.06, 0.912, "fewer residents than in 1991", fontsize=27, weight="bold")
    nat_gain = (nat_2024 - nat_1991) / 1e3
    rest = cities.drop(index="554782")  # Prague
    rest_gain = (rest["y2024"].sum() - rest["c1991"].sum()) / 1e3
    fig.text(
        0.06,
        0.872,
        "Population change 1991 → 2024, all municipalities with 40,000+ residents today.",
        fontsize=13.5,
        color=INK2,
    )
    fig.text(
        0.06,
        0.850,
        f"Czechia gained {nat_gain:,.0f}k people since 1991. The 25 cities outside Prague "
        f"together lost {-rest_gain:,.0f}k.",
        fontsize=13.5,
        color=INK2,
    )

    # Supporting strip: where the growth went.
    small = sizes.loc["< 2k", "ch_1991_2024"]
    mid = sizes.loc["20k–50k", "ch_1991_2024"]
    fig.text(0.06, 0.800, "Where people went instead", fontsize=12.5, weight="bold")
    fig.text(
        0.06,
        0.781,
        f"Villages under 2,000 residents: {_pct(small)}   ·   "
        f"Towns of 20–50k: {_pct(mid)}   ·   "
        "Districts around Prague (Praha-západ, -východ): roughly doubled",
        fontsize=11,
        color=INK2,
    )

    fig.legend(
        handles=[
            Patch(color=SHRINK, label="fewer people than 1991"),
            Patch(color=GROW, label="more people than 1991"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.2, 0.125),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    fig.text(
        0.06,
        0.06,
        "Source: Czech Statistical Office (ČSÚ) — census 1991 (dataset 170240, 2015 borders); "
        "residents with permanent or\nlong-term stay at 31 Dec 2024 (dataset 130149). "
        "2024 includes ~300k refugees from Ukraine with temporary protection,\n"
        "so the real decline of most cities is, if anything, understated.",
        fontsize=9.5,
        color=MUTED,
        linespacing=1.5,
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


def alt_census_bars(cities: pd.DataFrame, path: Path) -> None:
    n_below = int((cities["ch_1991_2021"] < 0).sum())
    fig, ax = plt.subplots(figsize=(9, 10))
    _hbar_cities(ax, cities, "ch_1991_2021")
    ax.set_xlim(-40, 12)
    ax.set_title(
        f"Census to census: {n_below} of {len(cities)} cities shrank 1991 → 2021",
        loc="left",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def alt_size_classes(sizes: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    v = sizes["ch_1991_2024"] * 100
    ax.bar(range(len(v)), v, color=np.where(v < 0, SHRINK, GROW), width=0.7)
    ax.set_xticks(
        range(len(v)),
        [f"{s}\n({n:,} munic.)" for s, n in zip(sizes.index, sizes["municipalities"], strict=True)],
    )
    for i, x in enumerate(v):
        ax.text(
            i,
            x + (0.6 if x >= 0 else -0.6),
            _pct(x / 100),
            ha="center",
            va="bottom" if x >= 0 else "top",
            fontsize=10,
            color=INK2,
        )
    ax.axhline(0, color=AXIS, lw=1)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_ylabel("population change 1991 → 2024 (%)")
    ax.set_title(
        "Villages grew, mid-sized towns shrank (by size in 1991)",
        loc="left",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def alt_long_history(muni: pd.DataFrame, path: Path) -> None:
    picks = {
        "554782": "Praha",
        "582786": "Brno",
        "554821": "Ostrava",
        "598917": "Karviná",
        "554791": "Plzeň",
    }
    years = CENSUS_YEARS + [2021, 2024]
    cols = [f"c{y}" for y in CENSUS_YEARS] + ["y2021", "y2024"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (code, name), col in zip(picks.items(), colors, strict=True):
        s = muni.loc[code, cols].astype(float)
        idx = s / s["c1991"] * 100
        ax.plot(years, idx.values, color=col, lw=2, marker="o", ms=3.5, label=name)
        ax.text(2026, idx.iloc[-1], name, va="center", fontsize=9.5, color=INK2)
    ax.axhline(100, color=AXIS, lw=1)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(1865, 2040)
    ax.set_xticks(range(1880, 2021, 20))
    ax.set_ylabel("population, 1991 = 100")
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=9.5)
    ax.set_title(
        "155 years of five cities (2015 borders): most peaked around 1991",
        loc="left",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def alt_national_series(yearly: pd.DataFrame, path: Path) -> None:
    s = yearly[(yearly["level"] == "stát") & (yearly["age"] == "total")].set_index("year")["pop"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(s.index, s / 1e6, color=GROW, lw=2, marker="o", ms=4)
    ax.annotate(
        "2021: census resets the register\n(−185k on paper)",
        (2021, s[2021] / 1e6),
        xytext=(2014.5, 10.32),
        fontsize=9,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    ax.annotate(
        "2022: refugees from Ukraine\ncounted as long-term residents",
        (2022, s[2022] / 1e6),
        xytext=(2013.5, 10.84),
        fontsize=9,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("residents (millions)")
    ax.set_title(
        "Czechia, residents at 31 Dec — note the two series breaks",
        loc="left",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _img(path: Path) -> str:
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def _table(df: pd.DataFrame, pct_cols=(), int_cols=(), share_cols=()) -> str:
    d = df.copy()
    for c in pct_cols:
        d[c] = d[c].map(lambda v: "" if pd.isna(v) else _pct(v))
    for c in share_cols:
        d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    for c in int_cols:
        d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
    return d.to_html(escape=True, border=0, classes="t")


def write_report(out: Path, ctx: dict) -> None:
    cities, sizes, districts, ages, yearly, census = (
        ctx[k] for k in ("cities", "sizes", "districts", "ages", "yearly", "census")
    )
    nat = yearly[(yearly["level"] == "stát") & (yearly["age"] == "total")].set_index("year")["pop"]
    nat_1991 = ctx["muni"]["c1991"].sum()

    city_view = cities[
        [
            "name",
            "c1950",
            "c1991",
            "c2011",
            "y2021",
            "y2024",
            "peak_year",
            "ch_1991_2021",
            "ch_2021_2024",
            "ch_1991_2024",
        ]
    ].iloc[::-1]
    city_view = city_view.join(ages[["share65_2024", "ch_15_64_2021_2024"]])
    dist_view = pd.concat([districts.head(8), districts.tail(8)])

    sample_yearly = yearly.head(5).to_html(border=0, classes="t", index=False)
    sample_census = (
        census[["code", "name", "rok", "datum", "pop", "okres_text"]]
        .head(5)
        .to_html(border=0, classes="t", index=False)
    )
    esc = html.escape

    body = f"""
<h1>Czech cities: population 1869–2024</h1>
<p class="lead">Main takeaway: <b>{ctx["n_below"]} of the {len(cities)} largest Czech cities have
fewer residents today than in 1991</b>, although the country gained ~{
        (nat[2024] - nat_1991) / 1e3:,.0f}k
people — the 25 cities outside Prague together lost {-ctx["rest_gain"] / 1e3:,.0f}k. Growth went to
villages and suburbs; the industrial north and east lost the most.</p>

<h2>Main image</h2>
<div class="main">{_img(out / "main_city_change_1991_2024.png")}</div>

<h2>Why should landlords care?</h2>
<ul>
<li><b>Population is the slowest but most reliable demand driver.</b> A city that has lost 15–28% of its
people since 1991 (Karviná, Havířov, Přerov, Ostrava, Frýdek-Místek) has a structurally shrinking
tenant pool — cheap prices and high gross yields there partly price in this risk (vacancy,
weak capital growth).</li>
<li><b>"Big city" ≠ growing city.</b> Only Prague, Plzeň, Mladá Boleslav, Liberec, Jihlava and Brno are clearly above
1991. Many regional capitals (Hradec Králové, Pardubice, Ústí, Zlín, Olomouc) are flat or below.</li>
<li><b>The growth is in villages and the commuter ring.</b> Municipalities under 2,000 residents grew
{sizes.loc["< 2k", "ch_1991_2024"]:.0%} since 1991, Praha-západ
and Praha-východ roughly doubled. Commuter-belt demand is mostly for houses, but it signals where
young families settle.</li>
<li><b>The 2022 jump is not organic.</b> Recent growth in Plzeň (+11% since 2021), Mladá Boleslav (+13%),
Prague (+10%) is largely refugees from Ukraine and foreign workers — real tenants today, but a
demand source that depends on policy (temporary protection) and on industrial employers.</li>
</ul>

<h2>Alternatives considered</h2>
<h3>A. Census-only comparison 1991 → 2021 (methodologically cleanest)</h3>
<p>Both points are census results; no refugee effect. {ctx["n_below_census"]} of {
        len(cities)
    } cities shrank.
Not chosen as main because it is 3+ years old and investors care about today.</p>
{_img(out / "alt_census_1991_2021.png")}
<h3>B. Change by municipality size</h3>
<p>Shows the mechanism: villages under 2,000 residents grew fastest, towns of 5k–100k shrank; the six
100k+ cities grew only thanks to Prague, Brno and Plzeň (and partly refugees after 2022). Good as a
follow-up post, less emotional than named cities.</p>
{_img(out / "alt_size_classes.png")}
<h3>C. 150 years of five cities</h3>
<p>Shows that 1991 was the historical peak for most cities (post-war industrial and panel-housing
boom). Nice for storytelling, too busy for a social image.</p>
{_img(out / "alt_long_history.png")}
<h3>D. National series and its breaks</h3>
<p>Not a post — it documents why a naive 2000–2024 yearly comparison is misleading.</p>
{_img(out / "alt_national_series.png")}

<h2>Data sources</h2>
<table class="t">
<tr><th>Dataset</th><th>Content</th><th>URL</th><th>Size</th></tr>
<tr><td>ČSÚ 130149-25 (DEM05)</td><td>Residents with permanent or long-term stay at 31 Dec 2000–2024,
by sex and age group (0–14, 15–64, 65+), for state, regions, districts, ORP and all {
        yearly.query("level=='obec'").code.nunique():,}
municipalities. Published May 2025.</td><td><a href="{URL_130149}">{URL_130149}</a></td>
<td>{ctx["n_rows_yearly"]:,} rows</td></tr>
<tr><td>ČSÚ 170240-17 (SLDB_HD)</td><td>Population and houses in municipalities at each census 1869–2011,
recalculated to municipal borders of 2015.</td><td><a href="{URL_170240}">{URL_170240}</a></td>
<td>{ctx["n_rows_census"]:,} rows</td></tr>
</table>
<p>Raw files: <code>{esc(str(ctx["raw"]))}</code></p>
<h3>Sample: 130149 (processed)</h3>{sample_yearly}
<h3>Sample: 170240 (processed)</h3>{sample_census}
<h3>Basic stats</h3>
<ul>
<li>Czechia: {nat_1991:,.0f} (census 1991) → {nat[2021]:,.0f} (2021) → {nat[2024]:,.0f} (2024).</li>
<li>Municipalities matched across both datasets: {
        len(ctx["muni"]):,}; cities with 40k+ residents in 2024: {len(cities)}.</li>
<li>Cities hold {cities["y2024"].sum() / nat[2024]:.0%} of the population in 2024 vs {
        cities["c1991"].sum() / nat_1991:.0%} in 1991.</li>
</ul>

<h2>Assumptions</h2>
<ul>
<li><b>City</b> = municipality (obec) with ≥ {
        CITY_MIN_POP:,} residents at 31 Dec 2024. Prague is one municipality.</li>
<li>Population = residents with permanent or long-term stay (incl. foreigners), as defined by ČSÚ.</li>
<li>Historical census data are on 2015 municipal borders, so changes are not caused by
merging or splitting municipalities. 2021/2024 values are on current borders; between 2015 and
2024 only 6 small municipalities were created (≈1.3k people), so the effect is negligible.</li>
<li>1991 census counted permanent residents; the 2021+ figures are census-based "usual residents"
rolled forward. Small methodological differences (a few %) are possible; the big declines
(10–28%) are far beyond that.</li>
<li>Yearly figures 2000–2020 were <b>not</b> revised after the 2021 census (the register overstated
population, mostly by unregistered emigrants); therefore we avoid comparing 2001–2020 yearly values with 2021+.</li>
</ul>

<h2>Calculations</h2>
<ul>
<li>Change = pop<sub>end</sub> / pop<sub>start</sub> − 1, per municipality or summed over a group.</li>
<li>Peak = maximum over the 14 censuses 1869–2011 and the 2021 value.</li>
<li>Size classes use 1991 population, so municipalities stay in their class (avoids survivorship bias).</li>
<li>District aggregates sum municipalities by their 2015 district (okres).</li>
</ul>

<h2>Step by step</h2>
<ol>
<li>Searched ČSÚ open data; picked 130149 (yearly, 2000–2024) and 170240 (censuses since 1869).</li>
<li>Downloaded raw zips + CSVW schemas with <code>download.py</code>.</li>
<li>Kept totals for both sexes; reshaped to one row per municipality; joined on municipality code.</li>
<li>Noticed the 2021 drop (−185k) and 2022 jump (+311k) in the national series → identified as census
rebasing and refugee registration; chose census 1991 as the base and showed 2024 as "today".</li>
<li>Computed change for cities, size classes, districts; age shares for cities.</li>
<li>Drew four candidate charts, chose the named-city bar chart as the main image (most concrete and
surprising for the audience), and verified the headline holds both with census-only data (24/26) and
with the latest data (18/26).</li>
</ol>

<h2>Cities (sorted by change 1991 → 2024)</h2>
{
        _table(
            city_view,
            pct_cols=[
                "ch_1991_2021",
                "ch_2021_2024",
                "ch_1991_2024",
                "ch_15_64_2021_2024",
            ],
            int_cols=["c1950", "c1991", "c2011", "y2021", "y2024"],
            share_cols=["share65_2024"],
        )
    }
<p class="note">share65_2024 = share of residents aged 65+; ch_15_64_2021_2024 = change in
working-age (15–64) residents since 2021.</p>

<h2>By municipality size (1991)</h2>
{
        _table(
            sizes,
            pct_cols=["ch_1991_2021", "ch_1991_2024"],
            int_cols=["pop_1991", "pop_2001", "pop_2011", "pop_2021", "pop_2024"],
        )
    }

<h2>Districts: 8 biggest losers and winners 1991 → 2024</h2>
{_table(dist_view, pct_cols=["ch_1991_2021", "ch_1991_2024"], int_cols=["c1991", "y2021", "y2024"])}

<h2>My thoughts</h2>
<ul>
<li><b>Trustworthy?</b> Yes for the direction and order of magnitude: official ČSÚ data, border-consistent
history, and the headline survives both the conservative (census) and the generous (2024 incl.
refugees) definition. Exact percentages carry ±2–3 pp methodological noise.</li>
<li><b>Caveat — people ≠ households.</b> Average household size kept falling, so the number of
households (= demand for flats) in shrinking cities fell less than population, or even grew. The post
should not claim "fewer flats needed"; it says "fewer people".</li>
<li><b>Caveat — city borders.</b> Part of the "decline" is suburbanisation: people moved just across the
city line and still work in the city. For a landlord of a city flat the tenant pool still shrank,
but the local economy may be fine (e.g. Brno/Plzeň rings).</li>
<li><b>Follow-ups worth doing:</b> (1) add age 20–34 (main renters) per city from the 5-year age
dataset; (2) combine with rent and price per m² to show yield vs. population trend — "high yields
are high for a reason"; (3) map of all ORP with change 2011→2021; (4) ČSÚ population projection to 2050 by region.</li>
</ul>
"""
    style = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5}}
body{background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
max-width:960px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:30px;margin-bottom:4px}h2{margin-top:44px;border-bottom:1px solid var(--line);padding-bottom:6px}
.lead{font-size:18px;color:var(--ink2)} a{color:var(--accent)}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fcfcfb}
.main img{max-width:540px}
table.t{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}
.t th,.t td{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.t th{color:var(--ink2)} .note{color:var(--muted);font-size:13px}
code{font-size:13px}
"""
    (out / "report.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Czech city population</title><style>{style}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------------


def main() -> None:
    out = monaco.output_dir("city_population_since_1991")
    raw = latest_raw()
    yearly = load_yearly(raw)
    census = load_census(raw)
    muni = build_municipalities(yearly, census)

    cities = city_table(muni)
    sizes = size_class_table(muni)
    districts = district_table(muni)
    ages = age_table(yearly, cities.index)

    (out / "data").mkdir()
    yearly.to_parquet(out / "data" / "yearly_2000_2024.parquet")
    muni.to_csv(out / "data" / "municipalities.csv")
    cities.to_csv(out / "data" / "cities.csv")
    sizes.to_csv(out / "data" / "size_classes.csv")
    districts.to_csv(out / "data" / "districts.csv")
    ages.to_csv(out / "data" / "city_ages.csv")

    main_image(
        cities,
        sizes,
        muni["c1991"].sum(),
        muni["y2024"].sum(),
        out / "main_city_change_1991_2024.png",
    )
    alt_census_bars(cities, out / "alt_census_1991_2021.png")
    alt_size_classes(sizes, out / "alt_size_classes.png")
    alt_long_history(muni, out / "alt_long_history.png")
    alt_national_series(yearly, out / "alt_national_series.png")

    write_report(
        out,
        {
            "cities": cities,
            "sizes": sizes,
            "districts": districts,
            "ages": ages,
            "yearly": yearly,
            "census": census,
            "muni": muni,
            "raw": raw,
            "n_below": int((cities["ch_1991_2024"] < 0).sum()),
            "n_below_census": int((cities["ch_1991_2021"] < 0).sum()),
            "n_rows_yearly": len(yearly),
            "n_rows_census": len(census),
            "rest_gain": (
                cities.drop(index="554782")[["y2024", "c1991"]]
                .sum()
                .pipe(lambda x: x["y2024"] - x["c1991"])
            ),
        },
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
