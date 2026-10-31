# ruff: noqa: E501  (long lines are Czech prose inside the HTML report and post texts)
"""How 10 years of investing would have turned out (2Q 2016 → 2Q 2026), from a Czech investor's view.

Compares S&P 500 (dividends reinvested), gold, an average Czech flat bought for cash and the same
flat bought with an 80 % mortgage at the interest rate of the time. Everything is in CZK.

Fair comparison: every strategy gets exactly the same money at the same time — 1 mil. Kč at the
start plus, every month, the amount of the mortgage installment. The mortgage strategy pays the
installment; the others invest it.

Reads the latest download from data/raw/investment_comparison/ (see download.py) and writes
processed data, one social-media image per variant, post texts and a Czech HTML report to
outputs/investment_comparison/<yymmdd_hhmmss>/.

ČSÚ flat prices are quarterly; 2Q 2026 is the newest quarter published, so the 10-year window is
2Q 2016 → 2Q 2026. Monthly series are taken at the middle month of the quarter (May 2016 → May 2026).

Usage: uv run scripts/20261030_investment_comparison/last_10_years.py
"""

import base64
import html
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import ScaledTranslation

import monaco

ANALYSIS = "investment_comparison"
RAW_ROOT = monaco.DATA_DIR / "raw" / ANALYSIS
START_Q, END_Q = pd.Period("2016Q2"), pd.Period("2026Q2")
OWN = 1_000_000  # own money at the start, CZK
TERM = 360  # mortgage term, months

URL_CSU = "https://csu.gov.cz/produkty/indexy-realizovanych-cen-bytu-2-ctvrtleti-2026"
URL_MIR = "https://data.ecb.europa.eu/data/datasets/MIR (série M.CZ.B.A2C.*.R.A.2250.CZK.N)"
URL_FX = "https://www.cnb.cz/cs/financni-trhy/devizovy-trh/kurzy-devizoveho-trhu/"
URL_SHILLER = "https://shillerdata.com/"
URL_FRED = "https://fred.stlouisfed.org/series/SP500"
URL_WB = "https://www.worldbank.org/en/research/commodity-markets"
URL_HICP = "https://ec.europa.eu/eurostat/databrowser/view/prc_hicp_minr/"
SOURCE = (
    "Zdroje: ČSÚ (indexy realizovaných cen starších bytů), ČNB (sazby nových hypoték, kurz USD/CZK),\n"
    "R. Shiller (S&P 500 vč. dividend), Světová banka (zlato). Měsíční průměry, vše v korunách.\n"
    "Nezapočteno: nájem a náklady bytu, daně, poplatky. Minulé výnosy nezaručují budoucí."
)

# Chart styling (reference dataviz palette). Colour follows the strategy in every chart.
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
STRATEGIES = {  # key: (label, colour)
    "byt_hypo": ("Byt s 80% hypotékou", "#2a78d6"),
    "sp500": ("Akcie S&P 500", "#eb6834"),
    "byt": ("Byt bez hypotéky", "#1baf7a"),
    "zlato": ("Zlato", "#eda100"),
}
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


def pct(x: float, signed: bool = False, digits: int = 1) -> str:
    """Czech percent: '12,4 %', '+3,5 %', '−2,0 %'."""
    s = f"{x * 100:{'+' if signed else ''}.{digits}f}"
    return s.replace("-", "−").replace(".", ",") + " %"


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ").replace("-", "−")


def kc(x: float) -> str:
    """Rounded to hundreds: '15 300 Kč'."""
    return num(round(x, -2)) + " Kč"


def mil(x: float, digits: int = 1) -> str:
    """Millions: '12,9 mil. Kč'."""
    return f"{x / 1e6:.{digits}f}".replace(".", ",").replace("-", "−") + " mil. Kč"


def dec(x: float, digits: int = 2) -> str:
    """Czech decimal: '1 234,56'."""
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def times(x: float) -> str:
    """Multiple: '3,2×'."""
    return f"{x:.1f}".replace(".", ",") + "×"


def qlabel(q: pd.Period) -> str:
    return f"{q.quarter}. čtvrtletí {q.year}"


# --------------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------------


def latest_raw() -> Path:
    return sorted(p for p in RAW_ROOT.iterdir() if p.is_dir())[-1]


def load_flats(raw: Path) -> pd.DataFrame:
    """ČSÚ realized prices of older flats, quarterly index (avg 2010 = 100): praha, mimo, cr."""
    df = pd.read_excel(raw / "csu_014007" / "01400726q2s.xlsx", header=None).iloc[9:, 1:5]
    df.columns = ["q", "praha", "mimo", "cr"]
    df = df.dropna(subset=["q"])
    df.index = pd.PeriodIndex([f"{q[3:]}Q{q[0]}" for q in df["q"]], freq="Q")
    return df[["praha", "mimo", "cr"]].astype(float)


def load_mir(raw: Path) -> pd.DataFrame:
    """ČNB new mortgage rates (% p.a.) by initial fixation: F ≤1 y, I 1–5 y, O 5–10 y, P >10 y, A all."""
    df = pd.read_csv(raw / "cnb_ecb_mir" / "mir_cz_house_purchase.csv")
    p = df.pivot(index="TIME_PERIOD", columns="MATURITY_NOT_IRATE", values="OBS_VALUE")
    p.index = pd.PeriodIndex(p.index, freq="M")
    return p


def load_usdczk(raw: Path) -> pd.Series:
    """ČNB daily CZK per 1 USD -> monthly average. The header line repeats when the list changes."""
    rows = []
    for path in sorted((raw / "cnb_fx").glob("kurzy_*.txt")):
        header: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if parts[0] == "Datum":
                header = parts
                continue
            value = parts[header.index("1 USD")].replace(",", ".")
            rows.append((pd.to_datetime(parts[0], format="%d.%m.%Y"), float(value)))
    s = pd.DataFrame(rows, columns=["date", "usd"]).set_index("date")["usd"]
    return s.groupby(s.index.to_period("M")).mean()


def load_spx(raw: Path) -> pd.DataFrame:
    """Shiller monthly S&P Composite: P (avg of daily closes), D (annualized dividend), tr index."""
    x = pd.read_excel(raw / "shiller" / "ie_data.xls", sheet_name="Data", header=None).iloc[8:, :3]
    x.columns = ["t", "P", "D"]
    x = x[pd.to_numeric(x["t"], errors="coerce").notna()]
    t = x["t"].astype(float)
    # Dates are written as year.month, e.g. 2016.1 = October 2016.
    x.index = pd.PeriodIndex(
        [pd.Period(year=int(v), month=round((v - int(v)) * 100), freq="M") for v in t]
    )
    x = x[["P", "D"]].astype(float)
    # Total return: monthly dividend (D / 12) reinvested at the month's price (Shiller's convention).
    x["tr"] = ((x["P"] + x["D"] / 12) / x["P"].shift(1)).fillna(1).cumprod()
    return x.dropna(subset=["D"])


def load_fred_spx(raw: Path) -> pd.Series:
    """FRED daily S&P 500 close -> monthly average, for cross-checking Shiller's P."""
    s = pd.read_csv(raw / "fred" / "SP500.csv", parse_dates=["observation_date"])
    s = s.set_index("observation_date")["SP500"].dropna()
    return s.groupby(s.index.to_period("M")).mean()


def load_gold(raw: Path) -> pd.Series:
    """World Bank Pink Sheet, gold USD per troy ounce, monthly average."""
    m = pd.read_excel(
        raw / "worldbank_cmo" / "CMO-Historical-Data-Monthly.xlsx",
        sheet_name="Monthly Prices",
        header=None,
    )
    col = next(i for i, v in m.iloc[4].items() if isinstance(v, str) and v.startswith("Gold"))
    g = m.iloc[6:, [0, col]].dropna()
    g.index = pd.PeriodIndex([s.replace("M", "-") for s in g[0]], freq="M")
    return pd.to_numeric(g[col], errors="coerce").dropna()


def load_hicp(raw: Path) -> pd.DataFrame:
    """Eurostat HICP Czechia, 2015 = 100: TOTAL (all items), CP041 (actual rentals)."""
    df = pd.read_csv(raw / "eurostat_hicp" / "hicp_cz.csv")
    p = df.pivot(index="TIME_PERIOD", columns="coicop18", values="OBS_VALUE")
    p.index = pd.PeriodIndex(p.index, freq="M")
    return p


@dataclass
class Data:
    flats: pd.DataFrame
    mir: pd.DataFrame
    usd: pd.Series
    spx: pd.DataFrame
    gold: pd.Series
    hicp: pd.DataFrame
    fred: pd.Series


def load(raw: Path) -> Data:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl print-area / xlrd deprecation noise
        return Data(
            flats=load_flats(raw),
            mir=load_mir(raw),
            usd=load_usdczk(raw),
            spx=load_spx(raw),
            gold=load_gold(raw),
            hicp=load_hicp(raw),
            fred=load_fred_spx(raw),
        )


# --------------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------------


def mid_month(q: pd.Period) -> pd.Period:
    return q.asfreq("M", "start") + 1


def flat_monthly(flats: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    """Quarterly index placed at the quarter's middle month, log-linear in between."""
    s = pd.Series(flats.values, index=[mid_month(q).to_timestamp() for q in flats.index])
    ts = months.to_timestamp()
    full = np.log(s).reindex(s.index.union(ts)).interpolate("time")
    return pd.Series(np.exp(full.reindex(ts).values), index=months)


def annuity(balance: float, rate: float, n: int) -> float:
    i = rate / 12
    return balance * i / (1 - (1 + i) ** -n)


@dataclass(frozen=True)
class Scenario:
    start: pd.Period = START_Q
    end: pd.Period = END_Q
    region: str = "cr"  # flats column: cr, praha, mimo
    fixation: str = "I"  # ČNB rate series used at the start and at every refixation
    fix_months: int = 60  # refixation period; >= horizon means the starting rate holds throughout
    ltv: float = 0.8
    buy_cost: float = 0.0  # one-off purchase costs as a share of the flat price, paid in cash
    spx_dividends: bool = True
    rent_yield: float = 0.0  # net rent per year as a share of the flat value at the start


def mortgage(price: float, sc: Scenario, months: pd.PeriodIndex, mir: pd.DataFrame) -> pd.DataFrame:
    """Monthly annuity schedule; at each refixation the installment is recomputed at the new rate."""
    balance = price * sc.ltv
    rows, rate, inst = [], np.nan, np.nan
    for k, m in enumerate(months[1:], start=1):
        if (k - 1) % sc.fix_months == 0:
            q = (m - 1).asfreq("Q")  # rate of the quarter the (re)fixation is signed in
            rate = mir.loc[q.asfreq("M", "start") : q.asfreq("M", "end"), sc.fixation].mean() / 100
            inst = annuity(balance, rate, TERM - (k - 1))
        interest = balance * rate / 12
        balance -= inst - interest
        rows.append((m, rate, inst, interest, balance))
    return pd.DataFrame(
        rows, columns=["month", "rate", "payment", "interest", "balance"]
    ).set_index("month")


def irr(cashflows: np.ndarray) -> float:
    """Annualized IRR of monthly cash flows (bisection on the monthly rate)."""
    t = np.arange(len(cashflows))
    lo, hi = -0.05, 0.1
    for _ in range(200):
        mid = (lo + hi) / 2
        if (cashflows / (1 + mid) ** t).sum() > 0:
            lo = mid
        else:
            hi = mid
    return (1 + mid) ** 12 - 1


def simulate(d: Data, sc: Scenario) -> dict:
    """Wealth paths of all strategies when each gets OWN at the start + the installment monthly."""
    months = pd.period_range(mid_month(sc.start), mid_month(sc.end), freq="M")
    flat = flat_monthly(d.flats[sc.region], months)
    spx_usd = d.spx["tr"] if sc.spx_dividends else d.spx["P"]
    prices = {
        "sp500": (spx_usd * d.usd).reindex(months),
        "zlato": (d.gold * d.usd).reindex(months),
        "byt": flat * (1 + sc.buy_cost),
    }
    assert not any(p.isna().any() for p in prices.values()), "missing monthly prices"

    price0 = OWN / (1 - sc.ltv + sc.buy_cost)  # flat price bought with OWN as down payment + costs
    sched = mortgage(price0, sc, months, d.mir)
    pay = sched["payment"].reindex(months, fill_value=0.0)
    invested = OWN + pay.cumsum()

    paths = {}
    units = {}
    for key, px in prices.items():
        units[key] = OWN / px.iloc[0] + (pay / px).cumsum()
        paths[key] = units[key] * px
    flat_value = price0 * flat / flat.iloc[0]
    paths["byt_hypo"] = flat_value - sched["balance"].reindex(months).fillna(price0 * sc.ltv)

    if sc.rent_yield:
        # Net rent per unit of flat grows with the HICP actual-rentals index; kept as cash.
        rent_idx = d.hicp["CP041"].reindex(months)
        rent_per_unit = sc.rent_yield / 12 * flat.iloc[0] * rent_idx / rent_idx.iloc[0]
        rent_unit = rent_per_unit.shift(1, fill_value=0.0)  # first rent a month after purchase
        paths["byt"] = paths["byt"] + (units["byt"].shift(1, fill_value=0) * rent_unit).cumsum()
        flat_units = price0 / flat.iloc[0]
        paths["byt_hypo"] = paths["byt_hypo"] + (flat_units * rent_unit).cumsum()

    paths = pd.DataFrame(paths)[list(STRATEGIES)]
    cf = -np.concatenate([[OWN], pay.iloc[1:].values])
    final = paths.iloc[-1]
    irrs = {}
    for key in paths:
        flows = cf.copy()
        flows[-1] += final[key]
        irrs[key] = irr(flows)
    return {
        "sc": sc,
        "months": months,
        "paths": paths,
        "invested": invested,
        "sched": sched,
        "price0": price0,
        "final": final,
        "irr": pd.Series(irrs),
        "total_in": invested.iloc[-1],
        "prices": pd.DataFrame(prices).assign(byt=flat),
    }


def lump_sum(d: Data, start: pd.Period = START_Q, end: pd.Period = END_Q) -> pd.DataFrame:
    """Value of OWN invested once at the start, no mortgage, no further contributions."""
    months = pd.period_range(mid_month(start), mid_month(end), freq="M")
    s = pd.DataFrame(
        {
            "sp500": (d.spx["tr"] * d.usd).reindex(months),
            "zlato": (d.gold * d.usd).reindex(months),
            "byt": flat_monthly(d.flats["cr"], months),
            "inflace": d.hicp["TOTAL"].reindex(months),
            "sp500_usd": d.spx["tr"].reindex(months),
            "sp500_price": (d.spx["P"] * d.usd).reindex(months),
            "zlato_usd": d.gold.reindex(months),
            "usd": d.usd.reindex(months),
            "praha": flat_monthly(d.flats["praha"], months),
            "mimo": flat_monthly(d.flats["mimo"], months),
        }
    )
    return OWN * s / s.iloc[0]


def max_drawdown(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min())


def robustness(d: Data, base: dict) -> pd.DataFrame:
    sc0 = base["sc"]
    rate_o = d.mir.loc["2016-04":"2016-06", "O"].mean()
    variants = {
        "Základ (fixace 1–5 let, refixace 2021)": sc0,
        f"Fixace 5–10 let, {pct(rate_o / 100, digits=2)} na celých 10 let": replace(
            sc0, fixation="O", fix_months=120
        ),
        "Sazba z roku 2016 na celých 10 let": replace(sc0, fix_months=120),
        "Okno 1Q 2016 → 1Q 2026": replace(sc0, start=pd.Period("2016Q1"), end=pd.Period("2026Q1")),
        "Okno 2Q 2015 → 2Q 2025": replace(sc0, start=pd.Period("2015Q2"), end=pd.Period("2025Q2")),
        "S&P 500 bez dividend": replace(sc0, spx_dividends=False),
        "Náklady na koupi bytu 4 % ceny": replace(sc0, buy_cost=0.04),
        "Hypotéka 70 % ceny": replace(sc0, ltv=0.7),
        "Hypotéka 90 % ceny": replace(sc0, ltv=0.9),
        "Byty jen Praha": replace(sc0, region="praha"),
        "Byty jen ČR bez Prahy": replace(sc0, region="mimo"),
        "S čistým nájmem 2 % ročně": replace(sc0, rent_yield=0.02),
        "S čistým nájmem 3 % ročně": replace(sc0, rent_yield=0.03),
        "S čistým nájmem 4 % ročně": replace(sc0, rent_yield=0.04),
    }
    rows = []
    for name, sc in variants.items():
        r = base if sc == sc0 else simulate(d, sc)
        f = r["final"]
        rows.append(
            {
                "varianta": name,
                "vloženo": r["total_in"],
                "splátka": r["sched"]["payment"].iloc[0],
                **{f"{k}": f[k] for k in STRATEGIES},
                "irr_hypo": r["irr"]["byt_hypo"],
                "irr_sp500": r["irr"]["sp500"],
                "hypo/sp500": (f["byt_hypo"] - r["total_in"]) / (f["sp500"] - r["total_in"]),
            }
        )
    return pd.DataFrame(rows).set_index("varianta")


# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------


def _clean(ax, grid: str = "y") -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so that neighbours are at least `gap` apart."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:], strict=True):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return list(out)


def social_image(
    path: Path,
    title: tuple[str, str],
    subtitle: list[str],
    draw: Callable,
    axes_rect: tuple[float, float, float, float] = (0.1, 0.17, 0.62, 0.58),
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
        fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.08, 0.105),
                   ncol=len(legend), frameon=False, fontsize=11)  # fmt: skip
    fig.text(0.06, 0.04, source, fontsize=9.5, color=MUTED, linespacing=1.5)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _mil_axis(v, _) -> str:
    return f"{v / 1e6:g}".replace(".", ",") + " mil." if v else "0"


def _line_chart(ax, months: pd.PeriodIndex, series: list[tuple[pd.Series, str, str, str]],
                gap: float) -> None:  # fmt: skip
    """series: (values, end label, colour, linestyle). Values in CZK, x = time."""
    x = months.to_timestamp()
    for values, _, color, ls in series:
        ax.plot(x, values.values, color=color, lw=2.6 if ls == "-" else 1.8, ls=ls,
                solid_capstyle="round")  # fmt: skip
    _clean(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(_mil_axis))
    ax.tick_params(labelsize=11)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, max(v.max() for v, *_ in series) * 1.06)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    # Direct end labels: coloured dot at the line end, ink text on the right, de-overlapped.
    ends = [(v.iloc[-1], label, color) for v, label, color, _ in series]
    ys = _spread([y for y, _, _ in ends], gap)
    shift = ax.transData + ScaledTranslation(14 / 72, 0, ax.figure.dpi_scale_trans)
    for (y, label, color), ly in zip(ends, ys, strict=True):
        ax.plot([x[-1]], [y], "o", color=color, ms=8, mec=SURFACE, mew=2, zorder=5, clip_on=False)
        ax.text(x[-1], ly, label, transform=shift, va="center", fontsize=12, color=INK,
                linespacing=1.3)  # fmt: skip


def chart_a_wealth(base: dict, path: Path) -> dict:
    p, inv, f = base["paths"], base["invested"], base["final"]
    gains = f - base["total_in"]
    best_other = gains.drop("byt_hypo").idxmax()
    ratio = gains["byt_hypo"] / gains[best_other]
    sched = base["sched"]
    r1, r2 = sched["rate"].iloc[0], sched["rate"].iloc[-1]
    series = [(p[k], f"{STRATEGIES[k][0]}\n{mil(f[k])}", STRATEGIES[k][1], "-") for k in STRATEGIES]
    series.append((inv, f"Vloženo celkem\n{mil(base['total_in'])}", MUTED, "--"))
    social_image(
        path,
        (
            "Byt na hypotéku vydělal za 10 let",
            f"{times(ratio).replace('×', 'krát')} víc než akcie i zlato",
        ),
        [
            f"Hodnota investice v korunách, {qlabel(START_Q)} → {qlabel(END_Q)}. Všechny čtyři strategie",
            f"dostaly stejné peníze: na začátku 1 mil. Kč, pak každý měsíc {kc(sched['payment'].iloc[0])} (splátka hypotéky).",
            "Byt s hypotékou: 1 mil. Kč jako 20% akontace, 80 % úvěr na 30 let, cena podle průměru ČR.",
            f"Sazba {pct(r1, digits=2)} (průměr nových hypoték, ČNB), od roku 2021 refixace na {pct(r2, digits=2)}.",
        ],
        lambda ax: _line_chart(ax, base["months"], series, gap=1.05e6),
    )
    return {"ratio": ratio, "best_other": best_other, "gains": gains, "r1": r1, "r2": r2}


def chart_b_lump(lump: pd.DataFrame, months: pd.PeriodIndex, path: Path) -> dict:
    end = lump.iloc[-1] / OWN
    keys = ["sp500", "zlato", "byt"]
    lo, hi = end[keys].min(), end[keys].max()
    series = [(lump[k], f"{STRATEGIES[k][0] if k != 'byt' else 'Průměrný byt'}\n{mil(lump[k].iloc[-1])}",
               STRATEGIES[k][1], "-") for k in keys]  # fmt: skip
    series.append((lump["inflace"], f"Inflace\n{mil(lump['inflace'].iloc[-1])}", MUTED, "--"))
    social_image(
        path,
        ("Bez hypotéky skončily akcie, zlato", "i byt skoro stejně"),
        [
            f"Hodnota 1 mil. Kč vloženého jednorázově v {qlabel(START_Q)}, bez dalších vkladů.",
            f"Za 10 let peníze vzrostly {times(lo)} až {times(hi)}. Akcie v dolarech by vydělaly víc ({times(end['sp500_usd'])}),",
            f"ale koruna k dolaru posílila o {pct(1 - lump['usd'].iloc[-1] / lump['usd'].iloc[0], digits=0)} a část zisku smazala.",
            "Šedá čárkovaná: kolik by stálo totéž co 1 mil. Kč v roce 2016 (inflace HICP).",
        ],
        lambda ax: _line_chart(ax, months, series, gap=0.32e6),
    )
    return {"end": end, "lo": lo, "hi": hi}


def chart_c_regions(regions: dict, path: Path) -> dict:
    rows = []
    for reg, label in (("mimo", "mimo Prahu"), ("cr", "průměr ČR"), ("praha", "Praha")):
        f = regions[reg]["final"]
        rows.append((f"Byt s hypotékou – {label}", f["byt_hypo"], STRATEGIES["byt_hypo"][1]))
        rows.append((f"Byt bez hypotéky – {label}", f["byt"], STRATEGIES["byt"][1]))
    f = regions["cr"]["final"]
    rows += [("Zlato", f["zlato"], STRATEGIES["zlato"][1]),
             ("Akcie S&P 500", f["sp500"], STRATEGIES["sp500"][1])]  # fmt: skip
    df = pd.DataFrame(rows, columns=["label", "value", "color"]).sort_values("value")
    total_in = regions["cr"]["total_in"]

    def draw(ax) -> None:
        y = np.arange(len(df))
        ax.barh(y, df["value"], color=df["color"], height=0.68)
        ax.set_yticks(y, df["label"], fontsize=12)
        ax.axvline(total_in, color=MUTED, lw=1.4, ls="--")
        ax.text(total_in, len(df) - 0.35, f" vloženo {mil(total_in)}", color=INK2, fontsize=10.5,
                va="bottom")  # fmt: skip
        for i, v in enumerate(df["value"]):
            ax.text(v + df["value"].max() * 0.01, i, mil(v), va="center", fontsize=11, color=INK2)
        _clean(ax, "x")
        ax.xaxis.set_major_formatter(FuncFormatter(_mil_axis))
        ax.set_xlim(0, df["value"].max() * 1.17)
        ax.set_ylim(-0.6, len(df) - 0.1)

    m, p = regions["mimo"]["final"], regions["praha"]["final"]
    social_image(
        path,
        ("Na hypotéku vydělaly nejvíc", "byty mimo Prahu"),
        [
            f"Hodnota v {qlabel(END_Q)}, stejné vklady od {qlabel(START_Q)}: 1 mil. Kč na začátku",
            f"a {kc(regions['cr']['sched']['payment'].iloc[0])} měsíčně, celkem {mil(regions['cr']['total_in'])}.",
            f"Byty mimo Prahu zdražily {times(regions['mimo']['prices']['byt'].iloc[-1] / regions['mimo']['prices']['byt'].iloc[0])}, pražské {times(regions['praha']['prices']['byt'].iloc[-1] / regions['praha']['prices']['byt'].iloc[0])}.",
            f"Hypotéka ten rozdíl znásobila: {mil(m['byt_hypo'])} proti {mil(p['byt_hypo'])}.",
        ],
        draw,
        axes_rect=(0.36, 0.17, 0.56, 0.6),
    )
    return {"df": df}


def chart_d_irr(base: dict, lump: pd.DataFrame, path: Path) -> dict:
    r = base["irr"].sort_values()
    infl = (lump["inflace"].iloc[-1] / lump["inflace"].iloc[0]) ** 0.1 - 1

    def draw(ax) -> None:
        y = np.arange(len(r))
        ax.barh(y, r.values * 100, color=[STRATEGIES[k][1] for k in r.index], height=0.6)
        ax.set_yticks(y, [STRATEGIES[k][0] for k in r.index], fontsize=13)
        for i, v in enumerate(r.values):
            ax.text(v * 100 + 0.3, i, pct(v), va="center", fontsize=13, color=INK, weight="bold")
        ax.axvline(infl * 100, color=MUTED, lw=1.4, ls="--")
        ax.text(infl * 100, len(r) - 0.45, f" inflace {pct(infl)} ročně", color=INK2, fontsize=11,
                va="bottom")  # fmt: skip
        _clean(ax, "x")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} %"))
        ax.set_xlim(0, r.max() * 100 * 1.18)
        ax.set_ylim(-0.6, len(r) - 0.1)

    social_image(
        path,
        (
            f"Byt na hypotéku vynášel {pct(r['byt_hypo'], digits=0)} ročně,",
            f"akcie i zlato kolem {pct((r['sp500'] + r['zlato']) / 2, digits=0)}",
        ),
        [
            f"Průměrné roční zhodnocení vlastních peněz, {qlabel(START_Q)} → {qlabel(END_Q)}, v korunách.",
            f"Stejné vklady u všech strategií: 1 mil. Kč na začátku + {kc(base['sched']['payment'].iloc[0])} měsíčně (splátka hypotéky).",
            "Výnos počítaný jako vnitřní výnosové procento (IRR) z těchto vkladů a konečné hodnoty.",
            "U bytu bez nájmu a nákladů, u akcií s reinvestovanými dividendami.",
        ],
        draw,
        axes_rect=(0.3, 0.2, 0.62, 0.55),
    )
    return {"irr": r, "infl": infl}


def chart_rates(d: Data, path: Path) -> None:
    """Methodology chart for the report: new mortgage rates by fixation."""
    m = d.mir.loc["2015-01":, ["F", "I", "O"]]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
    x = m.index.to_timestamp()
    for col, label, color in (("I", "fixace 1–5 let", "#2a78d6"), ("O", "fixace 5–10 let", "#eb6834"),
                              ("F", "plovoucí / do 1 roku", "#1baf7a")):  # fmt: skip
        ax.plot(x, m[col], color=color, lw=2, label=label)
    for q in (START_Q, START_Q + 20):
        ax.axvspan(q.start_time, q.end_time, color=GRID, alpha=0.8)
    _clean(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Úrokové sazby nových hypoték v ČR (ČNB); šedě čtvrtletí sjednání a refixace",
                 fontsize=10.5, loc="left", color=INK2)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Post texts
# --------------------------------------------------------------------------------------------


def post_texts(base: dict, lump: pd.DataFrame, regions: dict, rob: pd.DataFrame, k: dict) -> dict:
    f, total_in = base["final"], base["total_in"]
    pay = base["sched"]["payment"].iloc[0]
    end = k["B"]["end"]
    irr_ = k["D"]["irr"]
    fl = base["prices"]["byt"]
    flat_x = fl.iloc[-1] / fl.iloc[0]
    rent3 = rob.loc["S čistým nájmem 3 % ročně"]
    fix10 = rob.iloc[1]
    m, p = regions["mimo"], regions["praha"]
    return {
        "A": (
            f"Kdo v roce 2016 koupil byt na hypotéku, má dnes víc než ten, kdo stejné peníze dal do amerických akcií nebo do zlata. A to výrazně.\n\n"
            f"Spočítali jsme to férově: každá strategie dostala na začátku 1 mil. Kč a pak každý měsíc {kc(pay)} (přesně splátku hypotéky), celkem {mil(total_in)}. "
            f"Za 10 let z toho bylo: byt s 80% hypotékou {mil(f['byt_hypo'])}, zlato {mil(f['zlato'])}, akcie S&P 500 {mil(f['sp500'])}, byt bez hypotéky {mil(f['byt'])}.\n\n"
            f"Proč? Průměrný byt v ČR zdražil {times(flat_x)} a hypotéka s úrokem kolem {pct(k['A']['r1'], digits=1)} ten růst znásobila — vydělávali jste na ceně celého bytu, ne jen na svém milionu. A to ještě nepočítáme nájem.\n\n"
            "Jenže páka funguje oběma směry. Těchto 10 let bylo pro byty výjimečných: levné úvěry na začátku a ceny skoro bez poklesu. S dnešními sazbami kolem 5 % a cenami na rekordu je výchozí bod úplně jiný."
        ),
        "B": (
            f"Milion korun vložený v roce 2016 jednorázově, bez hypotéky. Kolik je to dnes? Akcie S&P 500 {mil(lump['sp500'].iloc[-1])}, zlato {mil(lump['zlato'].iloc[-1])}, průměrný byt v ČR {mil(lump['byt'].iloc[-1])}.\n\n"
            f"Skoro remíza. Americké akcie by přitom v dolarech vydělaly víc ({times(end['sp500_usd'])}), ale koruna k dolaru posílila o {pct(1 - lump['usd'].iloc[-1] / lump['usd'].iloc[0], digits=0)} a část zisku smazala.\n\n"
            f"Pro srovnání: kvůli inflaci dnes potřebujete {mil(lump['inflace'].iloc[-1])}, abyste koupili totéž, co za milion v roce 2016. Všechny tři investice inflaci porazily s velkou rezervou.\n\n"
            "Rozdíl mezi bytem a akciemi tedy nedělá samotný byt, ale to, jak se kupuje. Byt si většina lidí kupuje na úvěr — a to je ta páka, která rozhoduje."
        ),
        "C": (
            f"Byty mimo Prahu zdražily za 10 let {times(m['prices']['byt'].iloc[-1] / m['prices']['byt'].iloc[0])}, pražské „jen“ {times(p['prices']['byt'].iloc[-1] / p['prices']['byt'].iloc[0])}.\n\n"
            f"S 80% hypotékou se ten rozdíl zvětší: ze stejných vkladů (1 mil. Kč a {kc(pay)} měsíčně) by byt mimo Prahu dnes měl čistou hodnotu {mil(m['final']['byt_hypo'])}, pražský {mil(p['final']['byt_hypo'])}. Akcie S&P 500 {mil(f['sp500'])}.\n\n"
            "Regiony vyšly lépe, protože začínaly z nízkých cen. Za ten růst ale mohly i levné úvěry a nedostatek bytů — dnes jsou ceny i mimo Prahu jinde a výnos z nájmu nižší.\n\n"
            "Jako pronajímatel se při další koupi ptejte, jestli bude cena i nájem v daném městě dál růst. Minulých 10 let to platilo skoro všude, to ale není zákon."
        ),
        "D": (
            f"Byt koupený v roce 2016 s 80% hypotékou vynášel vlastním penězům v průměru {pct(irr_['byt_hypo'])} ročně. Akcie S&P 500 {pct(irr_['sp500'])}, zlato {pct(irr_['zlato'])}, byt bez hypotéky {pct(irr_['byt'])}. Inflace byla {pct(k['D']['infl'])} ročně.\n\n"
            f"U bytu přitom nepočítáme nájem. Kdyby čistý nájem (po nákladech a neobsazenosti) dělal 3 % ceny ročně, byl by výnos bytu na hypotéku {pct(rent3['irr_hypo'])} ročně.\n\n"
            f"Výsledek nestojí na detailech: s desetiletou fixací by to bylo {pct(fix10['irr_hypo'])}, s 4% náklady na koupi bytu {pct(rob.loc['Náklady na koupi bytu 4 % ceny', 'irr_hypo'])}.\n\n"
            "Je to ale ohlédnutí za výjimečnou dekádou. Stejný výpočet pro příštích 10 let začíná s dvojnásobnou sazbou a o třetinu nižším výnosem z nájmu. Páka zvětšuje zisky i ztráty."
        ),
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _img(path: Path) -> str:
    return (
        f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="">'
    )


def _post(text: str) -> str:
    paras = "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n"))
    return f'<div class="post"><div class="post-label">Text příspěvku</div>{paras}</div>'


def _table(df: pd.DataFrame, fmt: dict[str, Callable]) -> str:
    d = df.copy()
    for col, f in fmt.items():
        d[col] = d[col].map(lambda v, f=f: "" if pd.isna(v) else f(v))
    return d.to_html(escape=True, border=0, classes="t")


def write_report(out: Path, ctx: dict) -> None:
    d, base, lump, regions, rob, posts, k, imgs, raw, chk = (
        ctx[x] for x in ("d", "base", "lump", "regions", "rob", "posts", "k", "imgs", "raw", "chk")
    )
    f, total_in, sched = base["final"], base["total_in"], base["sched"]
    pay0, pay1 = sched["payment"].iloc[0], sched["payment"].iloc[-1]
    interest = sched["interest"].sum()
    price0 = base["price0"]
    fl = base["prices"]["byt"]
    flat_end_value = price0 * fl.iloc[-1] / fl.iloc[0]
    debt_end = sched["balance"].iloc[-1]
    mdd = ctx["mdd"]
    esc = html.escape
    names = {k_: v[0] for k_, v in STRATEGIES.items()}

    yearly = base["paths"].copy()
    yearly["vloženo"] = base["invested"]
    yearly = yearly[yearly.index.month == base["months"][0].month]
    yearly.index = yearly.index.year
    yearly = yearly.rename(columns=names)

    rob_view = rob.rename(columns={**names, "irr_hypo": "IRR byt s hyp.", "irr_sp500": "IRR S&P 500",
                                   "hypo/sp500": "zisk hyp. / zisk S&P"})  # fmt: skip
    m_fmt = {c: mil for c in ["vloženo", *names.values()]}

    sources = [
        ("ČSÚ — Indexy realizovaných cen starších bytů (014007-26), čtvrtletně, průměr 2010 = 100",
         URL_CSU, d.flats, d.flats.loc[START_Q:END_Q, "cr"]),
        ("ČNB — úrokové sazby nových úvěrů domácnostem na bydlení v CZK podle fixace (MIR), měsíčně, přes ECB Data Portal",
         URL_MIR, d.mir, d.mir.loc["2016-01":, "I"]),
        ("ČNB — kurzy devizového trhu, CZK za 1 USD, denně → měsíční průměr", URL_FX,
         d.usd.to_frame("CZK/USD"), d.usd.loc["2016-05":]),
        ("Robert Shiller — S&P Composite, měsíční průměr denních uzávěrek (P) a dividendy (D, roční)",
         URL_SHILLER, d.spx, d.spx.loc["2016-05":, "P"]),
        ("Světová banka — Pink Sheet, zlato USD/trojská unce, měsíční průměr", URL_WB,
         d.gold.to_frame("USD/oz"), d.gold.loc["2016-05":]),
        ("Eurostat — HICP Česko (data ČSÚ), 2015 = 100: celkem (TOTAL) a skutečné nájemné (CP041)",
         URL_HICP, d.hicp, d.hicp.loc["2016-05":, "TOTAL"]),
        ("FRED — S&P 500 denní uzávěrky (jen kontrola Shillerovy řady)", URL_FRED,
         d.fred.to_frame("SP500"), d.fred),
    ]  # fmt: skip
    src_html = ""
    for title, url, df, s in sources:
        src_html += (
            f'<h3>{esc(title)}</h3><p><a href="{esc(url)}">{esc(url)}</a><br>'
            f"Řádků: {len(df)}, období {esc(str(df.index.min()))} – {esc(str(df.index.max()))}. "
            f"Sledovaná řada v analyzovaném období: min {dec(s.min())}, max {dec(s.max())}, poslední {dec(s.iloc[-1])}.</p>"
            f"{df.tail(4).round(3).to_html(border=0, classes='t')}"
        )

    body = f"""
<h1>10 let investování: S&amp;P 500, zlato a byt (s hypotékou i bez)</h1>
<p class="lead">Hlavní závěr: <b>byt koupený v {qlabel(START_Q)} s 80% hypotékou vydělal za 10 let
{times(k["A"]["ratio"])} víc než akcie S&amp;P 500</b> ze stejných vkladů. Z {mil(total_in)} vložených
(1 mil. Kč + {kc(pay0)} měsíčně) by dnes bylo {mil(f["byt_hypo"])} v bytě s hypotékou, {mil(f["sp500"])} v akciích,
{mil(f["zlato"])} ve zlatě a {mil(f["byt"])} v bytě bez hypotéky. Bez páky by byt, zlato i akcie skončily
podobně ({times(k["B"]["lo"])}–{times(k["B"]["hi"])} za 10 let). Rozhodla páka levné hypotéky.</p>

<p class="note">Proč 2Q 2016 → 2Q 2026: ČSÚ zveřejňuje ceny bytů čtvrtletně, poslední je 2. čtvrtletí 2026
(vydáno 14. 9. 2026). Měsíční řady (akcie, zlato, kurz, sazby) bereme v prostředním měsíci čtvrtletí,
tj. květen 2016 → květen 2026 (121 měsíců, 120 splátek).</p>

<h2>Hlavní obrázek</h2>
<div class="main">{_img(imgs["A"])}</div>
<p>Jako hlavní doporučuji <b>variantu A</b>: odpovídá přesně na otázku („jak by se vyvíjela investice“), ukazuje
vývoj v čase i konečné částky a srovnání je férové — všechny strategie dostaly stejné peníze ve stejný čas.
Pro pronajímatele je překvapivé, jak velký náskok měl byt na hypotéku i bez započtení nájmu, a zároveň
to otevírá důležitou debatu o rizicích páky. Varianta B je dobrý navazující příspěvek („bez hypotéky remíza“).</p>

<h2>Varianta A — vývoj investice v čase (stejné vklady)</h2>
{_img(imgs["A"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Vklady:</b> každá strategie dostane 1 000 000 Kč v květnu 2016 a pak 120 měsíčních vkladů ve výši splátky
hypotéky ({kc(pay0)} do refixace, {kc(pay1)} po ní), celkem {mil(total_in)}.</li>
<li><b>Byt s hypotékou:</b> 1 mil. Kč = 20 % ceny → byt za {mil(price0)}, úvěr {mil(price0 * 0.8)} na 30 let, anuita.
Sazba = průměr ČNB za {qlabel(START_Q)} pro nové hypotéky s fixací 1–5 let ({pct(k["A"]["r1"], digits=2)}); po 5 letech
refixace na průměr za {qlabel(START_Q + 20)} ({pct(k["A"]["r2"], digits=2)}). Hodnota = cena bytu podle indexu ČSÚ − zbývající dluh.
Na konci: byt {mil(flat_end_value)}, dluh {mil(debt_end)}, zaplacené úroky celkem {mil(interest)}.</li>
<li><b>Byt bez hypotéky:</b> za každý vklad se „dokoupí“ podíl průměrného bytu za aktuální cenu (index ČSÚ ČR celkem).</li>
<li><b>S&amp;P 500:</b> celkový výnos (dividendy reinvestované), přepočteno měsíčním průměrem kurzu ČNB na koruny.</li>
<li><b>Zlato:</b> měsíční průměrná cena (Světová banka) přepočtená na koruny.</li>
<li><b>Graf:</b> čárový, hodnota v mil. Kč, šedá čárkovaná = kumulativně vloženo. Barvy jsou stejné ve všech
variantách (modrá byt s hypotékou, oranžová akcie, tyrkysová byt bez hypotéky, žlutá zlato). Popisky přímo u konců čar.</li>
<li><b>Kontrola:</b> tabulka robustnosti níže — náskok bytu s hypotékou drží ve všech testovaných variantách.</li>
</ul>
{_post(posts["A"])}

<h2>Varianta B — jednorázový milion bez hypotéky</h2>
{_img(imgs["B"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Výpočet:</b> 1 mil. Kč × (cena v květnu 2026 / cena v květnu 2016), bez dalších vkladů. Byt = index ČSÚ ČR.</li>
<li><b>Inflace:</b> HICP Česko (Eurostat, data ČSÚ) — kolik korun je potřeba dnes na koupi toho, co stálo 1 mil. Kč v roce 2016.</li>
<li><b>Kurzový efekt:</b> S&amp;P 500 v dolarech {times(k["B"]["end"]["sp500_usd"])}, v korunách {times(k["B"]["end"]["sp500"])};
zlato v dolarech {times(k["B"]["end"]["zlato_usd"])}, v korunách {times(k["B"]["end"]["zlato"])}.
Bez dividend by S&amp;P 500 v korunách dalo {times(k["B"]["end"]["sp500_price"])}.</li>
<li><b>Riziko po cestě (max. propad v měsíčních datech):</b> S&amp;P 500 v Kč {pct(mdd["sp500"], digits=0)}, zlato {pct(mdd["zlato"], digits=0)},
byt {pct(mdd["byt"], digits=0)}. Pozor: index bytů je čtvrtletní průměr transakcí, propady proto vypadají menší, než by byla
skutečná kolísavost ceny jednoho bytu.</li>
</ul>
{_post(posts["B"])}

<h2>Varianta C — Praha vs. zbytek republiky</h2>
{_img(imgs["C"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Data:</b> stejné jako A, jen index cen bytů ČSÚ pro Prahu, resp. ČR bez Prahy (ČR celkem = vážený průměr 23,1 : 76,9).</li>
<li><b>Výpočet:</b> stejné vklady jako A, stejná hypotéka (splátka nezávisí na regionu, protože vždy jde o 1 mil. Kč = 20 % ceny).</li>
<li><b>Graf:</b> vodorovné sloupce seřazené podle konečné hodnoty, barva = typ strategie, čárkovaně vloženo celkem.</li>
<li><b>Pozor:</b> výnos z nájmu je v Praze typicky nižší než v regionech, takže se započtením nájmu by rozdíl ve prospěch regionů byl spíš větší.
Index „ČR bez Prahy“ pokrývá celou republiku, jednotlivá města se mohou výrazně lišit.</li>
</ul>
{_post(posts["C"])}

<h2>Varianta D — roční výnos vlastních peněz (IRR)</h2>
{_img(imgs["D"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Výpočet:</b> vnitřní výnosové procento z měsíčních toků: −1 mil. Kč na začátku, −splátka každý měsíc, + konečná hodnota.
Převedeno na roční (p. a.). Je to číslo srovnatelné s „výnosem fondu“ nebo úrokem na spořicím účtu.</li>
<li><b>Inflace:</b> průměrná roční inflace HICP za stejné období.</li>
<li><b>Graf:</b> vodorovné sloupce seřazené podle výnosu.</li>
</ul>
{_post(posts["D"])}

<h2>Vývoj po letech (varianta A)</h2>
{_table(yearly, {c: mil for c in yearly.columns})}
<p class="note">Hodnoty vždy v květnu daného roku.</p>

<h2>Robustnost</h2>
<p>Každý řádek je celá simulace znovu se změněným jedním předpokladem. „zisk hyp. / zisk S&amp;P“ = (konečná hodnota − vloženo)
bytu s hypotékou děleno totéž pro S&amp;P 500.</p>
{_table(rob_view, {**m_fmt, "splátka": kc, "IRR byt s hyp.": pct, "IRR S&P 500": pct, "zisk hyp. / zisk S&P": times})}
<p class="note">Varianty s nájmem: čistý nájem (po nákladech, opravách a neobsazenosti) jako % ceny bytu v roce 2016, dále roste
podle indexu skutečného nájemného HICP (CP041, +{pct(d.hicp["CP041"].loc[mid_month(END_Q)] / d.hicp["CP041"].loc[mid_month(START_Q)] - 1, digits=0)} za 10 let).
Nájem se jen hromadí na účtu (bez úroku), takže je to spíš dolní odhad.</p>
<img src="data:image/png;base64,{base64.b64encode(imgs["R"].read_bytes()).decode()}" alt="" style="max-width:720px">

<h2>Datové zdroje</h2>
{src_html}
<p class="note">Kontrola Shiller vs. FRED: měsíční průměr denních uzávěrek se v období 5/2016–6/2026 liší nejvýš o {pct(chk["fred_diff"], digits=2)}.</p>

<h2>Předpoklady</h2>
<ul>
<li><b>Průměrný byt</b> = index realizovaných cen starších bytů ČSÚ (ČR celkem). Index měří změnu ceny, ne úroveň; částky
jsou proto „na 1 mil. Kč vlastních peněz“. Výsledky jsou lineární — se 2 mil. Kč vlastních by vše bylo dvojnásobné.
Byt za {mil(price0)} v roce 2016 byl nadprůměrně drahý, procenta ale platí pro jakoukoli cenu.</li>
<li><b>Hypotéka:</b> 80 % ceny (LTV), 30 let, anuitní splátka, sazba = průměr nových hypoték ČNB s fixací 1–5 let
(nejčastější fixace v roce 2016) za čtvrtletí sjednání, refixace po 5 letech na tehdejší průměr nových hypoték.
Bez poplatků bance, bez pojištění.</li>
<li><b>Stejné vklady:</b> peníze na splátky musel investor mít; ostatní strategie stejné peníze investovaly. Tak se srovnává
„co se stalo s mými penězi“, ne jen procento růstu ceny.</li>
<li><b>Nájem a náklady bytu</b> (fond oprav, daň z nemovitosti, pojištění, opravy, neobsazenost) v základu nezapočítány.
Čistý nájem je u pronajímaného bytu kladný, takže základní výpočet byt spíš podhodnocuje (viz robustnost s nájmem).</li>
<li><b>Daně a poplatky:</b> nezapočteny. Byt koupený v roce 2016 je po 10 letech osvobozen od daně z příjmu při prodeji
(časový test 5 let pro byty nabyté před 2021), akcie po 3 letech také. Srážková daň z dividend (15 %) by výnos S&amp;P snížila asi
o 0,2 p. b. ročně, poplatek ETF asi o 0,1 p. b. Daň z nabytí nemovitosti (4 %) v květnu 2016 platil prodávající — viz řádek „náklady 4 %“.</li>
<li><b>Prodej:</b> konečná hodnota je hodnota „na papíře“; prodej bytu stojí čas a provizi, prodej ETF/zlata skoro nic.</li>
<li><b>Měsíční průměry:</b> všechny ceny jsou průměry za měsíc (index bytů za čtvrtletí, mezi čtvrtletími logaritmicky interpolován).</li>
</ul>

<h2>Výpočty</h2>
<ul>
<li>Celkový výnos S&amp;P 500: TR<sub>t</sub> = TR<sub>t−1</sub> × (P<sub>t</sub> + D<sub>t</sub>/12) / P<sub>t−1</sub> (Shillerova konvence), v Kč × kurz CZK/USD.</li>
<li>Splátka: anuita B × i / (1 − (1 + i)<sup>−n</sup>), i = sazba/12, n = zbývající počet měsíců z 360; při refixaci přepočtena ze zůstatku.</li>
<li>Hodnota bytu s hypotékou: cena<sub>0</sub> × index<sub>t</sub> / index<sub>0</sub> − zůstatek úvěru<sub>t</sub>.</li>
<li>Ostatní strategie: počet „jednotek“ = 1 mil. / cena<sub>0</sub> + Σ splátka<sub>k</sub> / cena<sub>k</sub>; hodnota = jednotky × cena<sub>t</sub>.</li>
<li>IRR: měsíční sazba r řešící Σ CF<sub>t</sub> / (1 + r)<sup>t</sup> = 0 (bisekce), roční = (1 + r)<sup>12</sup> − 1.</li>
</ul>

<h2>Postup</h2>
<ol>
<li>Výběr zdrojů: ČSÚ (realizované, ne nabídkové ceny bytů), ČNB (sazby nových hypoték, kurz), Shiller (jediná volně dostupná měsíční řada
S&amp;P 500 s dividendami; ověřeno proti FRED), Světová banka (zlato), Eurostat/ČSÚ (inflace, nájemné).</li>
<li>Stažení skriptem <code>download.py</code> do <code>{esc(str(raw.relative_to(monaco.ROOT)))}</code>.</li>
<li>Zpracování: převod všeho na měsíční řady v Kč, interpolace čtvrtletního indexu bytů.</li>
<li>Simulace čtyř strategií se stejnými peněžními toky, výpočet hodnoty v čase a IRR.</li>
<li>Kandidátní grafy A–D, kontrola robustnosti (14 variant), kontrola zdrojových dat (Shiller vs. FRED).</li>
<li>Výstupy: obrázky 1080 × 1350 px, texty příspěvků, tento report, data v CSV.</li>
</ol>

<h2>Proč je to zajímavé pro pronajímatele a investory</h2>
<ul>
<li>Odpovídá na nejčastější otázku „mám koupit další byt, nebo dát peníze do ETF?“ — na skutečných číslech, ne na pocitech.</li>
<li>Ukazuje, že výhoda bytu nebyla v bytě samotném (bez páky remíza s akciemi i zlatem), ale v levném úvěru.
To je užitečné při rozhodování dnes, kdy jsou úvěry dvakrát dražší.</li>
<li>Srovnání Praha vs. regiony a citlivost na nájem pomáhají při výběru lokality.</li>
</ul>

<h2>Moje poznámky</h2>
<ul>
<li><b>Je to důvěryhodné?</b> Zdroje jsou oficiální a čísla jsou robustní na všechny testované změny předpokladů. Největší
nejistota je v tom, že index ČSÚ je průměr trhu — konkrétní byt mohl dopadnout jinak.</li>
<li><b>Výjimečné období.</b> 2016 byl začátek největšího růstu cen bytů v historii ČR, se sazbami hypoték kolem 2 %.
Stejný výpočet od roku 2008 nebo od 2022 by vypadal jinak. Varování o páce do příspěvků patří.</li>
<li><b>Riziko a likvidita nejsou v číslech.</b> Byt s 80% hypotékou znamená dluh {mil(price0 * 0.8)} na 1 mil. Kč vlastních. Kdyby ceny
klesly o 20 %, vlastní kapitál zmizí. S&amp;P 500 v korunách během období krátce ztratilo {pct(-mdd["sp500"], digits=0)}, ale bez dluhu.</li>
<li><b>Náklad na čas.</b> Pronajímání je práce, ETF ne. To čísla nezachycují.</li>
<li><b>Další kroky:</b> (1) přidat skutečný výnos z nájmu po městech (Deloitte Rent Index / ČSÚ); (2) stejné srovnání pro start
v každém roce 2009–2016; (3) „co dnes“: stejný model s dnešními sazbami a cenami jako scénáře.</li>
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
        f"<title>Investice za 10 let</title><style>{style}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------------


def main() -> None:
    out = monaco.output_dir(ANALYSIS)
    raw = latest_raw()
    d = load(raw)

    base = simulate(d, Scenario())
    regions = {
        r: (base if r == "cr" else simulate(d, Scenario(region=r))) for r in ("cr", "praha", "mimo")
    }
    lump = lump_sum(d)
    rob = robustness(d, base)
    window = pd.period_range("2016-05", "2026-06", freq="M")
    chk = {"fred_diff": float((d.spx["P"] / d.fred - 1).reindex(window).abs().max())}
    mdd = {k_: max_drawdown(lump[k_]) for k_ in ("sp500", "zlato", "byt")}

    (out / "data").mkdir()
    paths = base["paths"].assign(vlozeno=base["invested"])
    paths.to_csv(out / "data" / "strategie_mesicne.csv")
    base["sched"].to_csv(out / "data" / "hypoteka_splatky.csv")
    lump.to_csv(out / "data" / "jednorazove_1mil.csv")
    rob.to_csv(out / "data" / "robustnost.csv")
    base["prices"].to_csv(out / "data" / "ceny_kc_mesicne.csv")

    imgs = {
        "A": out / "A_investice_v_case.png",
        "B": out / "B_jednorazove_bez_hypoteky.png",
        "C": out / "C_praha_vs_regiony.png",
        "D": out / "D_rocni_vynos.png",
        "R": out / "metodika_sazby_hypotek.png",
    }
    k = {
        "A": chart_a_wealth(base, imgs["A"]),
        "B": chart_b_lump(lump, base["months"], imgs["B"]),
        "C": chart_c_regions(regions, imgs["C"]),
        "D": chart_d_irr(base, lump, imgs["D"]),
    }
    chart_rates(d, imgs["R"])
    posts = post_texts(base, lump, regions, rob, k)
    (out / "texty_prispevku.md").write_text(
        "\n\n".join(f"## Varianta {v}\n\n{t}" for v, t in posts.items()) + "\n", encoding="utf-8"
    )
    write_report(
        out,
        {
            "d": d,
            "base": base,
            "lump": lump,
            "regions": regions,
            "rob": rob,
            "posts": posts,
            "k": k,
            "imgs": imgs,
            "raw": raw,
            "chk": chk,
            "mdd": mdd,
        },
    )
    print(rob.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
