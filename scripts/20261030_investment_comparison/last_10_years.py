# ruff: noqa: E501  (long lines are Czech prose inside the HTML report and post texts)
"""How 10 years of investing would have turned out (2Q 2016 → 2Q 2026), from a Czech investor's view.

Puts 1 mil. Kč in once (no further contributions) into S&P 500 (dividends reinvested), gold, a
savings account, an average Czech flat bought for cash and rented out, and the same flat bought with
the 1 mil. Kč as a 20 % down payment on a mortgage at the interest rate of the time, also rented out.
Everything is in CZK.

The flats' monthly cash flow (rent − costs − installment) may be negative; it is charted separately
(B2), counted in the flat's net result (B, C) and in its IRR as further investment (D).

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
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

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
URL_DEP = "https://data.ecb.europa.eu/data/datasets/MIR (série M.CZ.B.L21+L22.*.R.A.2250.CZK.N)"
URL_CPI = "https://vdb.czso.cz/pll/eweb/package_show?id=010022"
SOURCE = (
    "Zdroje: ČSÚ (ceny starších bytů), ČNB (sazby hypoték a vkladů, kurz USD/CZK), R. Shiller\n"
    "(S&P 500 vč. dividend), Světová banka (zlato), Eurostat/ČSÚ (inflace HICP). Vše v korunách.\n"
    "Nezapočteno: nájem a náklady bytu, daně (kromě 15% daně z úroků), poplatky. Minulé výnosy nezaručují budoucí."
)
SOURCE_B = (
    "Zdroje: ČSÚ (ceny starších bytů), ČNB (sazby hypoték a vkladů, kurz USD/CZK), R. Shiller\n"
    "(S&P 500 vč. dividend), Světová banka (zlato), Eurostat/ČSÚ (inflace, nájemné), ČNB/IRI (výše nájmu 2016).\n"
    "Nezapočteno: daň z příjmu z nájmu, poplatky, úrok z hotovosti. Minulé výnosy nezaručují budoucí."
)

# Rented flats (variants B, D). Gross rent = 2016 price / PRICE_TO_RENT: ČNB Financial Stability
# Report 2015/2016, indicator table part 1 (PDF page 160), row TN.4 "Poměr cena bytu / roční
# nájemné (dle IRI)", latest value (early 2016) 21.9; 2015 average 21.5. One month a year empty,
# owner costs (repair fund, insurance, property tax, repairs) as a share of the 2016 price. Rent
# grows with HICP actual rentals (CP041), costs with HICP total.
URL_ZFS = "https://www.cnb.cz/export/sites/cnb/cs/financni-stabilita/.galleries/zpravy_fs/fs_2015-2016/fs_2015-2016.pdf"
PRICE_TO_RENT = 21.9
GROSS_YIELD = 1 / PRICE_TO_RENT
VACANCY = 1 / 12
OWNER_COST = 0.01
SAVINGS_TAX = 0.15  # withholding tax on deposit interest

# Chart styling (reference dataviz palette). Colour follows the strategy in every chart.
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
STRATEGIES = {  # key: (label, colour)
    "byt_hypo_najem": ("Pronajatý byt s hypotékou", "#2a78d6"),
    "sp500": ("Akcie S&P 500", "#eb6834"),
    "byt_najem": ("Pronajatý byt bez hypotéky", "#1baf7a"),
    "zlato": ("Zlato", "#eda100"),
    "sporici": ("Spořicí účet", "#6250d6"),
}
FLATS = ["byt_hypo_najem", "byt_najem"]
# Variant D only: flat bought for cash and left empty (owner costs, no rent). Same hue as the
# rented one, told apart by hatching.
EMPTY = "byt_prazdny"
EMPTY_LABEL = "Nepronajatý byt bez hypotéky"
# Variants B3, B4: where a flat's profit comes from (cumulative, CZK). Categorical order of the
# reference palette.
PARTS = {  # key: (label, colour)
    "growth": ("Růst ceny bytu", "#2a78d6"),
    "rent": ("Nájem", "#eb6834"),
    "costs": ("Náklady vlastníka", "#1baf7a"),
    "interest": ("Úroky hypotéky", "#eda100"),
}
D_LABELS = {
    EMPTY: "Nepronajatý byt\nbez hypotéky",
    "byt_hypo_najem": "Pronajatý byt\ns 80% hypotékou",
    "byt_najem": "Pronajatý byt\nbez hypotéky",
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


def load_deposits(raw: Path) -> pd.DataFrame:
    """ČNB household CZK deposit rates, new business (% p.a.): overnight (L21), agreed maturity ≤1 y (F)."""
    df = pd.read_csv(raw / "cnb_ecb_mir_deposits" / "mir_cz_deposits.csv")
    df["col"] = (
        df["BS_ITEM"].map({"L21": "overnight", "L22": "term"}) + "_" + df["MATURITY_NOT_IRATE"]
    )
    p = df.pivot(index="TIME_PERIOD", columns="col", values="OBS_VALUE")
    p.index = pd.PeriodIndex(p.index, freq="M")
    return p.rename(columns={"overnight_A": "overnight", "term_F": "term_1y"})


def load_cpi(raw: Path) -> pd.Series:
    """ČSÚ national CPI, all items, monthly index (avg 2015 = 100)."""
    df = pd.read_csv(raw / "csu_cpi" / "010022.csv")
    df = df[df["ucel_kod"].isna() & (df["casz_kod"] == "Z")]
    idx = pd.PeriodIndex(
        [f"{y}-{m:02d}" for y, m in zip(df["rok"], df["mesic"], strict=True)], freq="M"
    )
    return pd.Series(df["hodnota"].values, index=idx).sort_index()


@dataclass
class Data:
    flats: pd.DataFrame
    mir: pd.DataFrame
    usd: pd.Series
    spx: pd.DataFrame
    gold: pd.Series
    hicp: pd.DataFrame
    fred: pd.Series
    dep: pd.DataFrame
    cpi: pd.Series


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
            dep=load_deposits(raw),
            cpi=load_cpi(raw),
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


def savings_index(rates: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    """Value of 1 Kč on a deposit: each month earns that month's rate / 12, after withholding tax."""
    r = rates.reindex(months) / 100 * (1 - SAVINGS_TAX) / 12
    assert not r.isna().any(), "missing deposit rates"
    return (1 + r.shift(1, fill_value=0.0)).cumprod()


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
    buy_cost: float = 0.0  # one-off purchase costs as a share of the flat price, paid from OWN
    gross_yield: float = GROSS_YIELD  # rent per year as a share of the flat price at the start
    owner_cost: float = OWNER_COST  # owner costs per year as a share of the flat price at the start
    spx_dividends: bool = True


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


def flat_strategies(d: Data, sc: Scenario) -> dict:
    """Rented flats for OWN put in once: bought for cash or with a mortgage.

    Monthly cash flow = net rent − owner costs − installment (mortgage only); it starts
    the month after purchase. Net result = flat value − remaining debt + cumulative cash flow, i.e.
    out-of-pocket top-ups are subtracted and surpluses kept as cash without interest.
    """
    months = pd.period_range(mid_month(sc.start), mid_month(sc.end), freq="M")
    flat = flat_monthly(d.flats[sc.region], months)
    growth = flat / flat.iloc[0]
    rent_g = d.hicp["CP041"].reindex(months) / d.hicp["CP041"].reindex(months).iloc[0]
    cost_g = d.hicp["TOTAL"].reindex(months) / d.hicp["TOTAL"].reindex(months).iloc[0]
    after_purchase = pd.Series(np.arange(len(months)) > 0, index=months).astype(float)

    big = OWN / (1 - sc.ltv + sc.buy_cost)  # OWN covers the down payment and purchase costs
    small = OWN / (1 + sc.buy_cost)
    sched = mortgage(big, sc, months, d.mir)
    pay = sched["payment"].reindex(months, fill_value=0.0)
    debt = sched["balance"].reindex(months).fillna(big * sc.ltv)

    def rent(price0: float) -> pd.Series:
        return sc.gross_yield / 12 * (1 - VACANCY) * price0 * rent_g * after_purchase

    def costs(price0: float) -> pd.Series:
        return sc.owner_cost / 12 * price0 * cost_g * after_purchase

    cf = pd.DataFrame(
        {
            "byt_hypo_najem": rent(big) - costs(big) - pay,
            "byt_najem": rent(small) - costs(small),
            EMPTY: -costs(small),
        }
    )
    equity = pd.DataFrame(
        {
            "byt_hypo_najem": big * growth - debt,
            "byt_najem": small * growth,
            EMPTY: small * growth,
        }
    )
    return {
        "months": months,
        "cf": cf,
        "equity": equity,
        "wealth": equity + cf.cumsum(),
        "sched": sched,
        "big": big,
        "rent_big": rent(big),
        "costs_big": costs(big),
        # Net result − OWN split into its sources (repaid principal is the flat's own value, so
        # the mortgage only costs its interest). Purchase costs are left out.
        "parts": {
            "byt_hypo_najem": pd.DataFrame(
                {
                    "growth": big * (growth - 1),
                    "rent": rent(big).cumsum(),
                    "costs": -costs(big).cumsum(),
                    "interest": -sched["interest"].reindex(months, fill_value=0.0).cumsum(),
                }
            ),
            "byt_najem": pd.DataFrame(
                {
                    "growth": small * (growth - 1),
                    "rent": rent(small).cumsum(),
                    "costs": -costs(small).cumsum(),
                    "interest": 0.0,
                }
            ),
        },
    }


def lump_sum(d: Data, sc: Scenario) -> pd.DataFrame:
    """Value of OWN invested once at the start, no mortgage, no further contributions."""
    months = pd.period_range(mid_month(sc.start), mid_month(sc.end), freq="M")
    spx = d.spx["tr"] if sc.spx_dividends else d.spx["P"]
    s = pd.DataFrame(
        {
            "sp500": (spx * d.usd).reindex(months),
            "zlato": (d.gold * d.usd).reindex(months),
            "sporici": savings_index(d.dep["term_1y"], months),
            "byt": flat_monthly(d.flats[sc.region], months),
            "inflace": d.hicp["TOTAL"].reindex(months),
            "sporici_bezny": savings_index(d.dep["overnight"], months),
            "sp500_usd": d.spx["tr"].reindex(months),
            "sp500_price": (d.spx["P"] * d.usd).reindex(months),
            "zlato_usd": d.gold.reindex(months),
            "usd": d.usd.reindex(months),
        }
    )
    assert not s.isna().any().any(), "missing monthly prices"
    return OWN * s / s.iloc[0]


def per_year(value: float, months: pd.PeriodIndex) -> float:
    """Average yearly growth of OWN grown to `value` over `months` (B-style value, not IRR as in D)."""
    return (value / OWN) ** (12 / (len(months) - 1)) - 1


def run(d: Data, sc: Scenario | None = None) -> dict:
    """All strategies for OWN put in once at the start of the window: value paths and yearly return.

    Return of the flats = IRR of −OWN, the monthly cash flows (top-up = further investment, surplus
    = withdrawal) and the final equity; of the others = average yearly growth (their IRR).
    """
    sc = sc or Scenario()
    fl = flat_strategies(d, sc)
    lump = lump_sum(d, sc)
    paths = pd.concat([fl["wealth"], lump[["sp500", "zlato", "sporici"]]], axis=1)[list(STRATEGIES)]
    n = len(fl["months"]) - 1
    irrs = {k: (lump[k].iloc[-1] / OWN) ** (12 / n) - 1 for k in ("sp500", "zlato", "sporici")}
    for k in (*FLATS, EMPTY):
        flows = np.concatenate([[-OWN], fl["cf"][k].values[1:]])
        flows[-1] += fl["equity"][k].iloc[-1]
        irrs[k] = irr(flows)
    return {
        "sc": sc,
        "fl": fl,
        "lump": lump,
        "paths": paths,
        "final": paths.iloc[-1],
        "irr": pd.Series(irrs)[[*STRATEGIES, EMPTY]],
    }


def robustness(d: Data) -> pd.DataFrame:
    """Final value of OWN and yearly return (IRR) under one changed assumption at a time."""
    sc0 = Scenario()
    variants = {
        f"Základ (nájem {pct(GROSS_YIELD)}, náklady {pct(OWNER_COST, digits=0)}, fixace 1–5 let)": sc0,
        "Hrubý nájem 4 % ceny": replace(sc0, gross_yield=0.04),
        "Hrubý nájem 6 % ceny": replace(sc0, gross_yield=0.06),
        "Náklady vlastníka 0,5 % ceny": replace(sc0, owner_cost=0.005),
        "Náklady vlastníka 1,5 % ceny": replace(sc0, owner_cost=0.015),
        "Fixace 5–10 let na celých 10 let": replace(sc0, fixation="O", fix_months=120),
        "Sazba z roku 2016 na celých 10 let": replace(sc0, fix_months=120),
        "Hypotéka 70 % ceny": replace(sc0, ltv=0.7),
        "Hypotéka 90 % ceny": replace(sc0, ltv=0.9),
        "Náklady na koupi bytu 4 % ceny": replace(sc0, buy_cost=0.04),
        "Okno 1Q 2016 → 1Q 2026": replace(sc0, start=pd.Period("2016Q1"), end=pd.Period("2026Q1")),
        "Okno 2Q 2015 → 2Q 2025": replace(sc0, start=pd.Period("2015Q2"), end=pd.Period("2025Q2")),
        "Byty jen Praha": replace(sc0, region="praha"),
        "Byty jen ČR bez Prahy": replace(sc0, region="mimo"),
        "S&P 500 bez dividend": replace(sc0, spx_dividends=False),
    }
    rows = {}
    for name, sc in variants.items():
        r = run(d, sc)
        rows[name] = {
            **{STRATEGIES[k][0]: r["final"][k] for k in STRATEGIES},
            **{f"IRR {STRATEGIES[k][0]}": r["irr"][k] for k in (*FLATS, "sp500")},
        }
    return pd.DataFrame(rows).T


def max_drawdown(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min())


def inflation_table(d: Data) -> pd.DataFrame:
    """Inflation by year: HICP May→May (the analysis window) and ČSÚ CPI annual average."""
    rows = []
    for y in range(START_Q.year, END_Q.year):
        a, b = pd.Period(f"{y}-05", "M"), pd.Period(f"{y + 1}-05", "M")
        cpi_avg = (d.cpi[d.cpi.index.year == y + 1].mean() / d.cpi[d.cpi.index.year == y].mean() - 1
                   if (d.cpi.index.year == y + 1).sum() == 12 else np.nan)  # fmt: skip
        rows.append(
            {
                "období": f"5/{y} → 5/{y + 1}",
                "HICP (Eurostat/ČSÚ)": d.hicp.loc[b, "TOTAL"] / d.hicp.loc[a, "TOTAL"] - 1,
                "CPI ČSÚ, stejné období": d.cpi.loc[b] / d.cpi.loc[a] - 1
                if b in d.cpi.index
                else np.nan,
                "CPI ČSÚ, průměr roku konce": cpi_avg,
                "nájemné HICP (CP041)": d.hicp.loc[b, "CP041"] / d.hicp.loc[a, "CP041"] - 1,
            }
        )
    return pd.DataFrame(rows).set_index("období")


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
                gap: float, yfmt: Callable = _mil_axis, fontsize: float = 12) -> None:  # fmt: skip
    """series: (values, end label, colour, linestyle). Values in CZK, x = time."""
    x = months.to_timestamp()
    for values, _, color, ls in series:
        ax.plot(x, values.values, color=color, lw={"-": 2.6, ":": 2.4}.get(ls, 1.8), ls=ls,
                solid_capstyle="round")  # fmt: skip
    _clean(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
    ax.tick_params(labelsize=11)
    ax.set_xlim(x[0], x[-1])
    lo = min(v.min() for v, *_ in series)
    hi = max(v.max() for v, *_ in series)
    ax.set_ylim(min(0, lo * 1.08), hi * 1.06)
    if lo < 0:
        ax.axhline(0, color=INK2, lw=1)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    # Direct end labels: coloured dot at the line end, ink text on the right, de-overlapped.
    ends = [(v.iloc[-1], label, color) for v, label, color, _ in series]
    ys = _spread([y for y, _, _ in ends], gap)
    shift = ax.transData + ScaledTranslation(14 / 72, 0, ax.figure.dpi_scale_trans)
    for (y, label, color), ly in zip(ends, ys, strict=True):
        ax.plot([x[-1]], [y], "o", color=color, ms=8, mec=SURFACE, mew=2, zorder=5, clip_on=False)
        ax.text(x[-1], ly, label, transform=shift, va="center", fontsize=fontsize, color=INK,
                linespacing=1.3)  # fmt: skip


def chart_b_lump(base: dict, path: Path) -> dict:
    """1 mil. Kč put in once: rented flats (with / without mortgage), S&P, gold, savings."""
    fl, lump, paths = base["fl"], base["lump"], base["paths"]
    end = lump.iloc[-1] / OWN
    sched = fl["sched"]
    order = list(base["final"].sort_values(ascending=False).index)
    series = [(paths[k], f"{STRATEGIES[k][0]}\n{mil(paths[k].iloc[-1])}, {pct(per_year(paths[k].iloc[-1], fl['months']))} ročně",
               STRATEGIES[k][1], "-") for k in order]  # fmt: skip
    series.append(
        (lump["inflace"], f"Inflace (HICP)\n{mil(lump['inflace'].iloc[-1])}", MUTED, "--")
    )
    top = paths[order[0]].iloc[-1]
    # Claims in the subtitle and post text.
    assert order[0] == "byt_hypo_najem" and fl["cf"]["byt_hypo_najem"].sum() > 0
    assert base["final"].drop("sporici").min() > lump["inflace"].iloc[-1]
    assert base["final"]["sporici"] < lump["inflace"].iloc[-1]
    social_image(
        path,
        (
            "Milion z roku 2016: v pronajatém bytě",
            f"na hypotéku {mil(top)}, na spořáku {mil(lump['sporici'].iloc[-1])}",
        ),
        [
            f"Hodnota 1 mil. Kč vloženého jednorázově v {qlabel(START_Q)}. Byt bez hypotéky za 1 mil. Kč,",
            f"s hypotékou za {mil(fl['big'], 0)} (80% úvěr, sazba {pct(sched['rate'].iloc[0], digits=1)}, od 2021 {pct(sched['rate'].iloc[-1], digits=1)}).",
            "U bytů: cena − dluh + nájem − náklady − splátky. Co bylo nutné doplácet, je odečteno (graf B2).",
            f"Nájem {pct(GROSS_YIELD)} ceny ročně, 1 měsíc prázdný, náklady {pct(OWNER_COST, digits=0)} ceny. Spořicí účet: sazby ČNB",
            f"vkladů do 1 roku, po dani. Šedá čárkovaná: inflace, {pct(end['inflace'] ** 0.1 - 1)} ročně (2022–23 přes 10 %).",
        ],
        lambda ax: _line_chart(ax, fl["months"], series, gap=0.75e6, fontsize=11),
        axes_rect=(0.1, 0.15, 0.56, 0.6),
        source=SOURCE_B,
    )
    return {"end": end, "order": order}


def _kc_axis(v, _) -> str:
    return f"{v / 1000:+.0f} tis.".replace("-", "−") if v else "0"


def chart_b2_cashflow(fb: dict, path: Path) -> dict:
    """Monthly cash flow of the variant B flats: rent − costs − installment."""
    cf = fb["cf"].iloc[1:]
    months = fb["months"][1:]
    tot = cf.sum()
    series = [
        (cf[k], f"{STRATEGIES[k][0]}\n{kc(cf[k].iloc[-1])} měsíčně, celkem {mil(tot[k])}",
         STRATEGIES[k][1], "-")
        for k in FLATS
    ]  # fmt: skip
    neg = cf["byt_hypo_najem"] < 0
    last_neg = cf.index[neg][-1] if neg.any() else None
    assert last_neg is not None and not neg.loc[last_neg + 1 :].any(), (
        "title: negative, then positive"
    )
    years_neg = round((last_neg - cf.index[0]).n / 12)
    social_image(
        path,
        (
            f"Pronajatý byt na hypotéku {years_neg} let doplácel,",
            f"dnes vydělává {kc(cf['byt_hypo_najem'].iloc[-1])} měsíčně",
        ),
        [
            f"Měsíční cashflow pronajatých bytů z varianty B, {qlabel(START_Q)} → {qlabel(END_Q)}:",
            f"nájem − náklady − splátka hypotéky (byt za {mil(fb['big'], 0)}, 80% úvěr, splátka {kc(fb['sched']['payment'].iloc[0])}).",
            f"Nájem {pct(GROSS_YIELD)} ceny z roku 2016 ročně (1 měsíc v roce prázdný), roste s nájemným v HICP.",
            f"Náklady (fond oprav, pojištění, daň, opravy) {pct(OWNER_COST, digits=0)} ceny ročně, rostou s inflací.",
            "Pronajatý byt s hypotékou doplácel"
            + (f" do {last_neg.strftime('%m/%Y')}, pak byl v plusu." if last_neg else " nikdy.")
            + f" Za 10 let celkem {mil(tot['byt_hypo_najem'])}.",
        ],
        lambda ax: _line_chart(ax, months, series, gap=900, yfmt=_kc_axis, fontsize=11),
        axes_rect=(0.12, 0.17, 0.5, 0.58),
        source=SOURCE_B,
    )
    return {"cf": cf, "tot": tot, "last_neg": last_neg}


FLAT_PANELS = {
    "byt_hypo_najem": "Pronajatý byt s 80% hypotékou (byt za {price})",
    "byt_najem": "Pronajatý byt bez hypotéky (byt za {price})",
}


def _profit_parts(fl: dict) -> dict:
    """Variants B3, B4: cumulative profit sources of both rented flats; checks they add up."""
    parts = fl["parts"]
    for k in FLATS:
        gap = (parts[k].sum(axis=1) - (fl["wealth"][k] - OWN)).abs().max()
        assert gap < 1, f"profit parts of {k} do not add up (purchase costs?)"
    return parts


def _panel_title(ax, k: str, fl: dict) -> None:
    price = mil(fl["big"], 0) if k == "byt_hypo_najem" else mil(OWN, 0)
    ax.set_title(FLAT_PANELS[k].format(price=price), loc="left", fontsize=14, weight="bold",
                 color=INK, pad=10)  # fmt: skip


def _parts_legend() -> list:
    return [plt.matplotlib.patches.Patch(color=c, label=label) for label, c in PARTS.values()] + [
        plt.matplotlib.lines.Line2D([], [], color=INK, lw=2.4, label="Zisk celkem")
    ]


def chart_b3_parts_time(fl: dict, path: Path) -> dict:
    """Profit of the variant B flats over time, split into price growth, rent, costs, interest."""
    parts = _profit_parts(fl)
    x = fl["months"].to_timestamp()
    end = {k: parts[k].iloc[-1] for k in FLATS}
    hypo = end["byt_hypo_najem"]
    assert hypo["rent"] > -(hypo["costs"] + hypo["interest"]), (
        "title: rent covered costs + interest"
    )
    assert all(e["growth"] > e["rent"] for e in end.values()), "title: growth made most of it"

    def panel(ax, k: str) -> None:
        p = parts[k]
        for keys in (("growth", "rent"), ("costs", "interest")):  # stacked up / down from zero
            base = np.zeros(len(p))
            for c in keys:
                top = base + p[c].values
                ax.fill_between(x, base, top, color=PARTS[c][1], lw=1, edgecolor=SURFACE)
                base = top
        net = p.sum(axis=1)
        ax.plot(x, net.values, color=INK, lw=2.4, solid_capstyle="round")
        ax.axhline(0, color=INK2, lw=1)
        _clean(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(_mil_axis))
        ax.tick_params(labelsize=11)
        ax.set_xlim(x[0], x[-1])
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        lo, hi = (p.clip(upper=0).sum(axis=1).min(), p.clip(lower=0).sum(axis=1).max())
        ax.set_ylim(lo - (hi - lo) * 0.06, hi * 1.06)
        # End labels: band middles for the parts, line end for the total.
        e = p.iloc[-1]
        mids = {"growth": e["growth"] / 2, "rent": e["growth"] + e["rent"] / 2,
                "costs": e["costs"] / 2, "interest": e["costs"] + e["interest"] / 2}  # fmt: skip
        labels = [(mids[c], f"{PARTS[c][0]} {'+' if e[c] > 0 else ''}{mil(e[c])}")
                  for c in PARTS if e[c] != 0]  # fmt: skip
        labels.append((net.iloc[-1], f"Zisk celkem {mil(net.iloc[-1])}"))
        gap = (hi - lo) * 0.085
        ys = _spread([y for y, _ in labels], gap)
        shift = ax.transData + ScaledTranslation(10 / 72, 0, ax.figure.dpi_scale_trans)
        for (_, text), ly in zip(labels, ys, strict=True):
            ax.text(x[-1], ly, text, transform=shift, va="center", fontsize=11, color=INK,
                    weight="bold" if text.startswith("Zisk") else "normal")  # fmt: skip
        _panel_title(ax, k, fl)

    def draw(ax) -> None:
        ax.set_axis_off()
        panel(ax.inset_axes([0, 0.58, 1, 0.42]), "byt_hypo_najem")
        panel(ax.inset_axes([0, 0, 1, 0.42]), "byt_najem")

    social_image(
        path,
        ("Nájem pokryl náklady i úroky,", "zisk z bytu ale udělal hlavně růst ceny"),
        [
            f"Zisk z 1 mil. Kč vloženého v {qlabel(START_Q)} do pronajatého bytu (varianta B), načítaný od začátku.",
            "Nad nulou příjmy: růst ceny bytu a vybraný nájem. Pod nulou výdaje: náklady vlastníka",
            "a zaplacené úroky (splacená jistina zůstává v bytě, není to náklad). Černá čára = zisk celkem.",
            f"Nájem {pct(GROSS_YIELD)} ceny ročně, 1 měsíc prázdný; náklady {pct(OWNER_COST, digits=0)} ceny. Každý graf má vlastní měřítko.",
        ],
        draw,
        axes_rect=(0.1, 0.17, 0.58, 0.575),
        legend=_parts_legend(),
        source=SOURCE_B,
    )
    return {"end": end}


def chart_b4_parts(fl: dict, path: Path) -> dict:
    """Variant B3 without time: waterfall from the 1 mil. Kč put in to today's value, per flat."""
    parts = _profit_parts(fl)
    end = {k: parts[k].iloc[-1] for k in FLATS}

    def panel(ax, k: str) -> None:
        e = end[k]
        rows = [("Vloženo", 0.0, OWN, MUTED, mil(OWN))]
        level = OWN
        for c in PARTS:
            if e[c] == 0:
                continue
            sign = "+ " if e[c] > 0 else "− "
            rows.append((sign + PARTS[c][0].lower(), level, level + e[c], PARTS[c][1],
                         ("+" if e[c] > 0 else "") + mil(e[c])))  # fmt: skip
            level += e[c]
        rows.append(("= Hodnota dnes", 0.0, level, INK2, mil(level)))
        y = np.arange(len(rows))[::-1]
        for yi, (_, a, b, color, _) in zip(y, rows, strict=True):
            ax.barh(yi, b - a, left=a, color=color, height=0.62)
        for yi, (_, a, b, _, text) in zip(y, rows, strict=True):
            bold = text == rows[-1][4]
            ax.text(max(a, b) + level * 0.015, yi, text, va="center", fontsize=12, color=INK,
                    weight="bold" if bold else "normal")  # fmt: skip
        # Thin connectors from one bar's end to the next bar's start.
        for i in range(len(rows) - 1):
            b0 = rows[i][2]
            ax.plot([b0, b0], [y[i] - 0.31, y[i + 1] + 0.31], color=AXIS, lw=1)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=12.5)
        _clean(ax, "x")
        ax.xaxis.set_major_formatter(FuncFormatter(_mil_axis))
        ax.tick_params(labelsize=11)
        ax.set_xlim(0, level * 1.22)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        _panel_title(ax, k, fl)

    def draw(ax) -> None:
        ax.set_axis_off()
        panel(ax.inset_axes([0, 0.56, 1, 0.44]), "byt_hypo_najem")
        panel(ax.inset_axes([0, 0, 1, 0.38]), "byt_najem")

    hypo = end["byt_hypo_najem"]
    profit = hypo.sum()
    assert hypo["growth"] / profit > 0.5, "title: growth made most of the profit"
    social_image(
        path,
        (
            f"Z {mil(profit)} zisku bytu na hypotéku",
            f"{mil(hypo['growth'])} udělal růst ceny, nájem {mil(hypo['rent'])}",
        ),
        [
            f"Odkud se vzal zisk z 1 mil. Kč vloženého v {qlabel(START_Q)} do pronajatého bytu (varianta B),",
            f"stav v {qlabel(END_Q)}. Růst ceny podle indexu ČSÚ, nájem {pct(GROSS_YIELD)} ceny ročně (1 měsíc prázdný),",
            f"náklady vlastníka {pct(OWNER_COST, digits=0)} ceny ročně, úroky z hypotéky (sazba {pct(fl['sched']['rate'].iloc[0])}, od 2021 {pct(fl['sched']['rate'].iloc[-1])}).",
            "Splacená jistina není náklad — zůstává v hodnotě bytu. Každý graf má vlastní měřítko.",
        ],
        draw,
        axes_rect=(0.3, 0.12, 0.62, 0.65),
        source=SOURCE_B,
    )
    return {"end": end}


def _parts_bars(ax, fl: dict, keys: list[str]) -> dict:
    """Variants B5, B6: profit parts of the given flats as bars on one shared scale, each with its
    share of the total profit; the total with the yearly return on the 1 mil. Kč put in."""
    parts = _profit_parts(fl)
    end = {k: parts[k].iloc[-1] for k in keys}
    share = {k: e / e.sum() for k, e in end.items()}
    rows = []  # (y, label, value, colour, share or None); groups top to bottom
    y = 0.0
    groups = []
    for k in keys:
        groups.append((y, k))
        y -= 0.85
        for c in PARTS:
            if end[k][c] != 0:
                rows.append((y, PARTS[c][0], end[k][c], PARTS[c][1], share[k][c]))
                y -= 1
        rows.append((y, "Zisk celkem", end[k].sum(), INK2, None))
        y -= 1.5
    hi = max(r[2] for r in rows)
    lo = min(r[2] for r in rows)

    ys = [r[0] for r in rows]
    ax.barh(ys, [r[2] for r in rows], color=[r[3] for r in rows], height=0.66)
    ax.set_yticks(ys, [r[1] for r in rows], fontsize=12.5)
    dx = hi * 0.012
    for yi, label, val, _, sh in rows:
        x = max(val, 0) + dx
        sign = "+" if val > 0 and label != "Zisk celkem" else ""
        note = (f"{pct(sh, digits=0)} zisku" if sh is not None
                else f"{pct(per_year(OWN + val, fl['months']))} ročně z vkladu")  # fmt: skip
        ax.text(x, yi + 0.02, sign + mil(val), va="bottom", fontsize=12.5, color=INK,
                weight="bold")  # fmt: skip
        ax.text(x, yi - 0.02, note, va="top", fontsize=11, color=INK2)
    price = {"byt_hypo_najem": mil(fl["big"], 0), "byt_najem": mil(OWN, 0)}
    bounds = [g[0] for g in groups] + [-np.inf]
    for (gy, k), below in zip(groups, bounds[1:], strict=True):
        ax.text(-0.45, gy, f"{STRATEGIES[k][0]} (byt za {price[k]})", fontsize=14,
                weight="bold", color=INK, va="center", transform=ax.get_yaxis_transform(),
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 3})  # fmt: skip
        span = [r[0] for r in rows if below < r[0] < gy]  # zero line only along the rows
        ax.plot([0, 0], [min(span) - 0.45, max(span) + 0.45], color=INK2, lw=1)
    _clean(ax, "x")
    ax.xaxis.set_major_formatter(FuncFormatter(_mil_axis))
    ax.tick_params(labelsize=11)
    ax.set_xlim(lo * 1.6, hi * 1.3)
    ax.set_ylim(y + 1.0, 0.5)
    return {"end": end, "share": share}


def chart_b5_parts_bars(fl: dict, path: Path) -> dict:
    """Variant B4 as bars on one shared scale: each profit part with its share of the total profit."""
    end = {k: fl["parts"][k].iloc[-1] for k in FLATS}  # the title needs the shares before drawing
    rent_share = [end[k]["rent"] / end[k].sum() for k in FLATS]
    assert all(s_ < 0.5 for s_ in rent_share), "title: rent made the smaller part"
    res: dict = {}

    def draw(ax) -> None:
        res.update(_parts_bars(ax, fl, FLATS))

    rmin, rmax = (round(v * 100) for v in sorted(rent_share)[:: len(rent_share) - 1])
    rent_txt = f"{rmin} %" if rmin == rmax else f"{rmin}–{rmax} %"
    social_image(
        path,
        ("Zisk z pronajatého bytu udělal hlavně", f"růst ceny, nájem jen {rent_txt} zisku"),
        [
            f"Odkud se vzal zisk z 1 mil. Kč vloženého v {qlabel(START_Q)} do pronajatého bytu (varianta B),",
            f"stav v {qlabel(END_Q)}. Procenta = podíl na zisku celkem (hodnota dnes − vložený 1 mil. Kč),",
            "části se sčítají na 100 %. Obě varianty bytu ve stejném měřítku.",
            f"Nájem {pct(GROSS_YIELD)} ceny ročně (1 měsíc prázdný), náklady vlastníka {pct(OWNER_COST, digits=0)} ceny ročně.",
            "Splacená jistina není náklad — zůstává v hodnotě bytu, počítají se jen úroky.",
            "U zisku celkem průměrné roční zhodnocení vkladu 1 mil. Kč (složeně, jako ve variantě E).",
        ],
        draw,
        axes_rect=(0.3, 0.12, 0.6, 0.62),
        source=SOURCE_B,
    )
    return res


def chart_b6_hypo_parts(fl: dict, path: Path) -> dict:
    """Variant B5 for the rented flat with a mortgage only."""
    k = "byt_hypo_najem"
    e = fl["parts"][k].iloc[-1]
    share = e / e.sum()
    assert share["rent"] < 0.5 < share["growth"], "title: growth made most of the profit"
    res: dict = {}

    def draw(ax) -> None:
        res.update(_parts_bars(ax, fl, [k]))

    social_image(
        path,
        (
            f"Byt na hypotéku: {pct(share['growth'], digits=0)} zisku udělal",
            f"růst ceny, nájem jen {pct(share['rent'], digits=0)}",
        ),
        [
            f"Odkud se vzal zisk z 1 mil. Kč vloženého v {qlabel(START_Q)} jako 20% akontace na pronajatý byt",
            f"za {mil(fl['big'], 0)} (varianta B), stav v {qlabel(END_Q)}. Procenta = podíl na zisku celkem",
            "(hodnota dnes − vložený 1 mil. Kč), části se sčítají na 100 %.",
            f"Nájem {pct(GROSS_YIELD)} ceny ročně (1 měsíc prázdný), náklady vlastníka {pct(OWNER_COST, digits=0)} ceny ročně.",
            "Splacená jistina není náklad — zůstává v hodnotě bytu, počítají se jen úroky.",
            "U zisku celkem průměrné roční zhodnocení vkladu 1 mil. Kč (složeně, jako ve variantě E).",
        ],
        draw,
        axes_rect=(0.3, 0.2, 0.6, 0.5),
        source=SOURCE_B,
    )
    return res


NEG, POS = "#e34948", "#2a78d6"  # diverging pair: topping up / surplus


def chart_b7_hypo_time(fl: dict, path: Path) -> dict:
    """Rented flat with a mortgage over time: monthly rent vs. installment + costs (the gap is the
    cash flow), and below the value of the investment from the 1 mil. Kč put in to today."""
    k = "byt_hypo_najem"
    m = fl["months"][1:]
    x = m.to_timestamp()
    rent = fl["rent_big"].loc[m]
    pay = fl["sched"]["payment"].loc[m]
    out_ = pay + fl["costs_big"].loc[m]
    cf = fl["cf"][k].loc[m]
    neg = cf < 0
    last_neg = cf.index[neg][-1]
    assert neg.iloc[0] and not neg.loc[last_neg + 1 :].any(), "title: negative, then positive"
    years_neg = round((last_neg - m[0]).n / 12)
    topped_up, surplus = -cf[neg].sum(), cf[~neg].sum()
    debt = fl["sched"]["balance"].reindex(fl["months"]).fillna(fl["big"] * 0.8)
    price = fl["equity"][k] + debt
    wealth = fl["wealth"][k]
    xa = fl["months"].to_timestamp()

    def end_labels(ax, labels: list[tuple[float, str]], gap: float) -> None:
        ys = _spread([y for y, _ in labels], gap)
        shift = ax.transData + ScaledTranslation(10 / 72, 0, ax.figure.dpi_scale_trans)
        for (_, text), ly in zip(labels, ys, strict=True):
            ax.text(xa[-1], ly, text, transform=shift, va="center", fontsize=11, color=INK,
                    linespacing=1.3)  # fmt: skip

    def panel_title(ax, text: str) -> None:
        ax.text(0.06, 1.08, text, fontsize=14, weight="bold", color=INK,
                transform=blended_transform_factory(ax.figure.transFigure, ax.transAxes))  # fmt: skip

    def time_axis(ax) -> None:
        _clean(ax)
        ax.tick_params(labelsize=11)
        ax.set_xlim(xa[0], xa[-1])
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))

    def flows(ax) -> None:
        ax.fill_between(x, rent, out_, where=(out_ >= rent).values, interpolate=True, color=NEG,
                        alpha=0.35, lw=0)  # fmt: skip
        ax.fill_between(x, rent, out_, where=(rent >= out_).values, interpolate=True, color=POS,
                        alpha=0.35, lw=0)  # fmt: skip
        ax.plot(x, out_.values, color=INK2, lw=2.2)
        ax.plot(x, pay.values, color=INK2, lw=1.4, ls=":")
        ax.plot(x, rent.values, color=PARTS["rent"][1], lw=2.6)
        time_axis(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f} tis."))
        lo, hi = pay.min(), max(rent.max(), out_.max())
        ax.set_ylim(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.05)
        # Cash-flow callouts in the empty space above the deficit and below the surplus.
        ax.text(xa[0] + pd.Timedelta(days=200), hi - (hi - lo) * 0.12,
                f"Doplácelo se {years_neg} let:\nzpočátku {kc(-cf.iloc[0])} měsíčně,\n"
                f"celkem {kc(topped_up)}", va="top", fontsize=11.5, color=NEG, weight="bold",
                linespacing=1.35)  # fmt: skip
        ax.text(pd.Timestamp("2025-06-01"), pay.iloc[-1] + (hi - lo) * 0.1,
                f"Pak v plusu: dnes {kc(cf.iloc[-1])}\nměsíčně, celkem {kc(surplus)}", ha="right",
                va="bottom", fontsize=11.5, color=POS, weight="bold", linespacing=1.35)  # fmt: skip
        end_labels(ax, [(rent.iloc[-1], f"Nájem {kc(rent.iloc[-1])}"),
                        (out_.iloc[-1], f"Splátka + náklady\n{kc(out_.iloc[-1])}"),
                        (pay.iloc[-1], f"z toho splátka\n{kc(pay.iloc[-1])}")],
                   gap=(hi - lo) * 0.2)  # fmt: skip
        panel_title(ax, "Měsíčně: nájem proti splátce a nákladům (rozdíl = cashflow)")

    def value(ax) -> None:
        ax.plot(xa, price.values, color=PARTS["growth"][1], lw=1.8)
        ax.plot(xa, debt.values, color=NEG, lw=1.8, ls="--")
        ax.plot(xa, wealth.values, color=INK, lw=2.8, solid_capstyle="round")
        for xi, yi in ((xa[0], wealth.iloc[0]), (xa[-1], wealth.iloc[-1])):
            ax.plot([xi], [yi], "o", color=INK, ms=9, mec=SURFACE, mew=2, zorder=5, clip_on=False)
        ax.annotate(f"vklad {mil(OWN)}", (xa[0], wealth.iloc[0]),
                    xytext=(xa[0] + pd.Timedelta(days=90), price.max() * 0.19), va="center",
                    fontsize=11.5, weight="bold", color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 1})  # fmt: skip
        time_axis(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(_mil_axis))
        ax.set_ylim(0, price.max() * 1.08)
        end_labels(ax, [(price.iloc[-1], f"Cena bytu {mil(price.iloc[-1])}"),
                        (wealth.iloc[-1], f"Hodnota investice\n{mil(wealth.iloc[-1])}"),
                        (debt.iloc[-1], f"Dluh {mil(debt.iloc[-1])}")],
                   gap=price.max() * 0.13)  # fmt: skip
        panel_title(ax, "Hodnota investice: cena bytu − dluh + cashflow")

    def draw(ax) -> None:
        ax.set_axis_off()
        flows(ax.inset_axes([0, 0.56, 1, 0.38]))
        value(ax.inset_axes([0, 0, 1, 0.38]))

    sched = fl["sched"]
    social_image(
        path,
        (
            f"Prvních {years_neg} let se k bytu doplácelo,",
            f"z milionu je dnes {mil(wealth.iloc[-1])}",
        ),
        [
            f"Pronajatý byt za {mil(fl['big'], 0)} koupený v {qlabel(START_Q)} s 80% hypotékou, vklad 1 mil. Kč (varianta B).",
            f"Splátka {kc(sched['payment'].iloc[0])} (sazba {pct(sched['rate'].iloc[0])}, od 2021 {pct(sched['rate'].iloc[-1])}) zůstává skoro stejná, nájem roste",
            f"s nájemným v HICP. Nájem {pct(GROSS_YIELD)} ceny z roku 2016 ročně (1 měsíc prázdný), náklady",
            f"{pct(OWNER_COST, digits=0)} ceny ročně, rostou s inflací. Hodnota investice = cena bytu (index ČSÚ) − dluh",
            "+ součet cashflow (doplatky odečtené, přebytky přičtené bez úroku).",
        ],
        draw,
        axes_rect=(0.1, 0.13, 0.6, 0.62),
        source=SOURCE_B,
    )
    return {"years_neg": years_neg, "last_neg": last_neg, "topped_up": topped_up,
            "surplus": surplus, "wealth": wealth.iloc[-1]}  # fmt: skip


def chart_c_regions(regions: dict, path: Path) -> dict:
    rows = []
    for reg, label in (("mimo", "mimo Prahu"), ("cr", "průměr ČR"), ("praha", "Praha")):
        f = regions[reg]["final"]
        for key in FLATS:
            rows.append((f"{STRATEGIES[key][0]} – {label}", f[key], STRATEGIES[key][1]))
    f = regions["cr"]["final"]
    rows += [(STRATEGIES[k_][0], f[k_], STRATEGIES[k_][1]) for k_ in ("zlato", "sp500")]
    df = pd.DataFrame(rows, columns=["label", "value", "color"]).sort_values("value")

    def draw(ax) -> None:
        y = np.arange(len(df))
        ax.barh(y, df["value"], color=df["color"], height=0.68)
        ax.set_yticks(y, df["label"], fontsize=12)
        ax.axvline(OWN, color=MUTED, lw=1.4, ls="--")
        ax.text(OWN, len(df) - 0.35, f" vloženo {mil(OWN)}", color=INK2, fontsize=10.5,
                va="bottom")  # fmt: skip
        for i, v in enumerate(df["value"]):
            ax.text(v + df["value"].max() * 0.01, i, mil(v), va="center", fontsize=11, color=INK2)
        _clean(ax, "x")
        ax.xaxis.set_major_formatter(FuncFormatter(_mil_axis))
        ax.set_xlim(0, df["value"].max() * 1.17)
        ax.set_ylim(-0.6, len(df) - 0.1)

    m, p, cr = (regions[r]["final"]["byt_hypo_najem"] for r in ("mimo", "praha", "cr"))
    assert m > cr > p, "title claims regions outside Prague did best with a mortgage"
    growth = {r: regions[r]["fl"]["equity"]["byt_najem"].iloc[-1] / OWN for r in regions}
    social_image(
        path,
        ("Na hypotéku vydělaly nejvíc", "byty mimo Prahu"),
        [
            f"Hodnota 1 mil. Kč vloženého jednorázově v {qlabel(START_Q)}, stav v {qlabel(END_Q)}.",
            "Byty pronajaté; u bytů cena − dluh + nájem − náklady − splátky. Nájem jako % ceny",
            f"všude stejný ({pct(GROSS_YIELD)}). Byty mimo Prahu zdražily {times(growth['mimo'])}, pražské {times(growth['praha'])}.",
            f"Hypotéka ten rozdíl znásobila: {mil(m)} proti {mil(p)}.",
        ],
        draw,
        axes_rect=(0.42, 0.17, 0.5, 0.6),
        source=SOURCE_B,
    )
    return {"df": df}


def chart_d_irr(base: dict, path: Path) -> dict:
    r = base["irr"].sort_values()
    lump = base["lump"]
    infl = (lump["inflace"].iloc[-1] / lump["inflace"].iloc[0]) ** 0.1 - 1
    assert r["sporici"] < infl < r.drop("sporici").min(), (
        "subtitle claims only savings lost to inflation"
    )

    def draw(ax) -> None:
        y = np.arange(len(r))
        colors = [STRATEGIES["byt_najem" if k == EMPTY else k][1] for k in r.index]
        bars = ax.barh(y, r.values * 100, color=colors, height=0.6)
        bars[list(r.index).index(EMPTY)].set(hatch="///", edgecolor=SURFACE, linewidth=0)
        ax.set_yticks(
            y, [D_LABELS[k] if k in D_LABELS else STRATEGIES[k][0] for k in r.index], fontsize=13
        )
        for i, v in enumerate(r.values):
            ax.text(v * 100 + 0.3, i, pct(v), va="center", fontsize=13, color=INK, weight="bold")
        ax.axvline(infl * 100, color=MUTED, lw=1.4, ls="--")
        ax.text(infl * 100, len(r) - 0.45, f" inflace {pct(infl)} ročně (HICP)", color=INK2,
                fontsize=11, va="bottom")  # fmt: skip
        _clean(ax, "x")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} %"))
        ax.set_xlim(0, r.max() * 100 * 1.18)
        ax.set_ylim(-0.6, len(r) - 0.1)

    social_image(
        path,
        (
            f"Pronajatý byt na hypotéku vynášel {pct(r['byt_hypo_najem'], digits=0)} ročně,",
            f"akcie i zlato kolem {pct((r['sp500'] + r['zlato']) / 2, digits=0)}",
        ),
        [
            f"Průměrné roční zhodnocení 1 mil. Kč vloženého jednorázově, {qlabel(START_Q)} → {qlabel(END_Q)}.",
            "U bytů včetně nájmu, nákladů a splátek: doplatky z vlastní kapsy jako další vklad, přebytky",
            "jako výběr (vnitřní výnosové procento, IRR). U ostatních průměrný roční růst hodnoty.",
            f"Nájem {pct(GROSS_YIELD)} ceny ročně (1 měsíc prázdný), náklady {pct(OWNER_COST, digits=0)} ceny i u nepronajatého bytu.",
            "Spořicí účet: sazby ČNB nových vkladů do 1 roku, po 15% dani. Spořák jako jediný",
            "prohrál s inflací (ta byla tak vysoko hlavně kvůli letům 2022–23 s inflací přes 10 %).",
        ],
        draw,
        axes_rect=(0.3, 0.18, 0.62, 0.54),
        source=SOURCE_B,
    )
    return {"irr": r, "infl": infl}


def chart_e_value(base: dict, path: Path) -> dict:
    """Variant D's bars (incl. the empty flat), but valued as in B: yearly return of the final value."""
    fl, lump = base["fl"], base["lump"]
    v = pd.concat([base["final"], fl["wealth"][[EMPTY]].iloc[-1]])
    py = v.map(lambda x: per_year(x, fl["months"])).sort_values()
    v = v[py.index]
    infl = lump["inflace"].iloc[-1]
    infl_y = per_year(infl, fl["months"])
    assert py.idxmax() == "byt_hypo_najem" and py.idxmin() == "sporici", "title claims"
    assert py["sporici"] < infl_y < py.drop("sporici").min(), (
        "subtitle claims only savings lost to inflation"
    )

    def draw(ax) -> None:
        y = np.arange(len(py))
        colors = [STRATEGIES["byt_najem" if k == EMPTY else k][1] for k in py.index]
        bars = ax.barh(y, py.values * 100, color=colors, height=0.6)
        bars[list(py.index).index(EMPTY)].set(hatch="///", edgecolor=SURFACE, linewidth=0)
        ax.set_yticks(
            y, [D_LABELS[k] if k in D_LABELS else STRATEGIES[k][0] for k in py.index], fontsize=13
        )
        dx = py.max() * 100 * 0.015
        for i, (p, val) in enumerate(zip(py.values, v.values, strict=True)):
            x = max(p, infl_y) * 100 + dx  # clear of the inflation line
            ax.text(x, i + 0.03, pct(p), va="bottom", fontsize=13, color=INK, weight="bold")
            ax.text(x, i - 0.03, f"1 mil. → {mil(val)}", va="top", fontsize=11, color=INK2)
        ax.axvline(infl_y * 100, color=MUTED, lw=1.4, ls="--")
        ax.text(infl_y * 100, len(py) - 0.45, f" inflace {pct(infl_y)} ročně (HICP)", color=INK2,
                fontsize=11, va="bottom")  # fmt: skip
        _clean(ax, "x")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f} %"))
        ax.set_xlim(0, py.max() * 100 * 1.3)
        ax.set_ylim(-0.6, len(py) - 0.1)

    social_image(
        path,
        (
            f"Pronajatý byt na hypotéku vydělával {pct(py['byt_hypo_najem'], digits=0)} ročně,",
            f"akcie {pct(py['sp500'], digits=0)}, spořicí účet {pct(py['sporici'], digits=0)}",
        ),
        [
            f"Průměrný roční zisk 1 mil. Kč vloženého jednorázově, {qlabel(START_Q)} → {qlabel(END_Q)}",
            "(složeně, jako úrok z úroku). Počítáno z konečné hodnoty jako ve variantě B: u bytů",
            "cena − dluh + nájem − náklady − splátky, doplatky z vlastní kapsy odečtené.",
            f"Nájem {pct(GROSS_YIELD)} ceny ročně (1 měsíc prázdný), náklady {pct(OWNER_COST, digits=0)} ceny i u nepronajatého bytu.",
            "Spořicí účet: sazby ČNB nových vkladů do 1 roku, po 15% dani. Spořák jako jediný",
            "prohrál s inflací (ta byla tak vysoko hlavně kvůli letům 2022–23 s inflací přes 10 %).",
        ],
        draw,
        axes_rect=(0.3, 0.18, 0.62, 0.54),
        source=SOURCE_B,
    )
    return {"value": v, "per_year": py, "infl": infl}


def chart_savings(d: Data, path: Path) -> None:
    """Methodology chart for the report: household deposit rates used for the savings account."""
    m = d.dep.loc["2015-01":, ["term_1y", "overnight"]]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
    x = m.index.to_timestamp()
    ax.plot(
        x,
        m["term_1y"],
        color=STRATEGIES["sporici"][1],
        lw=2,
        label="nové vklady do 1 roku (základ)",
    )
    ax.plot(x, m["overnight"], color=MUTED, lw=2, ls="--", label="vklady na požádání (dolní odhad)")
    _clean(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Úrokové sazby vkladů domácností v Kč (ČNB), před daní", fontsize=10.5, loc="left",
                 color=INK2)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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


def post_texts(base: dict, regions: dict, rob: pd.DataFrame, k: dict, infl: pd.DataFrame) -> dict:
    fb, lump, w, irr_ = base["fl"], base["lump"], base["final"], base["irr"]
    cf, tot = k["B2"]["cf"], k["B2"]["tot"]
    m, p = regions["mimo"], regions["praha"]
    infl_y = (lump["inflace"].iloc[-1] / OWN) ** 0.1 - 1
    hicp = infl["HICP (Eurostat/ČSÚ)"]
    calm = hicp.drop(["5/2021 → 5/2022", "5/2022 → 5/2023"])
    calm_y = (1 + calm).prod() ** (1 / len(calm)) - 1
    irr_hypo = rob["IRR Pronajatý byt s hypotékou"]
    v, py = k["E"]["value"], k["E"]["per_year"]
    e3h, e3c = k["B3"]["end"]["byt_hypo_najem"], k["B3"]["end"]["byt_najem"]
    s5h, s5c = k["B5"]["share"]["byt_hypo_najem"], k["B5"]["share"]["byt_najem"]
    net = fb["parts"]["byt_hypo_najem"].sum(axis=1)
    peak = net.loc[:"2023-12"].idxmax()
    dip = net[peak] - net.loc[peak:"2024-12"].min()
    return {
        "B": (
            f"Milion korun v roce 2016. Kam s ním? Dnes by z něj bylo: pronajatý byt s 80% hypotékou {mil(w['byt_hypo_najem'])}, akcie S&P 500 {mil(w['sp500'])}, pronajatý byt bez hypotéky {mil(w['byt_najem'])}, zlato {mil(w['zlato'])} a spořicí účet {mil(w['sporici'])}.\n\n"
            f"U bytů je všechno započtené: cena bytu, zbývající dluh, nájem, náklady i splátky. Nájem z bytu na hypotéku za {mil(fb['big'], 0)} zpočátku na splátku a náklady úplně nestačil (doplácelo se do {k['B2']['last_neg'].strftime('%m/%Y')}), za 10 let ale po zaplacení splátek a nákladů zbylo {mil(tot['byt_hypo_najem'])}.\n\n"
            f"Spořicí účet jako jediný prohrál s inflací: vynášel po dani {pct(irr_['sporici'])} ročně, ceny rostly o {pct(infl_y)} ročně. Na stejný nákup jako za milion v roce 2016 dnes potřebujete {mil(lump['inflace'].iloc[-1])}.\n\n"
            "Bez hypotéky skončily byt, akcie i zlato podobně. Rozdíl udělala levná hypotéka — a ta dnes stojí dvakrát víc."
        ),
        "B2": (
            f"Byt na hypotéku vypadá v grafech skvěle. Jak ale vypadala peněženka měsíc po měsíci?\n\n"
            f"Pronajatý byt s 80% hypotékou (cena {mil(fb['big'], 0)}): nájem po odečtení prázdného měsíce {kc(fb['rent_big'].iloc[1])} měsíčně, splátka {kc(fb['sched']['payment'].iloc[0])}, náklady (fond oprav, pojištění, daň, opravy) {kc(fb['costs_big'].iloc[1])}. Na začátku tedy cashflow {kc(cf['byt_hypo_najem'].iloc[0])} měsíčně"
            f", od {(k['B2']['last_neg'] + 1).strftime('%m/%Y')} každý měsíc v plusu. Nájem rostl, splátka ne, a dnes byt vydělává {kc(cf['byt_hypo_najem'].iloc[-1])} měsíčně. Za 10 let celkem {mil(tot['byt_hypo_najem'])}.\n\n"
            f"Pronajatý byt bez hypotéky (za 1 mil. Kč) byl v plusu od začátku, dnes {kc(cf['byt_najem'].iloc[-1])} měsíčně, za 10 let {mil(tot['byt_najem'])}.\n\n"
            "Pozor na dnešek: s hypotékou kolem 5 % a nižším výnosem z nájmu by stejný byt začínal s cashflow hluboko v minusu. Než koupíte, spočítejte si, kolik budete měsíčně doplácet."
        ),
        "B3": (
            f"Odkud se bere zisk z pronajatého bytu? U bytu s 80% hypotékou (milion z roku 2016 jako akontace) za 10 let: růst ceny +{mil(e3h['growth'])}, nájem +{mil(e3h['rent'])}, náklady vlastníka {mil(e3h['costs'])}, úroky hypotéky {mil(e3h['interest'])}. Zisk celkem {mil(e3h.sum())}.\n\n"
            f"Nájem tedy zaplatil náklady i úroky a ještě zbylo {mil(e3h['rent'] + e3h['costs'] + e3h['interest'])}. Hlavní práci ale udělal růst ceny — a ten jistý není: když ceny bytů v letech 2022–23 klesly, zisk z bytu na hypotéku se na čas snížil o {mil(dip)}. Nájem přitom přibýval každý měsíc.\n\n"
            f"Bez hypotéky (byt za milion): růst ceny +{mil(e3c['growth'])}, nájem +{mil(e3c['rent'])}, náklady {mil(e3c['costs'])}, zisk {mil(e3c.sum())}. Poměr růstu ceny a nájmu je stejný — hypotéka ho jen znásobí, protože růst ceny dostáváte z celé ceny bytu, ne jen z vlastních peněz.\n\n"
            "Pro pronajímatele: nájem je stabilní, ale menší část výsledku. O tom, jak investice dopadne, rozhoduje hlavně to, co udělá cena bytu."
        ),
        "B4": (
            f"Milion v roce 2016 jako 20% akontace na pronajatý byt za {mil(fb['big'], 0)}. Co z něj za 10 let bylo: vloženo 1,0 mil. Kč, růst ceny bytu +{mil(e3h['growth'])}, nájem +{mil(e3h['rent'])}, náklady vlastníka {mil(e3h['costs'])}, úroky hypotéky {mil(e3h['interest'])}. Dnes {mil(OWN + e3h.sum())}.\n\n"
            f"Z příjmů bytu (růst ceny + nájem) tvořil nájem jen {pct(e3h['rent'] / (e3h['growth'] + e3h['rent']), digits=0)}. Zbytek je růst ceny, který se projeví až při prodeji (nebo jako vyšší hodnota zástavy).\n\n"
            f"Úroky za 10 let stály {mil(-e3h['interest'])} — díky sazbě kolem {pct(fb['sched']['rate'].iloc[0], digits=0)}. Splátky byly vyšší, ale splacená jistina není náklad: zůstává v hodnotě bytu jako menší dluh.\n\n"
            f"Bez hypotéky (byt za milion): růst ceny +{mil(e3c['growth'])}, nájem +{mil(e3c['rent'])}, náklady {mil(e3c['costs'])}, dnes {mil(OWN + e3c.sum())}."
        ),
        "B5": (
            f"Kolik ze zisku pronajatého bytu přinesl nájem? Méně, než by asi čekal leckterý pronajímatel. U bytu koupeného v roce 2016 na 80% hypotéku za milion vlastních peněz udělal růst ceny {pct(s5h['growth'], digits=0)} zisku (+{mil(e3h['growth'])}) a nájem {pct(s5h['rent'], digits=0)} (+{mil(e3h['rent'])}). Náklady vlastníka ubraly {pct(-s5h['costs'], digits=0)}, úroky hypotéky {pct(-s5h['interest'], digits=0)}. Zisk celkem {mil(e3h.sum())}.\n\n"
            f"Bez hypotéky (byt za milion) je to podobné: růst ceny {pct(s5c['growth'], digits=0)} zisku (+{mil(e3c['growth'])}), nájem {pct(s5c['rent'], digits=0)} (+{mil(e3c['rent'])}), náklady {pct(s5c['costs'], digits=0)}. Zisk celkem {mil(e3c.sum())}.\n\n"
            f"Ve stejném měřítku je dobře vidět, co udělala hypotéka: nájem i růst ceny se počítají z ceny bytu, a ta byla s hypotékou pětkrát vyšší. Úroky přitom stály jen {mil(-e3h['interest'])}.\n\n"
            "Nájem je jistý a pravidelný, ale o výsledku rozhoduje cena bytu. Kdo kupuje kvůli nájmu, měl by počítat i s tím, že cena 10 let neporoste."
        ),
        "B6": (
            f"Milion vložený v roce 2016 jako 20% akontace na pronajatý byt za {mil(fb['big'], 0)} je dnes {mil(OWN + e3h.sum())}, v průměru {pct(per_year(OWN + e3h.sum(), fb['months']))} ročně. Odkud se zisk {mil(e3h.sum())} vzal?\n\n"
            f"Růst ceny bytu {pct(s5h['growth'], digits=0)} zisku (+{mil(e3h['growth'])}), nájem {pct(s5h['rent'], digits=0)} (+{mil(e3h['rent'])}). Náklady vlastníka ubraly {pct(-s5h['costs'], digits=0)} ({mil(e3h['costs'])}), úroky hypotéky {pct(-s5h['interest'], digits=0)} ({mil(e3h['interest'])}).\n\n"
            "Nájem tedy pokryl náklady i úroky, ale velký zisk přišel z ceny bytu — a ten uvidíte až při prodeji. Hypotéka ho znásobila, protože růst ceny dostáváte z celé ceny bytu, ne jen z vlastních peněz.\n\n"
            "Kdo dnes kupuje kvůli nájmu, měl by si spočítat, jak by investice dopadla, kdyby cena bytu nerostla."
        ),
        "B7": (
            f"Jak vypadá pronajatý byt na hypotéku měsíc po měsíci? V roce 2016 milion vlastních peněz jako 20% akontace na byt za {mil(fb['big'], 0)}, zbytek hypotéka se splátkou {kc(fb['sched']['payment'].iloc[0])}.\n\n"
            f"Na začátku nájem na splátku a náklady nestačil: {k['B7']['years_neg']} let se doplácelo, zpočátku {kc(-cf['byt_hypo_najem'].iloc[0])} měsíčně, celkem {kc(k['B7']['topped_up'])}. Splátka ale zůstávala skoro stejná, zatímco nájem rostl. Od {(k['B7']['last_neg'] + 1).strftime('%m/%Y')} je byt v plusu a dnes zbývá {kc(cf['byt_hypo_najem'].iloc[-1])} měsíčně. Přebytky zatím dohromady {kc(k['B7']['surplus'])}.\n\n"
            f"Mezitím se z milionu stalo {mil(k['B7']['wealth'])}: cena bytu vzrostla na {mil(fb['equity']['byt_hypo_najem'].iloc[-1] + fb['sched']['balance'].iloc[-1])} a dluh klesl na {mil(fb['sched']['balance'].iloc[-1])}.\n\n"
            "Hypotéka na pronajatý byt tak ze začátku stojí peníze z vlastní kapsy. Než koupíte, spočítejte si, kolik let a kolik měsíčně budete doplácet — a jestli to vydržíte, i kdyby nájem rostl pomaleji."
        ),
        "C": (
            f"Byty mimo Prahu zdražily za 10 let {times(m['fl']['equity']['byt_najem'].iloc[-1] / OWN)}, pražské „jen“ {times(p['fl']['equity']['byt_najem'].iloc[-1] / OWN)}.\n\n"
            f"S 80% hypotékou se ten rozdíl zvětší. Kdo v roce 2016 dal milion jako akontaci na pronajatý byt mimo Prahu, má dnes po odečtení dluhu a se započtením nájmu, nákladů a splátek {mil(m['final']['byt_hypo_najem'])}. V Praze {mil(p['final']['byt_hypo_najem'])}. Stejný milion v akciích S&P 500 je dnes {mil(w['sp500'])}.\n\n"
            "Nájem přitom počítáme všude stejně vysoký vzhledem k ceně bytu. Ve skutečnosti vynáší pražský byt z nájmu méně, takže rozdíl by byl spíš ještě větší.\n\n"
            "Regiony vyšly lépe, protože začínaly z nízkých cen. Za ten růst ale mohly i levné úvěry a nedostatek bytů — dnes jsou ceny i mimo Prahu jinde a výnos z nájmu nižší. Jako pronajímatel se při další koupi ptejte, jestli bude cena i nájem v daném městě dál růst."
        ),
        "D": (
            f"Milion vložený v roce 2016 jako akontace na pronajatý byt s 80% hypotékou vynášel v průměru {pct(irr_['byt_hypo_najem'])} ročně. Akcie S&P 500 {pct(irr_['sp500'])}, zlato {pct(irr_['zlato'])}, pronajatý byt bez hypotéky {pct(irr_['byt_najem'])}, nepronajatý byt bez hypotéky {pct(irr_[EMPTY])}, spořicí účet po dani {pct(irr_['sporici'])}.\n\n"
            f"Inflace byla {pct(k['D']['infl'])} ročně. Vysoké číslo dělají hlavně roky 2022 a 2023 (od května do května {pct(hicp['5/2021 → 5/2022'])} a {pct(hicp['5/2022 → 5/2023'])}); bez nich by to bylo {pct(calm_y)} ročně. Spořicí účet tak jako jediný reálně ztrácel.\n\n"
            f"U bytů počítáme nájem {pct(GROSS_YIELD)} ceny ročně (jeden měsíc v roce prázdný), náklady vlastníka {pct(OWNER_COST, digits=0)} ceny ročně a doplatky na splátky jako další vklad. Výsledek na předpokladech moc nestojí: při nájmu 4 % by byt na hypotéku vynášel {pct(irr_hypo['Hrubý nájem 4 % ceny'])}, při 6 % {pct(irr_hypo['Hrubý nájem 6 % ceny'])}, s desetiletou fixací {pct(irr_hypo['Fixace 5–10 let na celých 10 let'])}.\n\n"
            "Je to ale ohlédnutí za výjimečnou dekádou. Stejný výpočet pro příštích 10 let začíná s dvojnásobnou sazbou a nižším výnosem z nájmu. Páka zvětšuje zisky i ztráty."
        ),
        "E": (
            f"Milion vložený v roce 2016 do pronajatého bytu s 80% hypotékou vydělával v průměru {pct(py['byt_hypo_najem'])} ročně. Dnes by z něj bylo {mil(v['byt_hypo_najem'])}.\n\n"
            f"Akcie S&P 500 vydělávaly {pct(py['sp500'])} ročně ({mil(v['sp500'])}), pronajatý byt bez hypotéky {pct(py['byt_najem'])} ({mil(v['byt_najem'])}), zlato {pct(py['zlato'])} ({mil(v['zlato'])}), nepronajatý byt bez hypotéky {pct(py[EMPTY])} ({mil(v[EMPTY])}) a spořicí účet po dani jen {pct(py['sporici'])} ({mil(v['sporici'])}).\n\n"
            f"Pronájem přidal bytu bez hypotéky proti prázdnému bytu {pct(py['byt_najem'] - py[EMPTY], digits=1).replace(' %', ' procentního bodu')} ročně. Většinu zisku udělal růst cen bytů, ne nájem. Hypotéka ho pak znásobila: s ní se kupoval byt za {mil(fb['big'], 0)} místo za 1 mil. Kč.\n\n"
            f"Pozor, inflace: to, co stálo milion v roce 2016, stojí dnes {mil(k['E']['infl'])}. Inflace byla {pct(per_year(k['E']['infl'], fb['months']))} ročně, spořicí účet tak reálně ztrácel a i ostatní zisky jsou v dnešních korunách menší, než vypadají."
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
    d, base, regions, rob, posts, k, imgs, raw, chk, infl, mdd = (
        ctx[x] for x in ("d", "base", "regions", "rob", "posts", "k", "imgs", "raw", "chk", "infl",
                         "mdd")
    )  # fmt: skip
    fb, lump, f = base["fl"], base["lump"], base["final"]
    sched = fb["sched"]
    big = fb["big"]
    growth = fb["equity"]["byt_najem"].iloc[-1] / OWN
    esc = html.escape
    names = {k_: v[0] for k_, v in STRATEGIES.items()}
    trio = f[["sp500", "zlato", "byt_najem"]] / OWN

    yearly = base["paths"].assign(inflace=lump["inflace"])
    yearly = yearly[yearly.index.month == fb["months"][0].month]
    yearly.index = yearly.index.year
    yearly = yearly.rename(columns={**names, "inflace": "Inflace (HICP)"})

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
        ("ČNB — úrokové sazby nových vkladů domácností v CZK (MIR), měsíčně, přes ECB Data Portal: vklady splatné na požádání "
         "(overnight) a vklady se splatností do 1 roku (term_1y); G, H = 1–2 roky, nad 2 roky", URL_DEP, d.dep,
         d.dep.loc["2016-05":, "term_1y"]),
        ("ČSÚ — Indexy spotřebitelských cen (CPI, 010022), celkem, měsíčně, průměr 2015 = 100 (kontrola inflace HICP)",
         URL_CPI, d.cpi.to_frame("CPI"), d.cpi.loc["2016-05":]),
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

    rob_fmt = {c: (pct if c.startswith("IRR") else mil) for c in rob.columns}
    m, p = regions["mimo"], regions["praha"]

    body = f"""
<h1>10 let investování: milion v bytě, akciích, zlatě nebo na spořicím účtu</h1>
<p class="lead">Hlavní závěr: <b>1 mil. Kč vložený v {qlabel(START_Q)} jako akontace na pronajatý byt s 80% hypotékou
je dnes {mil(f["byt_hypo_najem"])}</b> (po odečtení dluhu, včetně nájmu, nákladů a splátek). Stejný milion v akciích S&amp;P 500
je {mil(f["sp500"])}, v pronajatém bytě bez hypotéky {mil(f["byt_najem"])}, ve zlatě {mil(f["zlato"])} a na spořicím účtu
{mil(f["sporici"])}. Bez páky skončily byt, akcie i zlato podobně ({times(trio.min())}–{times(trio.max())} za 10 let).
Rozhodla páka levné hypotéky. Spořicí účet jako jediný prohrál s inflací.</p>

<p class="note">Všechny varianty počítají s jednorázovým vkladem 1 mil. Kč v květnu 2016, bez dalších vkladů. U bytu s hypotékou
se doplatky z vlastní kapsy (když nájem nepokryl splátku a náklady) odečítají od výsledku (B, C), resp. počítají jako další vklad (D).
Proč 2Q 2016 → 2Q 2026: ČSÚ zveřejňuje ceny bytů čtvrtletně, poslední je 2. čtvrtletí 2026 (vydáno 14. 9. 2026).
Měsíční řady (akcie, zlato, kurz, sazby) bereme v prostředním měsíci čtvrtletí, tj. květen 2016 → květen 2026 (121 měsíců, 120 splátek).</p>

<h2>Hlavní obrázek</h2>
<div class="main">{_img(imgs["B"])}</div>
<p>Jako hlavní doporučuji <b>variantu B</b>: odpovídá na jednoduchou otázku „co by dnes bylo z milionu z roku 2016“, ukazuje vývoj
v čase i konečné částky všech pěti možností včetně inflace. Pro pronajímatele je překvapivé, jak velký náskok měl byt na hypotéku,
a pro běžné střadatele, že spořicí účet reálně ztrácel. Varianta B2 je přirozený navazující příspěvek o tom, kolik se po cestě doplácelo.</p>

<h2>Varianta B — jednorázový milion: pronajaté byty (s hypotékou i bez), akcie, zlato, spořicí účet</h2>
{_img(imgs["B"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Vklad:</b> jednorázově 1 mil. Kč v květnu 2016, dál už nic. Akcie, zlato a spořicí účet: 1 mil. Kč × (hodnota v květnu 2026 / hodnota v květnu 2016).</li>
<li><b>Byt bez hypotéky:</b> byt za 1 mil. Kč (podíl průměrného bytu podle indexu ČSÚ ČR). <b>Byt s hypotékou:</b> 1 mil. Kč = 20% akontace → byt za {mil(big)},
úvěr {mil(big * 0.8)} na 30 let, anuita. Sazba = průměr ČNB za {qlabel(START_Q)} pro nové hypotéky s fixací 1–5 let ({pct(sched["rate"].iloc[0], digits=2)}),
po 5 letech refixace na průměr za {qlabel(START_Q + 20)} ({pct(sched["rate"].iloc[-1], digits=2)}). Splátka {kc(sched["payment"].iloc[0])}, po refixaci {kc(sched["payment"].iloc[-1])}.
Na konci: byt {mil(big * growth)}, dluh {mil(sched["balance"].iloc[-1])}, zaplacené úroky celkem {mil(sched["interest"].sum())}.</li>
<li><b>Pronajatý byt:</b> hrubý nájem {pct(GROSS_YIELD)} ceny z roku 2016 ročně, 1 měsíc v roce prázdný (čistý příjem {pct(GROSS_YIELD * (1 - VACANCY))} ceny), nájem dál roste podle indexu
skutečného nájemného HICP (CP041, +{pct(d.hicp["CP041"].loc[mid_month(END_Q)] / d.hicp["CP041"].loc[mid_month(START_Q)] - 1, digits=0)} za 10 let). U bytu s hypotékou to dělá
{kc(fb["rent_big"].iloc[1])} měsíčně na začátku a {kc(fb["rent_big"].iloc[-1])} na konci.</li>
<li><b>Náklady vlastníka</b> (fond oprav, pojištění, daň z nemovitosti, drobné opravy): {pct(OWNER_COST, digits=0)} ceny z roku 2016 ročně, rostou s inflací HICP.
U bytu s hypotékou {kc(fb["costs_big"].iloc[1])} měsíčně na začátku.</li>
<li><b>Výsledek bytů</b> = cena bytu podle indexu ČSÚ − zbývající dluh + součet měsíčního cashflow (nájem − náklady − splátka). Záporné cashflow (doplatky z vlastní kapsy) se
odečítá bez úroku, kladné se hromadí jako hotovost bez úroku. Výnos se zohledněním toho, kdy se doplácelo, ukazuje varianta D (IRR).</li>
<li><b>Spořicí účet:</b> měsíční úročení průměrnou sazbou ČNB pro nové vklady domácností se splatností do 1 roku, po 15% srážkové dani (viz sekce Spořicí účet).</li>
<li><b>Inflace:</b> HICP Česko — kolik korun je potřeba dnes na koupi toho, co stálo 1 mil. Kč v roce 2016 (podrobně v sekci Inflace).</li>
<li><b>Kurzový efekt:</b> S&amp;P 500 v dolarech {times(k["B"]["end"]["sp500_usd"])}, v korunách {times(k["B"]["end"]["sp500"])};
zlato v dolarech {times(k["B"]["end"]["zlato_usd"])}, v korunách {times(k["B"]["end"]["zlato"])}.
Bez dividend by S&amp;P 500 v korunách dalo {times(k["B"]["end"]["sp500_price"])}.</li>
<li><b>Riziko po cestě (max. propad v měsíčních datech):</b> S&amp;P 500 v Kč {pct(mdd["sp500"], digits=0)}, zlato {pct(mdd["zlato"], digits=0)},
byt {pct(mdd["byt"], digits=0)}. Pozor: index bytů je čtvrtletní průměr transakcí, propady proto vypadají menší, než by byla
skutečná kolísavost ceny jednoho bytu.</li>
<li><b>Roční zhodnocení</b> u popisků = (konečná hodnota / 1 mil. Kč)<sup>1/10</sup> − 1 (víc ve variantě E).</li>
<li><b>Graf:</b> čárový; modrá byt s hypotékou, tyrkysová byt bez hypotéky, oranžová akcie, žlutá zlato, fialová spořicí účet, šedá čárkovaná inflace.
Barvy jsou stejné ve všech variantách. Popisky u konců čar seřazené podle konečné hodnoty.</li>
<li><b>Kontrola:</b> tabulka robustnosti níže — náskok pronajatého bytu s hypotékou drží ve všech testovaných variantách.</li>
</ul>
<h3>Výše nájmu — zdroj</h3>
<p>Hrubý roční nájem = cena bytu / {dec(PRICE_TO_RENT, 1)} = <b>{pct(GROSS_YIELD, digits=2)} ceny bytu</b>. Poměr {dec(PRICE_TO_RENT, 1)} je ukazatel
<b>„Poměr cena bytu / roční nájemné (dle IRI)“</b> ČNB, řádek TN.4 v tabulce indikátorů finanční stability (část 1, str. 160 PDF)
<a href="{esc(URL_ZFS)}">Zprávy o finanční stabilitě 2015/2016</a> (ČNB, zveřejněno 14. 6. 2016). Hodnoty: 2010 21,9; 2011 21,7; 2012 21,3; 2013 21,6;
2014 20,7; 2015 21,5; poslední údaj z roku 2016 21,9 — ten používáme, protože byt kupujeme v květnu 2016.
ČNB ho počítá z nabídkových cen bytů a nabídkového tržního nájemného, které publikuje Institut regionálních informací (IRI), průměr ČR.
Stejný ukazatel ČNB používá jako „výnos z pronájmu bytu“ (roční nájemné / cena bytu 68 m²; ZFS 2015/2016, graf II.24).</p>
<ul>
<li><b>Proč nabídkové ceny:</b> skutečné nájmy i ceny bývají o něco nižší než nabídkové; poměr se tím z velké části vyruší.</li>
<li><b>Průměr ČR:</b> v Praze je výnos z nájmu nižší, v regionech vyšší. Ve variantě C proto počítáme se stejným nájmem jako % ceny všude (zjednodušení).</li>
<li><b>Vývoj nájmu po roce 2016:</b> index skutečného nájemného HICP (CP041). Ten zahrnuje i regulované a dlouhodobé nájmy, tržní nájmy rostly spíš rychleji — nájem je tedy spíš podhodnocený.</li>
<li><b>Neobsazenost a náklady:</b> 1 měsíc v roce bez nájmu a {pct(OWNER_COST, digits=0)} ceny ročně na fond oprav, pojištění, daň z nemovitosti a opravy jsou naše předpoklady (citlivost v tabulce robustnosti).</li>
<li>Daň z příjmu z nájmu není započtena (s 30% výdajovým paušálem by brala asi 10 % hrubého nájmu).</li>
</ul>
{_post(posts["B"])}

<h2>Varianta B2 — měsíční cashflow bytů z varianty B</h2>
{_img(imgs["B2"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Cashflow</b> = čistý nájem − náklady vlastníka − splátka hypotéky (jen s hypotékou), měsíc po měsíci, stejné předpoklady jako B.</li>
<li><b>Součty za 10 let:</b> {"; ".join(f"{esc(STRATEGIES[c][0])} {mil(k['B2']['tot'][c])}" for c in FLATS)}.</li>
<li><b>Skoky:</b> refixace hypotéky v roce 2021 změnila splátku; nájem roste podle měsíčního indexu nájemného, náklady podle inflace.</li>
<li><b>Graf:</b> čárový, Kč měsíčně, nulová čára zvýrazněná. Stejné barvy jako v B.</li>
</ul>
{_post(posts["B2"])}

<h2>Varianta B3 — odkud se bere zisk z bytu, v čase</h2>
{_img(imgs["B3"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Stejné byty a předpoklady jako B</b> (s hypotékou a bez), zisk = výsledek bytu z B − vložený 1 mil. Kč, načítaný od května 2016.</li>
<li><b>Rozklad zisku:</b> růst ceny = cena<sub>0</sub> × (index<sub>t</sub> / index<sub>0</sub> − 1); nájem = součet vybraného nájmu; náklady vlastníka = součet nákladů;
úroky = součet zaplacených úroků. Splátka = úrok + jistina, a splacená jistina zvyšuje čistou hodnotu bytu (menší dluh) — proto se mezi náklady počítají jen úroky.
Součet čtyř částí se přesně rovná zisku z B (kontrolováno v kódu).</li>
<li><b>Konec období:</b> {"; ".join(f"{esc(STRATEGIES[c][0])}: " + ", ".join(f"{esc(PARTS[pp][0].lower())} {mil(k['B3']['end'][c][pp])}" for pp in PARTS if k["B3"]["end"][c][pp] != 0) for c in FLATS)}.</li>
<li><b>Graf:</b> dva panely (každý vlastní měřítko, aby byl vidět i byt bez hypotéky), skládané plochy — nad nulou příjmy, pod nulou výdaje, černá čára = zisk celkem.
Barvy částí v pořadí referenční palety; nesouvisí s barvami strategií v ostatních variantách.</li>
<li><b>Pozor:</b> nájem a náklady jsou sečtené bez úroku a bez daně z příjmu; růst ceny je „na papíře“ (bez nákladů na prodej).</li>
</ul>
{_post(posts["B3"])}

<h2>Varianta B4 — odkud se bere zisk z bytu, bez časové osy</h2>
{_img(imgs["B4"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Stejný rozklad jako B3</b>, jen stav na konci období jako vodopádový graf: vloženo 1 mil. Kč → + růst ceny → + nájem → − náklady → − úroky → hodnota dnes (= výsledek z B).</li>
<li><b>Graf:</b> vodorovné plovoucí sloupce, tenké spojnice mezi koncem a začátkem navazujících sloupců; vložený a konečný sloupec šedě (barvy částí jako v B3). Každý panel vlastní měřítko.</li>
</ul>
{_post(posts["B4"])}

<h2>Varianta B5 — růst ceny vs. nájem ve stejném měřítku</h2>
{_img(imgs["B5"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Stejná čísla jako B3/B4</b> (stav na konci období), jen jiné zobrazení: vodorovné sloupce od nuly, růst ceny a nájem přímo pod sebou, obě varianty bytu
na jedné ose (stejné měřítko), aby byl vidět i efekt hypotéky.</li>
<li><b>Relativní hodnota = podíl na zisku celkem</b> (hodnota dnes − vložený 1 mil. Kč); části se sčítají na 100 %, náklady a úroky jsou záporné.
Podíl vůči vloženému milionu jsme nepoužili — byl by číselně stejný jako částka v mil. Kč. Podíl na příjmech (bez nákladů) je v textu B4.</li>
<li><b>Podíly:</b> {"; ".join(f"{esc(STRATEGIES[c][0])}: " + ", ".join(f"{esc(PARTS[pp][0].lower())} {pct(k['B5']['share'][c][pp], digits=0)}" for pp in PARTS if k["B5"]["end"][c][pp] != 0) for c in FLATS)}.</li>
<li><b>Zisk celkem</b> navíc s průměrným ročním zhodnocením vloženého 1 mil. Kč: (hodnota dnes / 1 mil.)<sup>1/10</sup> − 1, stejně jako ve variantě E
({"; ".join(f"{esc(STRATEGIES[c][0])} {pct(k['E']['per_year'][c])}" for c in FLATS)}).</li>
<li><b>Graf:</b> barvy částí jako v B3/B4, zisk celkem šedě.</li>
</ul>
{_post(posts["B5"])}

<h2>Varianta B6 — zdroje zisku jen pro pronajatý byt s hypotékou</h2>
{_img(imgs["B6"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Stejný graf jako B5</b>, jen pro pronajatý byt s 80% hypotékou: části zisku jako vodorovné sloupce s podílem na zisku celkem,
u zisku celkem průměrné roční zhodnocení vkladu (jako E). Stejná čísla a předpoklady jako B.</li>
</ul>
{_post(posts["B6"])}

<h2>Varianta B7 — pronajatý byt s hypotékou v čase: cashflow a hodnota investice</h2>
{_img(imgs["B7"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Jen pronajatý byt s 80% hypotékou</b>, stejná čísla a předpoklady jako B a B2.</li>
<li><b>Horní graf (Kč měsíčně):</b> nájem (po odečtení prázdného měsíce), výdaje = splátka + náklady vlastníka, tečkovaně samotná splátka.
Plocha mezi nájmem a výdaji je měsíční cashflow: červeně doplácení (do {k["B7"]["last_neg"].strftime("%m/%Y")}, celkem {kc(k["B7"]["topped_up"])}),
modře přebytek (celkem {kc(k["B7"]["surplus"])}). Skok v roce 2021 = refixace hypotéky.</li>
<li><b>Dolní graf (mil. Kč):</b> cena bytu podle indexu ČSÚ, zbývající dluh a hodnota investice = cena − dluh + součet cashflow (= výsledek bytu z B).
Začíná vkladem 1 mil. Kč, končí na {mil(k["B7"]["wealth"])}.</li>
<li><b>Proč dva grafy:</b> měsíční toky (tisíce Kč) a hodnota bytu (miliony Kč) mají jiné měřítko; jedna osa by toky zploštila, dvě osy by byly zavádějící.</li>
<li><b>Barvy:</b> nájem oranžově a cena bytu modře jako v B3–B6, doplácení/přebytek jako divergentní pár červená/modrá.</li>
</ul>
{_post(posts["B7"])}

<h2>Varianta C — Praha vs. zbytek republiky</h2>
{_img(imgs["C"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Data:</b> stejné jako B, jen index cen bytů ČSÚ pro Prahu, resp. ČR bez Prahy (ČR celkem = vážený průměr 23,1 : 76,9).</li>
<li><b>Výpočet:</b> jednorázově 1 mil. Kč, pronajaté byty se stejnými předpoklady jako B (nájem {pct(GROSS_YIELD)} ceny, náklady {pct(OWNER_COST, digits=0)} ceny,
stejná hypotéka — splátka nezávisí na regionu, protože vždy jde o 1 mil. Kč = 20 % ceny). Akcie a zlato jsou stejné jako v B.</li>
<li><b>Ceny:</b> byty mimo Prahu zdražily {times(m["fl"]["equity"]["byt_najem"].iloc[-1] / OWN)}, pražské {times(p["fl"]["equity"]["byt_najem"].iloc[-1] / OWN)},
průměr ČR {times(growth)}.</li>
<li><b>Graf:</b> vodorovné sloupce seřazené podle konečné hodnoty, barva = typ strategie, čárkovaně vložený 1 mil. Kč.</li>
<li><b>Pozor:</b> výnos z nájmu je v Praze typicky nižší než v regionech, takže s regionálním nájmem by rozdíl ve prospěch regionů byl spíš větší.
Index „ČR bez Prahy“ pokrývá celou republiku, jednotlivá města se mohou výrazně lišit.</li>
</ul>
{_post(posts["C"])}

<h2>Varianta D — roční výnos vlastních peněz</h2>
{_img(imgs["D"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Akcie, zlato, spořicí účet:</b> průměrný roční růst 1 mil. Kč za 10 let, (hodnota<sub>konec</sub> / 1 mil.)<sup>1/10</sup> − 1.</li>
<li><b>Pronajaté byty:</b> vnitřní výnosové procento (IRR) z měsíčních toků: −1 mil. Kč na začátku, každý měsíc cashflow bytu (doplatek = další vklad,
přebytek = výběr), na konci + čistá hodnota bytu (cena − dluh). Převedeno na roční (p. a.). U akcií a zlata dává stejný výpočet přesně průměrný roční růst,
takže čísla jsou srovnatelná.</li>
<li><b>Předpoklady bytů</b> jako ve variantě B: hrubý nájem {pct(GROSS_YIELD)} ceny ročně, 1 měsíc v roce prázdný, náklady vlastníka {pct(OWNER_COST, digits=0)} ceny ročně.</li>
<li><b>Nepronajatý byt bez hypotéky</b> (jen v D): byt za 1 mil. Kč, který stojí prázdný (nebo v něm bydlí majitel — ušetřený nájem nepočítáme).
Toky: −1 mil. Kč, každý měsíc náklady vlastníka ({pct(OWNER_COST, digits=0)} ceny ročně, rostou s inflací, celkem {mil(-base["fl"]["cf"][EMPTY].sum())}), na konci cena bytu {mil(base["fl"]["equity"][EMPTY].iloc[-1])}.
V grafu stejná barva jako pronajatý byt bez hypotéky, odlišená šrafováním. Rozdíl proti pronajatému bytu ({pct(base["irr"]["byt_najem"] - base["irr"][EMPTY], digits=1).replace(" %", " p. b.")}) je čistý přínos nájmu.</li>
<li><b>Citlivost:</b> IRR pronajatého bytu s hypotékou v tabulce robustnosti níže (sloupec „IRR Pronajatý byt s hypotékou“) — od
{pct(rob["IRR Pronajatý byt s hypotékou"].min())} do {pct(rob["IRR Pronajatý byt s hypotékou"].max())}.</li>
<li><b>Inflace:</b> průměrná roční inflace HICP za stejné období (geometrický průměr; podrobně v sekci Inflace).</li>
<li><b>Graf:</b> vodorovné sloupce seřazené podle výnosu.</li>
</ul>
{_post(posts["D"])}

<h2>Varianta E — konečná hodnota a průměrné roční zhodnocení</h2>
{_img(imgs["E"])}
<h3>Jak graf vznikl</h3>
<ul>
<li><b>Stejné strategie jako D</b> (včetně nepronajatého bytu bez hypotéky), ale <b>výpočet jako B</b>: konečná hodnota 1 mil. Kč v {qlabel(END_Q)},
u bytů cena − dluh + součet měsíčního cashflow (doplatky odečtené bez úroku, přebytky jako hotovost bez úroku).</li>
<li><b>Průměrné roční zhodnocení</b> = (konečná hodnota / 1 mil. Kč)<sup>1/10</sup> − 1 (geometrický průměr). U akcií, zlata a spořicího účtu
je to totéž číslo jako v D. U bytů se liší: D (IRR) bere v úvahu, kdy se doplácelo a kdy přebytky přišly, E bere jen konečný stav jako v B
(peníze z cashflow bez úroku). Rozdíl: {"; ".join(f"{esc(D_LABELS[c].replace(chr(10), ' '))} E {pct(k['E']['per_year'][c])} vs. D {pct(base['irr'][c])}" for c in (*FLATS, EMPTY))}.</li>
<li><b>Hodnoty:</b> {"; ".join(f"{esc(D_LABELS.get(c, STRATEGIES.get(c, (c,))[0]).replace(chr(10), ' '))} {mil(k['E']['value'][c])} ({pct(k['E']['per_year'][c])} ročně)" for c in k["E"]["value"].index[::-1])}.</li>
<li><b>Inflace:</b> svislá čárkovaná čára = průměrná roční inflace HICP; 1 mil. Kč z roku 2016 dnes odpovídá {mil(k["E"]["infl"])}.</li>
<li><b>Graf:</b> vodorovné sloupce = průměrné roční zhodnocení v %, seřazené podle něj; pod procentem konečná hodnota. Barvy jako v D (nepronajatý byt šrafovaně).</li>
</ul>
{_post(posts["E"])}

<h2>Vývoj po letech (varianta B)</h2>
{_table(yearly, {c: mil for c in yearly.columns})}
<p class="note">Hodnoty vždy v květnu daného roku, u bytů po odečtení dluhu a včetně kumulovaného cashflow.</p>

<h2>Robustnost</h2>
<p>Každý řádek je celý výpočet znovu se změněným jedním předpokladem: konečná hodnota 1 mil. Kč v každé strategii a roční výnos (IRR, viz D)
pro byty a akcie. Řádky „Byty jen Praha / ČR bez Prahy“ jsou varianta C.</p>
{_table(rob, rob_fmt)}
<img src="data:image/png;base64,{base64.b64encode(imgs["R"].read_bytes()).decode()}" alt="" style="max-width:720px">
<h2>Inflace — odkud je číslo a proč je tak vysoké</h2>
<p><b>Zdroj:</b> harmonizovaný index spotřebitelských cen (HICP) za Česko z Eurostatu, datová sada
<a href="{esc(URL_HICP)}">prc_hicp_minr</a>, položka TOTAL, index 2015 = 100. Data do Eurostatu dodává ČSÚ.
Kontrola: národní index spotřebitelských cen ČSÚ (<a href="{esc(URL_CPI)}">CPI, datová sada 010022</a>), ze kterého ČSÚ počítá oficiální „míru inflace“.</p>
<p><b>Výsledek:</b> HICP {dec(d.hicp.loc[mid_month(START_Q), "TOTAL"])} (5/2016) → {dec(d.hicp.loc[mid_month(END_Q), "TOTAL"])} (5/2026), tj. +{pct(lump["inflace"].iloc[-1] / OWN - 1, digits=1)} za 10 let,
<b>{pct(k["D"]["infl"])} ročně</b>. ČSÚ CPI za stejné období, které má k dispozici (5/2016 → {esc(str(d.cpi.index[-1]))}): {pct(chk["cpi_y"])} ročně, HICP za totéž období {pct(chk["hicp_y"])} ročně — oba indexy se shodují.</p>
<p><b>Proč tolik:</b> průměr táhnou nahoru dva roky energetické krize. Od května 2021 do května 2023 vzrostly ceny o {pct(chk["shock"], digits=0)}.
Bez těchto dvou let by průměr byl {pct(chk["calm_y"])} ročně (blízko 2% cíle ČNB), medián ročních hodnot je {pct(infl["HICP (Eurostat/ČSÚ)"].median())}.
Pocit, že „inflace je normálně kolem 2–3 %“, je správný — jenže v tomto desetiletí byla dvě léta výjimečná a ta se do průměru propisují.</p>
{_table(infl, {c: (lambda v: pct(v, signed=True)) for c in infl.columns})}
<p class="note">Sloupec „CPI ČSÚ, průměr roku konce“ je oficiální roční míra inflace ČSÚ (průměr 12 měsíců proti předchozím 12) za rok, ve kterém období končí — to jsou čísla, která se uvádí ve zprávách
(např. 2022: 15,1 %, 2023: 10,7 %). ČSÚ CPI je v datové sadě zatím do {esc(str(d.cpi.index[-1]))}.</p>

<h2>Spořicí účet — jaká sazba</h2>
<p>ČNB nezveřejňuje samostatnou průměrnou sazbu spořicích účtů v dlouhé řadě. Použili jsme statistiku úrokových sazeb MFI (ČNB, distribuuje ECB):</p>
<ul>
<li><b>Základ: nové vklady domácností se splatností do 1 roku</b> (průměr {pct(d.dep.loc["2016-05":"2026-05", "term_1y"].mean() / 100, digits=2)} p. a. za období, minimum {pct(d.dep.loc["2016-05":"2026-05", "term_1y"].min() / 100, digits=2)}, maximum {pct(d.dep.loc["2016-05":"2026-05", "term_1y"].max() / 100, digits=2)}).
Odpovídá tomu, co dostal aktivní střadatel, který přecházel za lepším spořicím účtem nebo termínovaným vkladem (v roce 2023 nejlepší spořicí účty kolem 6 %, v polovině 2025 kolem 3 %).</li>
<li><b>Dolní odhad: vklady splatné na požádání</b> (běžné i spořicí účty dohromady, průměr {pct(d.dep.loc["2016-05":"2026-05", "overnight"].mean() / 100, digits=2)} p. a.). Tak dopadl ten, kdo nechal peníze ležet:
1 mil. Kč → {mil(lump["sporici_bezny"].iloc[-1], 2)}.</li>
<li>Úrok se připisuje měsíčně, z úroku se strhává 15% daň. Výsledek základu: 1 mil. Kč → {mil(lump["sporici"].iloc[-1], 2)} ({pct((lump["sporici"].iloc[-1] / OWN) ** 0.1 - 1, digits=2)} ročně po dani).</li>
</ul>
<img src="data:image/png;base64,{base64.b64encode(imgs["S"].read_bytes()).decode()}" alt="" style="max-width:720px">

<h2>Datové zdroje</h2>
{src_html}
<h3>ČNB — Zpráva o finanční stabilitě 2015/2016, ukazatel cena bytu / roční nájemné (IRI)</h3>
<p><a href="{esc(URL_ZFS)}">{esc(URL_ZFS)}</a><br>Tabulka indikátorů, část 1 (str. 160 PDF), řádek TN.4. Použitá hodnota {dec(PRICE_TO_RENT, 1)}
(poslední údaj 2016) → hrubý výnos z nájmu {pct(GROSS_YIELD, digits=2)}. Hodnota je v kódu jako konstanta; PDF je uložené v surových datech.</p>
<p class="note">Kontrola Shiller vs. FRED: měsíční průměr denních uzávěrek se v období 5/2016–6/2026 liší nejvýš o {pct(chk["fred_diff"], digits=2)}.</p>

<h2>Předpoklady</h2>
<ul>
<li><b>Průměrný byt</b> = index realizovaných cen starších bytů ČSÚ (ČR celkem). Index měří změnu ceny, ne úroveň; částky
jsou proto „na 1 mil. Kč vlastních peněz“. Výsledky jsou lineární — se 2 mil. Kč vlastních by vše bylo dvojnásobné.
Byt za {mil(big)} v roce 2016 byl nadprůměrně drahý, procenta ale platí pro jakoukoli cenu.</li>
<li><b>Hypotéka:</b> 80 % ceny (LTV), 30 let, anuitní splátka, sazba = průměr nových hypoték ČNB s fixací 1–5 let
(nejčastější fixace v roce 2016) za čtvrtletí sjednání, refixace po 5 letech na tehdejší průměr nových hypoték.
Bez poplatků bance, bez pojištění.</li>
<li><b>Jednorázový vklad:</b> každá strategie dostane 1 mil. Kč v květnu 2016 a nic dalšího. Jediné další peníze jsou doplatky
na splátku hypotéky, když nájem nestačil — ty se od výsledku odečítají (B, C), resp. v IRR počítají jako další vklad (D).</li>
<li><b>Nájem a náklady bytu:</b> byty jsou pronajaté, nájem podle ČNB/IRI, neobsazenost a náklady vlastníka jsou naše předpoklady — viz „Výše nájmu — zdroj“.</li>
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
<li>Cashflow bytu: nájem<sub>t</sub> = cena<sub>0</sub> × {dec(GROSS_YIELD * 100, 2)} % / 12 × 11/12 × index nájemného<sub>t</sub> / index<sub>0</sub>; náklady<sub>t</sub> = cena<sub>0</sub> × 1 % / 12 × HICP<sub>t</sub> / HICP<sub>0</sub>;
cashflow = nájem − náklady − splátka. Výsledek bytu = cena<sub>0</sub> × index<sub>t</sub> / index<sub>0</sub> − zůstatek úvěru<sub>t</sub> + Σ cashflow.</li>
<li>Akcie, zlato, spořicí účet: 1 mil. × hodnota<sub>t</sub> / hodnota<sub>0</sub>.</li>
<li>IRR (varianta D): měsíční sazba r řešící Σ CF<sub>t</sub> / (1 + r)<sup>t</sup> = 0 (bisekce), CF = −1 mil. na začátku, měsíční cashflow, + čistá hodnota bytu na konci; roční = (1 + r)<sup>12</sup> − 1.</li>
</ul>

<h2>Postup</h2>
<ol>
<li>Výběr zdrojů: ČSÚ (realizované, ne nabídkové ceny bytů), ČNB (sazby nových hypoték, kurz), Shiller (jediná volně dostupná měsíční řada
S&amp;P 500 s dividendami; ověřeno proti FRED), Světová banka (zlato), Eurostat/ČSÚ (inflace, nájemné), ČNB (sazby vkladů pro spořicí účet), ČSÚ CPI (kontrola inflace).</li>
<li>Stažení skriptem <code>download.py</code> do <code>{esc(str(raw.relative_to(monaco.ROOT)))}</code>.</li>
<li>Zpracování: převod všeho na měsíční řady v Kč, interpolace čtvrtletního indexu bytů.</li>
<li>Výpočet pěti strategií pro jednorázový 1 mil. Kč: hodnota v čase, měsíční cashflow pronajatých bytů, roční výnos (IRR).</li>
<li>Kandidátní grafy B, B2, B3, B4, B5, B6, B7, C, D, E (varianta A se stejnými měsíčními vklady byla vyřazena), kontrola robustnosti (15 variant), kontrola zdrojových dat (Shiller vs. FRED, HICP vs. CPI ČSÚ).</li>
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
<li><b>Riziko a likvidita nejsou v číslech.</b> Byt s 80% hypotékou znamená dluh {mil(big * 0.8)} na 1 mil. Kč vlastních. Kdyby ceny
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

    base = run(d)
    regions = {
        r: (base if r == "cr" else run(d, Scenario(region=r))) for r in ("cr", "praha", "mimo")
    }
    rob = robustness(d)
    infl = inflation_table(d)
    window = pd.period_range("2016-05", "2026-06", freq="M")
    cpi_end = d.cpi.index[-1]
    yrs = (cpi_end - mid_month(START_Q)).n / 12
    hicp = d.hicp["TOTAL"]
    calm = infl["HICP (Eurostat/ČSÚ)"].drop(["5/2021 → 5/2022", "5/2022 → 5/2023"])
    chk = {
        "fred_diff": float((d.spx["P"] / d.fred - 1).reindex(window).abs().max()),
        "cpi_y": (d.cpi[cpi_end] / d.cpi[mid_month(START_Q)]) ** (1 / yrs) - 1,
        "hicp_y": (hicp[cpi_end] / hicp[mid_month(START_Q)]) ** (1 / yrs) - 1,
        "shock": hicp[pd.Period("2023-05", "M")] / hicp[pd.Period("2021-05", "M")] - 1,
        "calm_y": (1 + calm).prod() ** (1 / len(calm)) - 1,
    }
    mdd = {k_: max_drawdown(base["lump"][k_]) for k_ in ("sp500", "zlato", "byt")}

    (out / "data").mkdir()
    base["paths"].to_csv(out / "data" / "strategie_mesicne.csv")
    base["fl"]["cf"].to_csv(out / "data" / "byty_cashflow_mesicne.csv")
    pd.concat(base["fl"]["parts"], axis=1).to_csv(out / "data" / "byty_zdroje_zisku_mesicne.csv")
    base["fl"]["sched"].to_csv(out / "data" / "hypoteka_splatky.csv")
    base["lump"].to_csv(out / "data" / "jednorazove_1mil.csv")
    pd.DataFrame({r: v["final"] for r, v in regions.items()}).to_csv(out / "data" / "regiony.csv")
    base["irr"].to_csv(out / "data" / "rocni_vynos.csv")
    rob.to_csv(out / "data" / "robustnost.csv")
    infl.to_csv(out / "data" / "inflace_po_letech.csv")

    imgs = {
        "B": out / "B_jednorazove_1mil.png",
        "B2": out / "B2_cashflow_bytu.png",
        "B3": out / "B3_zdroje_zisku_v_case.png",
        "B4": out / "B4_zdroje_zisku.png",
        "B5": out / "B5_rust_ceny_vs_najem.png",
        "B6": out / "B6_byt_s_hypotekou_zdroje_zisku.png",
        "B7": out / "B7_byt_s_hypotekou_v_case.png",
        "C": out / "C_praha_vs_regiony.png",
        "D": out / "D_rocni_vynos.png",
        "E": out / "E_hodnota_a_zisk_rocne.png",
        "R": out / "metodika_sazby_hypotek.png",
        "S": out / "metodika_sazby_vkladu.png",
    }
    k = {
        "B": chart_b_lump(base, imgs["B"]),
        "B2": chart_b2_cashflow(base["fl"], imgs["B2"]),
        "B3": chart_b3_parts_time(base["fl"], imgs["B3"]),
        "B4": chart_b4_parts(base["fl"], imgs["B4"]),
        "B5": chart_b5_parts_bars(base["fl"], imgs["B5"]),
        "B6": chart_b6_hypo_parts(base["fl"], imgs["B6"]),
        "B7": chart_b7_hypo_time(base["fl"], imgs["B7"]),
        "C": chart_c_regions(regions, imgs["C"]),
        "D": chart_d_irr(base, imgs["D"]),
        "E": chart_e_value(base, imgs["E"]),
    }
    chart_rates(d, imgs["R"])
    chart_savings(d, imgs["S"])
    posts = post_texts(base, regions, rob, k, infl)
    (out / "texty_prispevku.md").write_text(
        "\n\n".join(f"## Varianta {v}\n\n{t}" for v, t in posts.items()) + "\n", encoding="utf-8"
    )
    write_report(
        out,
        {
            "d": d,
            "base": base,
            "regions": regions,
            "rob": rob,
            "posts": posts,
            "k": k,
            "imgs": imgs,
            "raw": raw,
            "chk": chk,
            "mdd": mdd,
            "infl": infl,
        },
    )
    print(rob.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
