"""Download raw data for the 10-year investment comparison (S&P 500, gold, Czech flats, mortgage).

Writes to data/raw/investment_comparison/<yymmdd_hhmmss>/<source>/.

Usage: uv run scripts/20261030_investment_comparison/download.py
"""

import ssl
import urllib.request
from datetime import datetime

import certifi

import monaco

FIRST_YEAR, LAST_YEAR = 2015, 2026
SOURCES = {
    # ČSÚ: indices of realized prices of older flats (Praha, ČR bez Prahy, ČR), quarterly since
    # 2008, average 2010 = 100. https://csu.gov.cz/produkty/indexy-realizovanych-cen-bytu-2-ctvrtleti-2026
    "csu_014007": {
        "01400726q2s.xlsx": "https://csu.gov.cz/docs/107508/4981ed9b-235a-b4cd-e9a3-fc78659d0745/01400726q2s.xlsx",
        "01400726q2-n.xlsx": "https://csu.gov.cz/docs/107508/5c7735c6-06bd-56a5-e0bc-7ff5c158390c/01400726q2-n.xlsx",
    },
    # ČNB MFI interest rate statistics as distributed by the ECB Data Portal: CZK loans to
    # households for house purchase, new business, by initial rate fixation (F, I, O, P, A).
    "cnb_ecb_mir": {
        "mir_cz_house_purchase.csv": "https://data-api.ecb.europa.eu/service/data/MIR/M.CZ.B.A2C..R.A.2250.CZK.N?format=csvdata",
    },
    # ČNB daily exchange rates, one file per year.
    "cnb_fx": {
        f"kurzy_{y}.txt": "https://www.cnb.cz/cs/financni-trhy/devizovy-trh/kurzy-devizoveho-trhu/"
        f"kurzy-devizoveho-trhu/rok.txt?rok={y}"
        for y in range(FIRST_YEAR, LAST_YEAR + 1)
    },
    # Robert Shiller's monthly S&P Composite price, dividends, earnings and CPI.
    # https://shillerdata.com/
    "shiller": {
        "ie_data.xls": "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/1449043e-a1ed-4e2e-8e9c-7b9a2a064493/ie_data.xls",
    },
    # FRED daily S&P 500 close (last 10 years), used to cross-check Shiller's monthly price.
    "fred": {
        "SP500.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",
    },
    # World Bank Commodity Price Data (Pink Sheet), monthly averages incl. gold in USD/troy oz.
    # https://www.worldbank.org/en/research/commodity-markets
    "worldbank_cmo": {
        "CMO-Historical-Data-Monthly.xlsx": "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx",
    },
    # Eurostat HICP for Czechia (data supplied by ČSÚ), monthly index 2015 = 100, COICOP 2018:
    # all items (TOTAL) and actual rentals for housing (CP041). The older prc_hicp_midx dataset
    # stopped at 2025-12.
    "eurostat_hicp": {
        "hicp_cz.csv": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/prc_hicp_minr/"
        f"M.I15.TOTAL+CP041.CZ/?format=SDMX-CSV&startPeriod={FIRST_YEAR}-01",
    },
}


def main() -> None:
    root = (
        monaco.DATA_DIR / "raw" / "investment_comparison" / datetime.now().strftime("%y%m%d_%H%M%S")
    )
    # The ECB endpoint's chain is not in the macOS Python default trust store; use certifi's.
    ctx = ssl.create_default_context(cafile=certifi.where())
    for source, files in SOURCES.items():
        target = root / source
        target.mkdir(parents=True)
        for name, url in files.items():
            print(f"{url} -> {target / name}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
                (target / name).write_bytes(r.read())
    print(f"Saved to {root}")


if __name__ == "__main__":
    main()
