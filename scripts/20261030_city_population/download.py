"""Download raw ČSÚ population data and ČÚZK boundaries for the city population analysis.

Writes to data/raw/city_population/<yymmdd_hhmmss>/<source>/.

Usage: uv run scripts/20261030_city_population/download.py
"""

import urllib.parse
import urllib.request
import zipfile
from datetime import datetime

import monaco

BASE = "https://csu.gov.cz"
RUIAN = (
    "https://ags.cuzk.gov.cz/arcgis/rest/services/RUIAN/Prohlizeci_sluzba_nad_daty_RUIAN/MapServer"
)
SOURCES = {
    # Population by sex in municipalities as of 31 Dec, yearly since 2000 (permanent + long-term
    # residents). https://csu.gov.cz/produkty/obyvatelstvo-k-3112-podle-pohlavi-v-obcich
    "csu_130149": [
        "/docs/107508/caaa6ad1-ddd1-c0df-1683-3c23ccf1f0c7/130149-25data052025.zip",
        "/docs/107508/59a2a4da-5451-6714-8cc1-0c68742e1e12/130149-25schema052025.json",
    ],
    # Population and houses in municipalities by census, 1869-2011, recalculated to 2015 borders.
    # https://csu.gov.cz/obyvatelstvo-a-domy-v-obcich-podle-vysledku-scitani-od-roku-1869
    "csu_170240": [
        "/docs/107516/0c242a34-ca97-0209-581f-dde18a20a25f/170240-17data.zip",
        "/docs/107516/eb201fae-a878-a6be-6d42-a9959ec773aa/170240-17schema.json",
    ],
}
# ČÚZK RÚIAN map service: current boundaries of municipalities (layer 12) and regions (layer 17)
# in S-JTSK (EPSG:5514), generalized to 200 m, as Esri JSON.
RUIAN_LAYERS = {"obce.json": 12, "kraje.json": 17}
RUIAN_QUERY = {
    "where": "1=1",
    "outFields": "kod,nazev",
    "returnGeometry": "true",
    "maxAllowableOffset": "200",
    "geometryPrecision": "0",
    "outSR": "5514",
    "f": "json",
}


def main() -> None:
    root = monaco.DATA_DIR / "raw" / "city_population" / datetime.now().strftime("%y%m%d_%H%M%S")
    for source, paths in SOURCES.items():
        target = root / source
        target.mkdir(parents=True)
        for path in paths:
            dest = target / path.rsplit("/", 1)[1]
            print(f"{BASE}{path} -> {dest}")
            urllib.request.urlretrieve(f"{BASE}{path}", dest)
            if dest.suffix == ".zip":
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(target)
    target = root / "cuzk_ruian"
    target.mkdir()
    for name, layer in RUIAN_LAYERS.items():
        url = f"{RUIAN}/{layer}/query?{urllib.parse.urlencode(RUIAN_QUERY)}"
        print(f"{url} -> {target / name}")
        urllib.request.urlretrieve(url, target / name)
    print(f"Saved to {root}")


if __name__ == "__main__":
    main()
