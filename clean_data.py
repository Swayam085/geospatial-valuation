"""
clean_data.py

Day 2 (Week 1): clean the raw King County housing data, handle price outliers,
validate coordinates, and save a processed version ready for Week 2 feature
engineering and Week 3 graph construction.

Run with:
    python clean_data.py

Expects the raw file at: data/raw/kc_house_data.csv
Writes cleaned output to: data/processed/kc_house_data_clean.csv
                           data/processed/kc_house_data_clean.geojson
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths -- change these if your folder names are different
# ---------------------------------------------------------------------------
RAW_PATH = "data/raw/kc_house_data.csv"
PROCESSED_DIR = "data/processed"
CSV_OUT = os.path.join(PROCESSED_DIR, "kc_house_data_clean.csv")
GEOJSON_OUT = os.path.join(PROCESSED_DIR, "kc_house_data_clean.geojson")

# King County's real geographic bounding box -- anything outside this is bad data
LAT_MIN, LAT_MAX = 47.0, 47.9
LON_MIN, LON_MAX = -122.6, -121.2


def load_raw_data(path):
    """Load the raw CSV and print a quick first look, so you can see what you're working with."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {path}")

    print("\nMissing values per column (only columns with >0 shown):")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing if len(missing) > 0 else "  none")
    return df


def fix_dtypes(df):
    """Parse the date column properly and treat zipcode as a category, not a number."""
    df = df.copy()
    if "date" in df.columns:
        # King County's date column looks like '20141013T000000'
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype(str)
    return df


def drop_duplicate_sales(df):
    """
    Some houses were sold more than once in this dataset (same 'id').
    We keep the most recent sale per house and report how many rows this affected.
    """
    df = df.copy()
    n_before = len(df)
    if "date" in df.columns:
        df = df.sort_values("date")
    df = df.drop_duplicates(subset="id", keep="last")
    n_after = len(df)
    print(f"\nDropped {n_before - n_after} duplicate sales (kept the most recent sale per house)")
    return df


def validate_coordinates(df):
    """
    Flag rows whose lat/long fall outside King County's real bounding box.
    We report these rather than silently ignoring them -- bad geocoding is
    worth knowing about.
    """
    df = df.copy()
    in_bounds = df["lat"].between(LAT_MIN, LAT_MAX) & df["long"].between(LON_MIN, LON_MAX)
    n_bad = int((~in_bounds).sum())
    print(f"\n{n_bad} rows have coordinates outside King County's bounding box")
    if n_bad > 0:
        print("Dropping these rows -- they can't be placed on a real King County map")
    return df[in_bounds].copy()


def cap_price_outliers(df, lower_pct=0.01, upper_pct=0.99):
    """
    Cap extreme prices at the 1st/99th percentile instead of deleting the rows.
    Adds a 'price_was_capped' flag so these rows are still visible later --
    the brief specifically wants you to study price behavior in
    high-variance/gentrifying areas, so don't erase the evidence.
    """
    df = df.copy()
    lower = df["price"].quantile(lower_pct)
    upper = df["price"].quantile(upper_pct)

    df["price_was_capped"] = (df["price"] < lower) | (df["price"] > upper)
    n_capped = int(df["price_was_capped"].sum())

    df["price"] = df["price"].clip(lower=lower, upper=upper)

    print(f"\nPrice capping range: [{lower:,.0f}, {upper:,.0f}]")
    print(f"{n_capped} rows ({n_capped / len(df):.1%}) were capped, not deleted")
    return df


def build_geodataframe(df):
    """Convert to a GeoDataFrame with a proper Point geometry column -- the
    canonical spatial object that Week 3's graph construction builds on."""
    geometry = [Point(xy) for xy in zip(df["long"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    print(f"\nBuilt GeoDataFrame with CRS: {gdf.crs}")
    return gdf


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_raw_data(RAW_PATH)
    df = fix_dtypes(df)
    df = drop_duplicate_sales(df)
    df = validate_coordinates(df)
    df = cap_price_outliers(df)
    gdf = build_geodataframe(df)

    # Plain CSV -- the easiest format for Week 2 feature engineering
    gdf.drop(columns="geometry").to_csv(CSV_OUT, index=False)
    # GeoJSON -- keeps the geometry, handy if Week 3 wants to load a spatial file directly
    gdf.to_file(GEOJSON_OUT, driver="GeoJSON")

    print(f"\nSaved cleaned CSV to:     {CSV_OUT}")
    print(f"Saved cleaned GeoJSON to: {GEOJSON_OUT}")
    print(f"\nFinal shape: {gdf.shape[0]:,} rows x {gdf.shape[1]} columns")


if __name__ == "__main__":
    main()