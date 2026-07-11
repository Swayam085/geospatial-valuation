"""
data_loader.py
--------------
Load and validate the raw King County housing dataset.

Owner   : Vrushabh (Geospatial Data Engineer)
Project : Geospatial Valuation via Spatial Embeddings
Week 1 / Day 1
"""

from pathlib import Path
import pandas as pd

# Columns every downstream notebook/script expects to exist
REQUIRED_COLUMNS = [
    "id", "lat", "long", "price", "sqft_living",
    "bedrooms", "bathrooms", "yr_built", "zipcode",
]

# Approximate bounding box for King County, WA — used only as a sanity
# check to catch obviously bad geocoding, not a strict scientific boundary.
KING_COUNTY_BOUNDS = {
    "lat_min": 47.0,
    "lat_max": 47.9,
    "long_min": -122.6,
    "long_max": -121.0,
}


def load_raw_data(filepath: str = "data/raw/kc_house_data.csv") -> pd.DataFrame:
    """
    Load the raw housing dataset from CSV and validate its schema.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file (relative to project root by default).

    Returns
    -------
    pd.DataFrame
        Raw housing dataframe, unmodified apart from being read into memory.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")

    df = pd.read_csv(path)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise if any required column is missing from the dataframe."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Run basic data quality checks: row count, duplicate ids, nulls,
    out-of-range coordinates, and a couple of sanity checks on
    price/sqft.

    Returns
    -------
    dict
        Summary report. Print it or log it, don't act on it silently.
    """
    report = {}

    report["n_rows"] = len(df)
    report["n_duplicate_ids"] = int(df["id"].duplicated().sum())
    report["null_counts"] = df.isnull().sum().to_dict()

    out_of_bounds = df[
        (df["lat"] < KING_COUNTY_BOUNDS["lat_min"])
        | (df["lat"] > KING_COUNTY_BOUNDS["lat_max"])
        | (df["long"] < KING_COUNTY_BOUNDS["long_min"])
        | (df["long"] > KING_COUNTY_BOUNDS["long_max"])
    ]
    report["n_out_of_bounds_coords"] = len(out_of_bounds)

    report["n_invalid_price"] = int((df["price"] <= 0).sum())
    report["n_invalid_sqft"] = int((df["sqft_living"] <= 0).sum())

    return report


def print_quality_report(report: dict) -> None:
    """Pretty-print the dict returned by check_data_quality()."""
    print("Data Quality Report")
    print("--------------------")
    for key, value in report.items():
        if key == "null_counts":
            continue
        print(f"{key}: {value}")
    print("\nNull values per column:")
    for col, cnt in report["null_counts"].items():
        print(f"  {col}: {cnt}")


if __name__ == "__main__":
    data = load_raw_data()
    quality_report = check_data_quality(data)
    print_quality_report(quality_report)