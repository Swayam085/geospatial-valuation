"""
build_map.py

Day 2 (Week 1): first geographic visualization -- a Folium map showing where
prices are high/low across King County. This lets you *see* spatial pricing
trends before any model exists.

Run with:
    python build_map.py

Expects: data/processed/kc_house_data_clean.csv (created by clean_data.py)
Writes:  outputs/price_map.html  -- open this file in a browser
"""

import os
import pandas as pd
import folium
from folium.plugins import HeatMap

CSV_IN = "data/processed/kc_house_data_clean.csv"
OUT_DIR = "outputs"
MAP_OUT = os.path.join(OUT_DIR, "price_map.html")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_IN)

    # Center the map on the average location in the dataset
    center = [df["lat"].mean(), df["long"].mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    # Heatmap layer weighted by price -- brighter/warmer = more expensive
    heat_data = df[["lat", "long", "price"]].values.tolist()
    HeatMap(heat_data, radius=8, blur=10, max_zoom=13).add_to(m)

    # A sample of individual points you can click for an exact price --
    # capped at 300 so the HTML file doesn't get huge
    sample = df.sample(min(300, len(df)), random_state=0)
    for _, row in sample.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["long"]],
            radius=2,
            color="#444444",
            fill=True,
            fill_opacity=0.4,
            popup=f"${row['price']:,.0f}",
        ).add_to(m)

    m.save(MAP_OUT)
    print(f"Saved interactive map to {MAP_OUT} -- open it in a browser to explore")


if __name__ == "__main__":
    main()