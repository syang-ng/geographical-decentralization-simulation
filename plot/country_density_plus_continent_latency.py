
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os
import pandas as pd
import pycountry
import re


from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from collections import Counter
from geographiclib.geodesic import Geodesic
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm


# -------- continent mapping --------
_CONTINENT_RULES = [
    (r"^us-|^northamerica-", "North America"),
    (r"^southamerica-",      "South America"),
    (r"^europe-",            "Europe"),
    (r"^asia-",              "Asia"),
    (r"^australia-",         "Oceania"),
    (r"^me-",                "Middle East"),
    (r"^africa-",            "Africa"),
]


def to_continent(region: str) -> str:
    for pat, name in _CONTINENT_RULES:
        if re.match(pat, region):
            return name
    return "Other"


def iso2_to_iso3(iso2: str) -> str | None:
    try:
        c = pycountry.countries.get(alpha_2=str(iso2).upper())
        return c.alpha_3 if c else None
    except Exception:
        return None


def load_world_geojson(path: str) -> gpd.GeoDataFrame:
    """
    Load a countries GeoJSON and return a GeoDataFrame with an ISO-3 column named 'iso3'.
    Tested with: https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json
    """
    world = gpd.read_file(path)

    # Try common ISO-3 locations
    if "id" in world.columns:
        world["iso3"] = world["id"].astype(str).str.upper()
    elif "ISO_A3" in world.columns:
        world["iso3"] = world["ISO_A3"].astype(str).str.upper()
    elif "iso_a3" in world.columns:
        world["iso3"] = world["iso_a3"].astype(str).str.upper()
    else:
        raise ValueError(
            f"Cannot find an ISO-3 field. Columns are: {list(world.columns)}. "
            "Expected one of: id, ISO_A3, iso_a3."
        )

    # Drop invalid placeholders if any
    world = world[world["iso3"].notna() & (world["iso3"] != "-99")].copy()

    # Ensure lon/lat CRS
    if world.crs is None:
        world = world.set_crs(epsg=4326)
    else:
        world = world.to_crs(epsg=4326)

    return world


# -------- main figure --------
def figure_country_density_plus_continent_latency(
    regions_csv: str,
    latency_csv: str,
    validators_csv: str,
    world_geojson_path: str,
    outpath: str | None = None,
    fmt: str = "png",
    dpi: int = 300,
    # two separate colormaps (IMPORTANT)
    cmap_country: str = "viridis_r",     # validator density
    cmap_latency: str = "plasma",    # latency arcs
    # styles
    missing_color: str = "#e3e3e3",
    country_edgecolor: str = "gray",
    country_edge_lw: float = 0.3,
    country_alpha: float = 0.45,
    node_size: int = 120,
    label_nodes: bool = True,
    label_arcs: bool = True,
    lw_min: float = 0.8,
    lw_max: float = 4.0,
    highlight_pairs: list | None = None,  # e.g. [("North America","Europe")]
):
    # -------------------------
    # Load world polygons
    # -------------------------
    world = load_world_geojson(world_geojson_path)

    # -------------------------
    # Country density from validators.csv
    # -------------------------
    dfv = pd.read_csv(validators_csv)
    if "country" not in dfv.columns:
        raise ValueError(f"{validators_csv} must contain column 'country' (ISO-2 codes)")

    country_counter = Counter(dfv["country"].astype(str).values)
    df_country = pd.DataFrame(country_counter.most_common(), columns=["iso2", "value"])
    df_country["iso3"] = df_country["iso2"].apply(iso2_to_iso3)
    df_country = df_country[["iso3", "value"]].dropna()
    df_country["iso3"] = df_country["iso3"].astype(str).str.upper()

    # Merge to world
    gdf = world.merge(df_country, how="left", on="iso3")

    # Country color scale
    vv = gdf["value"].dropna()
    if len(vv) == 0:
        raise ValueError("All country 'value' are NaN after merge. Check ISO mapping and geojson ISO-3 field.")
    vmin_c, vmax_c = float(vv.min()), float(vv.max())
    norm_country = LogNorm(vmin=max(vmin_c, 1), vmax=vmax_c)
    sm_country = ScalarMappable(norm=norm_country, cmap=cmap_country)

    # -------------------------
    # Load latency inputs, build continent backbone
    # -------------------------
    reg = pd.read_csv(regions_csv)
    lat = pd.read_csv(latency_csv)

    need_r_cols = {"Region", "Nearest City Latitude", "Nearest City Longitude"}
    if not need_r_cols.issubset(reg.columns):
        raise ValueError(f"{regions_csv} must contain {need_r_cols}")

    need_l_cols = {"sending_region", "receiving_region", "milliseconds"}
    if not need_l_cols.issubset(lat.columns):
        raise ValueError(f"{latency_csv} must contain {need_l_cols}")

    reg["Continent"] = reg["Region"].map(to_continent)
    cont_map = dict(zip(reg["Region"], reg["Continent"]))

    valid = set(reg["Region"])
    lat = lat[lat["sending_region"].isin(valid) & lat["receiving_region"].isin(valid)].copy()
    lat["A"] = lat["sending_region"].map(cont_map)
    lat["B"] = lat["receiving_region"].map(cont_map)

    # undirected continent pairs (median ms)
    lat["pair"] = lat.apply(lambda r: tuple(sorted([r["A"], r["B"]])), axis=1)
    edges = lat.groupby("pair")["milliseconds"].median().reset_index()
    edges[["c1", "c2"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges_inter = edges[edges["c1"] != edges["c2"]].copy()

    # representative coords per continent = mean of member region coords
    rep = (
        reg.groupby("Continent")[["Nearest City Longitude", "Nearest City Latitude"]]
        .mean()
        .rename(columns={"Nearest City Longitude": "lon", "Nearest City Latitude": "lat"})
    )

    used_conts = sorted({c for pair in edges_inter["pair"] for c in pair})
    rep = rep.loc[rep.index.intersection(used_conts)]

    # Latency color scale
    vals = edges_inter["milliseconds"].to_numpy(dtype=float)
    vmin_l, vmax_l = float(np.nanmin(vals)), float(np.nanmax(vals))
    norm_latency = Normalize(vmin=vmin_l, vmax=vmax_l)
    sm_latency = ScalarMappable(norm=norm_latency, cmap=cmap_latency)

    def width_for(ms: float) -> float:
        if vmax_l == vmin_l:
            return (lw_min + lw_max) / 2.0
        t = (ms - vmin_l) / (vmax_l - vmin_l)
        inv = 1.0 - t
        return lw_min + inv * (lw_max - lw_min)

    # -------------------------
    # Plotting setup (Cartopy GeoAxes)
    # -------------------------
    fig, ax = plt.subplots(
        figsize=(16, 10),
        dpi=dpi,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)
    ax.coastlines(linewidth=0.5, color="gray", zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", zorder=0)

    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # -------------------------
    # Draw country shading (Cartopy-safe, independent cmap)
    # -------------------------
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        val = row["value"]
        face = missing_color if pd.isna(val) else sm_country.to_rgba(float(val))
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor=face,
            edgecolor=country_edgecolor,
            linewidth=country_edge_lw,
            alpha=country_alpha,
            zorder=1,
        )

    # -------------------------
    # Draw continent arcs (your great-circle logic) with independent cmap
    # -------------------------
    ARC_LABEL_OFFSETS = {
        ("Europe", "Asia"): (0, 5),
        ("Asia", "Europe"): (0, 5),
        ("Middle East", "Europe"): (-4, -5),
        ("Europe", "Middle East"): (-4, -5),
    }

    geod = Geodesic.WGS84
    highlight_set = set()
    if highlight_pairs:
        highlight_set = {tuple(sorted(p)) for p in highlight_pairs}

    for _, row in edges_inter.iterrows():
        a, b = row["pair"]
        if a not in rep.index or b not in rep.index:
            continue

        lon1, lat1 = rep.loc[a, ["lon", "lat"]]
        lon2, lat2 = rep.loc[b, ["lon", "lat"]]

        line = geod.InverseLine(lat1, lon1, lat2, lon2)
        pts = np.linspace(0, line.s13, 80)
        lats = [line.Position(s)["lat2"] for s in pts]
        lons = [line.Position(s)["lon2"] for s in pts]
        lons = np.rad2deg(np.unwrap(np.deg2rad(lons)))

        ms = float(row["milliseconds"])
        color = sm_latency.to_rgba(ms)
        lw = width_for(ms)
        z = 2

        if tuple(sorted((a, b))) in highlight_set:
            lw *= 1.4
            z = 3

        ax.plot(
            lons, lats,
            color=color,
            linewidth=lw,
            alpha=0.95,
            transform=ccrs.PlateCarree(),
            zorder=z,
        )

        if label_arcs:
            mid = line.Position(line.s13 / 2.0)
            dx, dy = ARC_LABEL_OFFSETS.get((a, b), (0, 0))
            ax.text(
                mid["lon2"] + dx,
                mid["lat2"] + dy,
                f"{int(round(ms))}ms",
                fontsize=18,
                ha="center",
                va="center",
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=4,
            )

    # -------------------------
    # Nodes + continent labels
    # -------------------------
    ax.scatter(
        rep["lon"], rep["lat"],
        s=node_size,
        facecolors="white",
        edgecolors="black",
        linewidths=1.4,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    if label_nodes:
        HALO = [pe.withStroke(linewidth=1.0, foreground="white")]
        for cont, r in rep.iterrows():
            ax.annotate(
                cont,
                xy=(r["lon"], r["lat"]),
                xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                xytext=(0, 18),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="bold",
                path_effects=HALO,
                zorder=7,
                clip_on=False,
            )

    # -------------------------
    # Two colorbars (paper layout)
    # -------------------------
    sm_latency.set_array(np.array([vmin_l, vmax_l], dtype=float))
    sm_country.set_array(np.array([vmin_c, vmax_c], dtype=float))

    # Latency: right
    cbar_lat = fig.colorbar(
        sm_latency,
        ax=ax,
        orientation="vertical",
        fraction=0.028,
        pad=0.02,
        shrink=0.58,
    )
    cbar_lat.set_label("Median Latency (ms)", fontsize=20)
    cbar_lat.ax.tick_params(labelsize=20)

    # Country density: bottom
    cbar_country = fig.colorbar(
        sm_country,
        ax=ax,
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
    )
    cbar_country.set_label("")
    pos = cbar_country.ax.get_position()
    cbar_country.ax.set_position([
        pos.x0 - 0.2,
        pos.y0,
        pos.width,
        pos.height,
    ])
    cbar_country.ax.text(
        1.33, 0.5,
        "Number of Validators",
        ha="right",
        va="center",
        rotation=0,
        fontsize=20,
        transform=cbar_country.ax.transAxes,
    )
    cbar_country.ax.tick_params(labelsize=20)

    ax.tick_params(axis="both", labelsize=16)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight" if fmt != "pdf" else None)
        print("✓ saved", outpath)

    plt.show()


# -------- run as script --------
if __name__ == "__main__":
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "data"))
    FIGURE_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "figure"))

    figure_country_density_plus_continent_latency(
        regions_csv=os.path.join(DATA_DIR, "gcp_regions.csv"),
        latency_csv=os.path.join(DATA_DIR, "gcp_latency.csv"),
        validators_csv=os.path.join(DATA_DIR, "validators.csv"),
        world_geojson_path=os.path.join(DATA_DIR, "world_countries.geo.json"),
        outpath=os.path.join(FIGURE_DIR, "country_density_plus_continent_latency.pdf"),
        fmt="pdf",
        highlight_pairs=[("North America", "Europe")],
        label_arcs=True,
    )
