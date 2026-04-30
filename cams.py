"""
cams.py — CAMS EAC4 Downloader · Post-processor · Visualizer
=============================================================
Downloads, processes, and visualizes CAMS EAC4 reanalysis data for a single
point or a rectangular area, one variable and one month at a time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER CONFIGURATION  ← Edit this section before running
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Time ─────────────────────────────────────────────────────────────────────
YEAR  = 2019        # int  – year  to download  (2003 – 2024)
MONTH = 7           # int  – month to download  (1 – 12)

# ── Variable ──────────────────────────────────────────────────────────────────
# Choose ONE key from the catalogue below:
#   "ozone"              → O₃  surface mixing ratio + total column
#   "nitrogen_dioxide"   → NO₂ surface mixing ratio + total column
#   "carbon_monoxide"    → CO  surface mixing ratio + total column
#   "sulphur_dioxide"    → SO₂ surface mixing ratio + total column
#   "dust"               → Dust aerosol optical depth at 550 nm
#   "pm25"               → PM2.5 surface mass concentration
#   "pm10"               → PM10 surface mass concentration
VARIABLE = "nitrogen_dioxide"

# ── Domain mode ──────────────────────────────────────────────────────────────
# "point"  → download data for a single lat/lon (nearest grid cell)
# "area"   → download data for a bounding box
MODE = "area"

# Point settings (used when MODE = "point")
LAT =  -16.5    # Latitude  of point
LON =  -68.15   # Longitude of point

# Area settings (used when MODE = "area")  [N, W, S, E]
AREA_N =  -8.0
AREA_W =  -70.0
AREA_S =  -24.0
AREA_E =  -57.0

# ── Output ───────────────────────────────────────────────────────────────────
OUTDIR = "cams_output"   # all files land here

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python cams.py -o download       # fetch NetCDF from ADS
    python cams.py -o postprocess    # convert NetCDF → CSV
    python cams.py -o visualize      # plot CSV → PNG/PDF
    python cams.py -o all            # run all three in sequence

Requirements
    pip install cdsapi xarray netCDF4 numpy pandas matplotlib cartopy

ADS API key (~/.adsapirc):
    url: https://ads.atmosphere.copernicus.eu/api
    key: <your-key>
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import argparse
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rasterio


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variable catalogue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  key  →  { sl: single-level CDS name or None,
#            pl: pressure-level CDS name or None,
#            label: human-readable,
#            units_raw: native units,
#            units_out: output units after conversion,
#            conv: conversion factor raw → out,
#            cmap: matplotlib colormap }

CATALOGUE = {
    "ozone": {
        "sl": "total_column_ozone",
        "pl": "ozone",
        "label": "Ozone (O₃)",
        "units_raw_sl": "kg m⁻²", "units_out_sl": "DU",   "conv_sl": 1 / 2.1415e-5,
        "units_raw_pl": "kg kg⁻¹","units_out_pl": "ppb",  "conv_pl": (28.97 / 48) * 1e9,
        "cmap": "viridis",
    },
    "nitrogen_dioxide": {
        "sl": "total_column_nitrogen_dioxide",
        "pl": "nitrogen_dioxide",
        "label": "Nitrogen Dioxide (NO₂)",
        "units_raw_sl": "kg m⁻²", "units_out_sl": "µmol m⁻²", "conv_sl": 1 / (46e-3 / 6.022e23 * 1e6),
        "units_raw_pl": "kg kg⁻¹","units_out_pl": "ppb",       "conv_pl": (28.97 / 46) * 1e9,
        "cmap": "YlOrRd",
    },
    "carbon_monoxide": {
        "sl": "total_column_carbon_monoxide",
        "pl": "carbon_monoxide",
        "label": "Carbon Monoxide (CO)",
        "units_raw_sl": "kg m⁻²", "units_out_sl": "kg m⁻²", "conv_sl": 1.0,
        "units_raw_pl": "kg kg⁻¹","units_out_pl": "ppb",     "conv_pl": (28.97 / 28) * 1e9,
        "cmap": "plasma",
    },
    "sulphur_dioxide": {
        "sl": "total_column_sulphur_dioxide",
        "pl": "sulphur_dioxide",
        "label": "Sulphur Dioxide (SO₂)",
        "units_raw_sl": "kg m⁻²", "units_out_sl": "kg m⁻²", "conv_sl": 1.0,
        "units_raw_pl": "kg kg⁻¹","units_out_pl": "ppb",     "conv_pl": (28.97 / 64) * 1e9,
        "cmap": "hot_r",
    },
    "dust": {
        "sl": "dust_aerosol_optical_depth_550nm",
        "pl": None,
        "label": "Dust AOD 550 nm",
        "units_raw_sl": "–", "units_out_sl": "–", "conv_sl": 1.0,
        "units_raw_pl": None,"units_out_pl": None,"conv_pl": None,
        "cmap": "YlOrBr",
    },
    "pm25": {
        "sl": "particulate_matter_2.5um",
        "pl": None,
        "label": "PM2.5",
        "units_raw_sl": "kg m⁻³", "units_out_sl": "µg m⁻³", "conv_sl": 1e9,
        "units_raw_pl": None,     "units_out_pl": None,       "conv_pl": None,
        "cmap": "Reds",
    },
    "pm10": {
        "sl": "particulate_matter_10um",
        "pl": None,
        "label": "PM10",
        "units_raw_sl": "kg m⁻³", "units_out_sl": "µg m⁻³", "conv_sl": 1e9,
        "units_raw_pl": None,     "units_out_pl": None,       "conv_pl": None,
        "cmap": "Oranges",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _month_date_range(year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}"


def _file_tag(variable: str, year: int, month: int, mode: str) -> str:
    return f"cams_{variable}_{year}{month:02d}_{mode}"


def _area_from_config(mode: str) -> list[float]:
    """Return [N, W, S, E] depending on mode."""
    if mode == "point":
        pad = 1.0
        return [LAT + pad, LON - pad, LAT - pad, LON + pad]
    else:
        return [AREA_N, AREA_W, AREA_S, AREA_E]


def _nc_paths(outdir: Path, tag: str) -> dict[str, Path]:
    return {
        "sl": outdir / f"{tag}_sl.nc",
        "pl": outdir / f"{tag}_pl.nc",
    }


def _csv_paths(outdir: Path, tag: str) -> dict[str, Path]:
    return {
        "sl": outdir / f"{tag}_sl.csv",
        "pl": outdir / f"{tag}_pl.csv",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1 — DOWNLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_download(MONTH):
    """Fetch raw NetCDF files from the Copernicus ADS."""
    try:
        import cdsapi
    except ImportError:
        sys.exit("ERROR: cdsapi not installed. Run: pip install cdsapi")

    if VARIABLE not in CATALOGUE:
        sys.exit(f"ERROR: unknown variable '{VARIABLE}'. "
                 f"Choose from: {list(CATALOGUE.keys())}")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    meta   = CATALOGUE[VARIABLE]
    tag    = _file_tag(VARIABLE, YEAR, MONTH, MODE)
    paths  = _nc_paths(outdir, tag)
    area   = _area_from_config(MODE)
    date_r = _month_date_range(YEAR, MONTH)
    times  = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

    print(f"\n{'='*60}")
    print(f"  CAMS EAC4 — DOWNLOAD")
    print(f"{'='*60}")
    print(f"  Variable : {meta['label']} ({VARIABLE})")
    print(f"  Period   : {YEAR}-{MONTH:02d}")
    print(f"  Mode     : {MODE}")
    if MODE == "point":
        print(f"  Point    : lat={LAT}, lon={LON}")
    else:
        print(f"  Area     : N={AREA_N} W={AREA_W} S={AREA_S} E={AREA_E}")
    print(f"  Output   : {outdir.resolve()}\n")

    client = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api")

    # ── Single-level ──────────────────────────────────────────────────────
    if meta["sl"] and not paths["sl"].exists():
        print(f"  → Downloading single-level: {meta['sl']} …")
        client.retrieve(
            "cams-global-reanalysis-eac4",
            {
                "variable":    meta["sl"],
                "date":        date_r,
                "time":        times,
                "area":        area,
                "data_format": "netcdf",
            },
            str(paths["sl"]),
        )
        print(f"    Saved → {paths['sl'].name}")
    elif paths["sl"].exists():
        print(f"  ✓ {paths['sl'].name} already exists – skipping.")

    # ── Pressure-level ────────────────────────────────────────────────────
    if meta["pl"] and not paths["pl"].exists():
        print(f"  → Downloading pressure-level (1000 hPa): {meta['pl']} …")
        client.retrieve(
            "cams-global-reanalysis-eac4",
            {
                "variable":       meta["pl"],
                "pressure_level": "1000",
                "date":           date_r,
                "time":           times,
                "area":           area,
                "data_format":    "netcdf",
            },
            str(paths["pl"]),
        )
        print(f"    Saved → {paths['pl'].name}")
    elif meta["pl"] and paths["pl"].exists():
        print(f"  ✓ {paths['pl'].name} already exists – skipping.")

    print("\n  Download complete ✓\n")

def download_year():
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    for MONTH in months:
        run_download(MONTH)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2 — POST-PROCESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _open_nc(nc_path: Path, mode: str):
    """Open a NetCDF and optionally select the nearest point."""
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    if "pressure_level" in ds.dims:
        ds = ds.squeeze("pressure_level", drop=True)

    if mode == "point":
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"
        ds = ds.sel({lat_dim: LAT, lon_dim: LON}, method="nearest")

    return ds


def _ds_to_df(ds, conv: float, out_units: str, col_name: str) -> pd.DataFrame:
    """Convert an xarray Dataset/DataArray to a tidy DataFrame."""
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    var_name  = list(ds.data_vars)[0]
    da        = ds[var_name]

    # For point mode: 1-D time series
    if da.ndim == 1:
        df = pd.DataFrame({col_name: da.values * conv},
                          index=pd.to_datetime(ds[time_dim].values))
        df.index.name = "time"
    else:
        # For area mode: keep as xarray (caller decides what to do)
        df = None

    return df, da * conv, out_units


def run_postprocess():
    """Convert raw NetCDF files to processed CSV files."""
    try:
        import xarray as xr
    except ImportError:
        sys.exit("ERROR: xarray not installed. Run: pip install xarray netCDF4")

    outdir = Path(OUTDIR)
    meta   = CATALOGUE[VARIABLE]
    tag    = _file_tag(VARIABLE, YEAR, MONTH, MODE)
    nc_p   = _nc_paths(outdir, tag)
    csv_p  = _csv_paths(outdir, tag)

    print(f"\n{'='*60}")
    print(f"  CAMS EAC4 — POST-PROCESS")
    print(f"{'='*60}")
    print(f"  Variable : {meta['label']}")
    print(f"  Mode     : {MODE}\n")

    processed = {}

    # ── Single-level ──────────────────────────────────────────────────────
    if meta["sl"] and nc_p["sl"].exists():
        print(f"  Processing {nc_p['sl'].name} …")
        ds  = _open_nc(nc_p["sl"], MODE)
        col = f"{VARIABLE}_sl_{meta['units_out_sl'].replace(' ','').replace('⁻','m').replace('²','2').replace('³','3').replace('µ','u')}"
        df, da_conv, units = _ds_to_df(ds, meta["conv_sl"], meta["units_out_sl"], col)

        if MODE == "point" and df is not None:
            # Add daily mean column
            df["daily_mean"] = df[col].resample("D").transform("mean")
            df.to_csv(csv_p["sl"])
            print(f"    → {csv_p['sl'].name}  [{units}]")
            print(f"\n  Summary [{units}]:")
            print(df[col].describe().round(4).to_string())
        else:
            # Area: save monthly mean map as NetCDF
            da_mean = da_conv.mean(dim="valid_time" if "valid_time" in da_conv.dims else "time")
            nc_mean = outdir / f"{tag}_sl_monthmean.nc"
            da_mean.to_netcdf(nc_mean)
            print(f"    → Monthly mean map saved: {nc_mean.name}  [{units}]")
            tif_sl = outdir / f"{tag}_sl_monthmean.tif"
            save_monthly_tiff(nc_mean, tif_sl, units, VARIABLE)

        processed["sl"] = (df, da_conv, units)

    elif meta["sl"]:
        print(f"  WARNING: {nc_p['sl'].name} not found – run download first.")

    # ── Pressure-level ────────────────────────────────────────────────────
    if meta["pl"] and nc_p["pl"].exists():
        print(f"\n  Processing {nc_p['pl'].name} …")
        ds  = _open_nc(nc_p["pl"], MODE)
        col = f"{VARIABLE}_surface_{meta['units_out_pl'].replace(' ','')}"
        df, da_conv, units = _ds_to_df(ds, meta["conv_pl"], meta["units_out_pl"], col)

        if MODE == "point" and df is not None:
            df["daily_mean"] = df[col].resample("D").transform("mean")
            df.to_csv(csv_p["pl"])
            print(f"    → {csv_p['pl'].name}  [{units}]")
            print(f"\n  Summary [{units}]:")
            print(df[col].describe().round(4).to_string())
        else:
            da_mean = da_conv.mean(dim="valid_time" if "valid_time" in da_conv.dims else "time")
            nc_mean = outdir / f"{tag}_pl_monthmean.nc"
            da_mean.to_netcdf(nc_mean)
            print(f"    → Monthly mean map saved: {nc_mean.name}  [{units}]")

        processed["pl"] = (df, da_conv, units)

    elif meta["pl"]:
        print(f"  WARNING: {nc_p['pl'].name} not found – run download first.")

    print("\n  Post-process complete ✓\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3 — VISUALIZE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _plot_timeseries(df: pd.DataFrame, col: str, meta: dict,
                     units: str, out_path: Path, title: str):
    """Time-series plot for point mode."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[col], lw=0.8, alpha=0.7, color="steelblue",
            label="3-hourly")

    if "daily_mean" in df.columns:
        ax.plot(df.index, df["daily_mean"], lw=2, color="darkblue",
                label="Daily mean")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{meta['label']} [{units}]")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=450)
    plt.close(fig)
    print(f"    → {out_path.name}")


def _plot_map(nc_path: Path, meta: dict, units: str, out_path: Path, title: str):
    """Spatial map for area mode. Uses cartopy if available, else plain imshow."""
    import xarray as xr

    # open_dataset is safer — open_dataarray fails if variable name was lost on save
    ds = xr.open_dataset(nc_path)
    var_name = list(ds.data_vars)[0]
    print(var_name)
    da = ds[var_name]

    lat_dim = "latitude" if "latitude" in da.dims else "lat"
    lon_dim = "longitude" if "longitude" in da.dims else "lon"
    lats = da[lat_dim].values
    lons = da[lon_dim].values
    # Squeeze any leftover size-1 dimensions (time, level, etc.)
    data = da.values.squeeze()

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig, ax = plt.subplots(figsize=(10, 10),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="cornflowerblue")
        im = ax.pcolormesh(lons, lats, data, cmap=meta["cmap"],
                           transform=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
        gl.top_labels = gl.right_labels = False
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(data, origin="upper", aspect="auto",
                       extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                       cmap=meta["cmap"])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{meta['label']} [{units}]")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out_path.name}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GeoTIFF export helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def save_monthly_tiff(nc_path: Path, out_path: Path, units: str, variable: str):
    """
    Save a monthly-mean NetCDF grid as a GeoTIFF with full spatial metadata.

    The output is a single-band Float32 GeoTIFF in WGS-84 (EPSG:4326).
    NoData is set to -9999. Metadata tags embed the variable name, units,
    and source file so the TIFF is self-documenting in any GIS tool.

    Requirements: rasterio  (pip install rasterio)
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
    except ImportError:
        print("  WARNING: rasterio not installed – skipping GeoTIFF export.")
        print("           Run:  pip install rasterio")
        return

    import xarray as xr

    # ── Load data ────────────────────────────────────────────────────────────
    ds       = xr.open_dataset(nc_path)
    var_name = list(ds.data_vars)[0]
    da       = ds[var_name]

    lat_dim = "latitude" if "latitude" in da.dims else "lat"
    lon_dim = "longitude" if "longitude" in da.dims else "lon"

    lats = da[lat_dim].values   # 1-D, descending (N→S) or ascending
    lons = da[lon_dim].values   # 1-D, ascending  (W→E)
    data = da.values.squeeze().astype("float32")

    # rasterio expects rows ordered N→S (first row = northernmost)
    if lats[0] < lats[-1]:          # ascending → flip
        lats = lats[::-1]
        data = data[::-1, :]

    nodata = float(-9999)
    data   = np.where(np.isfinite(data), data, nodata)

    # ── Build affine transform ───────────────────────────────────────────────
    # from_bounds(west, south, east, north, width, height)
    transform = from_bounds(
        west=float(lons.min()),
        south=float(lats.min()),
        east=float(lons.max()),
        north=float(lats.max()),
        width=data.shape[1],
        height=data.shape[0],
    )

    # ── Write GeoTIFF ────────────────────────────────────────────────────────
    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     data.shape[1],
        "height":    data.shape[0],
        "count":     1,                        # single band
        "crs":       CRS.from_epsg(4326),      # WGS-84 geographic
        "transform": transform,
        "nodata":    nodata,
        "compress":  "lzw",                    # lossless compression
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            variable=variable,
            units=units,
            source=nc_path.name,
            crs="EPSG:4326 (WGS-84)",
            nodata=str(nodata),
        )

    print(f"    → GeoTIFF saved: {out_path.name}  [{units}]")
    print(f"       Size   : {data.shape[1]} × {data.shape[0]} px  "
          f"(lon × lat)")
    print(f"       Extent : W={lons.min():.3f} E={lons.max():.3f} "
          f"S={lats.min():.3f} N={lats.max():.3f}")
    print(f"       Range  : min={data[data != nodata].min():.4g}  "
          f"max={data[data != nodata].max():.4g}")

def run_visualize():
    """Generate PNG plots from processed output files."""
    # try:
    #     import matplotlib
    #     matplotlib.use("Agg")
    #     import matplotlib.pyplot as plt
    # except ImportError:
    #     sys.exit("ERROR: matplotlib not installed. Run: pip install matplotlib")

    outdir = Path(OUTDIR)
    meta   = CATALOGUE[VARIABLE]
    tag    = _file_tag(VARIABLE, YEAR, MONTH, MODE)
    csv_p  = _csv_paths(outdir, tag)
    month_name = calendar.month_name[MONTH]

    print(f"\n{'='*60}")
    print(f"  CAMS EAC4 — VISUALIZE")
    print(f"{'='*60}")
    print(f"  Variable : {meta['label']}")
    print(f"  Mode     : {MODE}\n")

    
    if MODE == "point":
        # ── Time-series plots ─────────────────────────────────────────────
        for level, csv_file in csv_p.items():
            if not csv_file.exists():
                continue
            if level == "pl" and meta["pl"] is None:
                continue
            if level == "sl" and meta["sl"] is None:
                continue

            df      = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            units   = meta[f"units_out_{level}"]
            col     = [c for c in df.columns if c != "daily_mean"][0]
            level_s = "Total Column" if level == "sl" else "Surface (~1000 hPa)"
            title   = (f"{meta['label']} — {level_s}\n"
                       f"lat={LAT}, lon={LON}  |  {month_name} {YEAR}")
            out_png = outdir / f"{tag}_{level}_timeseries.png"
            _plot_timeseries(df, col, meta, units, out_png, title)

        # ── Diurnal cycle plot ────────────────────────────────────────────
        for level, csv_file in csv_p.items():
            if not csv_file.exists():
                continue
            if (level == "pl" and meta["pl"] is None) or \
               (level == "sl" and meta["sl"] is None):
                continue

            df    = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            col   = [c for c in df.columns if c != "daily_mean"][0]
            units = meta[f"units_out_{level}"]
            diurnal = df[col].groupby(df.index.hour).mean()

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(diurnal.index, diurnal.values, "o-", color="tomato", lw=2)
            ax.set_xticks(range(0, 24, 3))
            ax.set_xlabel("Hour (UTC)")
            ax.set_ylabel(f"{meta['label']} [{units}]")
            level_s = "Total Column" if level == "sl" else "Surface"
            ax.set_title(f"Diurnal Cycle — {meta['label']} {level_s}\n"
                         f"lat={LAT}, lon={LON}  |  {month_name} {YEAR}",
                         fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_diurnal = outdir / f"{tag}_{level}_diurnal.png"
            fig.savefig(out_diurnal, dpi=150)
            plt.close(fig)
            print(f"    → {out_diurnal.name}")

        
    else:
        # ── Spatial map plots ─────────────────────────────────────────────
        for level in ("sl", "pl"):
            # Skip if this variable has no data at this level
            if meta[level] is None:
                continue

            nc_mean = outdir / f"{tag}_{level}_monthmean.nc"
            if not nc_mean.exists():
                print(f"  WARNING: {nc_mean.name} not found – run postprocess first.")
                continue

            units   = meta[f"units_out_{level}"]
            level_s = "Total Column" if level == "sl" else "Surface (~1000 hPa)"
            title   = (f"{meta['label']} — {level_s} — Monthly Mean\n"
                       f"{month_name} {YEAR}")
            out_png = outdir / f"{tag}_{level}_map.png"
            print(f"  Plotting {nc_mean.name} ...")
            try:
                _plot_map(nc_mean, meta, units, out_png, title)
            except Exception as e:
                print(f"  ERROR in _plot_map: {e}")
                import traceback; traceback.print_exc()

    print("\n  Visualize complete ✓\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPERATIONS = {
    "download":      run_download,
    "download_year": download_year,
    "postprocess":   run_postprocess,
    "visualize":     run_visualize,
}


def main():
    parser = argparse.ArgumentParser(
        description="CAMS EAC4 downloader · post-processor · visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Operations:
  download     → fetch NetCDF from ADS
  postprocess  → convert NetCDF → CSV (point) or monthly-mean NetCDF (area)
  visualize    → generate PNG figures
  all          → run all three in order

Edit the USER CONFIGURATION block at the top of the script to set:
  YEAR, MONTH, VARIABLE, MODE, LAT/LON or AREA_*, OUTDIR
        """,
    )
    parser.add_argument(
        "-o", "--operation",
        required=True,
        choices=["download", "download_year", "postprocess", "visualize", "all"],
        help="Operation to run",
    )
    args = parser.parse_args()

    if args.operation == "all":
        for op_name, op_fn in OPERATIONS.items():
            op_fn()
    else:
        OPERATIONS[args.operation]()


if __name__ == "__main__":
    main()
