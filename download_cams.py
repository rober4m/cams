import argparse
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# CAMS EAC4 variable catalogue
# ---------------------------------------------------------------------------
# Each entry:  CDS name  →  (short_label, units, description)
SINGLE_LEVEL_VARS = {
    # ── Reactive gases (total column) ───────────────────────────────────────
    "total_column_ozone":
        ("tco3",    "kg m⁻²",  "Total column ozone (Dobson-units = value / 2.1415e-5)"),
    "total_column_nitrogen_dioxide":
        ("tcno2",   "kg m⁻²",  "Total column NO₂"),
    "total_column_carbon_monoxide":
        ("tcco",    "kg m⁻²",  "Total column CO"),
    "total_column_sulphur_dioxide":
        ("tcso2",   "kg m⁻²",  "Total column SO₂"),

    # ── Aerosol optical depths at 550 nm ────────────────────────────────────
    "total_aerosol_optical_depth_550nm":
        ("aod550",  "dimensionless", "Total aerosol optical depth at 550 nm"),
    "dust_aerosol_optical_depth_550nm":
        ("duaod550","dimensionless", "Dust aerosol optical depth at 550 nm"),
    "organic_matter_aerosol_optical_depth_550nm":
        ("omaod550","dimensionless", "Organic matter aerosol optical depth at 550 nm"),
    "black_carbon_aerosol_optical_depth_550nm":
        ("bcaod550","dimensionless", "Black carbon aerosol optical depth at 550 nm"),
    "sulphate_aerosol_optical_depth_550nm":
        ("suaod550","dimensionless", "Sulphate aerosol optical depth at 550 nm"),

    # ── Particulate matter (surface mass concentration) ─────────────────────
    "particulate_matter_2.5um":
        ("pm2p5",   "kg m⁻³",  "PM2.5 surface mass concentration"),
    "particulate_matter_10um":
        ("pm10",    "kg m⁻³",  "PM10 surface mass concentration"),
}

# Pressure-level vars to grab at the surface (1000 hPa ≈ surface)
PRESSURE_LEVEL_VARS = {
    "ozone":
        ("o3",  "kg kg⁻¹", "Ozone mass mixing ratio"),
    "nitrogen_dioxide":
        ("no2", "kg kg⁻¹", "NO₂ mass mixing ratio"),
    "carbon_monoxide":
        ("co",  "kg kg⁻¹", "CO mass mixing ratio"),
    "sulphur_dioxide":
        ("so2", "kg kg⁻¹", "SO₂ mass mixing ratio"),
}

# Surface-like pressure level available in EAC4
SURFACE_PRESSURE_LEVEL = "1000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_year_chunks(start: int, end: int, chunk: int = 3) -> list[list[int]]:
    years = list(range(start, end + 1))
    return [years[i:i + chunk] for i in range(0, len(years), chunk)]


def tight_area(lat: float, lon: float, pad: float = 1.0) -> list[float]:
    """Return [N, W, S, E] bounding box around a point."""
    return [lat + pad, lon - pad, lat - pad, lon + pad]


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------
def download_single_level_chunk(client: cdsapi.Client, years: list[int],
                                lat: float, lon: float, out_path: Path):
    """Download CAMS EAC4 single-level variables for a chunk of years."""
    str_years = [str(y) for y in years]

    request = {
        "variable": list(SINGLE_LEVEL_VARS.keys()),
        "date": f"{years[0]}-01-01/{years[-1]}-12-31",
        "time": ["00:00", "03:00", "06:00", "09:00", "12:00",
                 "15:00", "18:00", "21:00"],
        "area": tight_area(lat, lon),
        "data_format": "netcdf",
    }

    print(f"  [Single-level] Requesting years {years[0]}–{years[-1]} …")
    client.retrieve("cams-global-reanalysis-eac4", request, str(out_path))
    print(f"  Saved → {out_path}")


def download_pressure_level_chunk(client: cdsapi.Client, years: list[int],
                                  lat: float, lon: float, out_path: Path):
    """Download CAMS EAC4 pressure-level variables at ~surface (1000 hPa)."""
    request = {
        "variable": list(PRESSURE_LEVEL_VARS.keys()),
        "pressure_level": SURFACE_PRESSURE_LEVEL,
        "date": f"{years[0]}-01-01/{years[-1]}-12-31",
        "time": ["00:00", "03:00", "06:00", "09:00", "12:00",
                 "15:00", "18:00", "21:00"],
        "area": tight_area(lat, lon),
        "data_format": "netcdf",
    }

    print(f"  [Pressure-level] Requesting years {years[0]}–{years[-1]} …")
    client.retrieve("cams-global-reanalysis-eac4", request, str(out_path))
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def _open_and_select(nc_path: Path, lat: float, lon: float) -> xr.Dataset:
    ds = xr.open_dataset(nc_path)
    # CAMS uses 'latitude'/'longitude' or 'lat'/'lon' depending on version
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"
    ds = ds.sel({lat_dim: lat, lon_dim: lon}, method="nearest")
    # Drop pressure level dimension if present (we only have one level)
    if "pressure_level" in ds.dims:
        ds = ds.squeeze("pressure_level", drop=True)
    return ds


def process_single_level(nc_path: Path, lat: float, lon: float) -> pd.DataFrame:
    ds = _open_and_select(nc_path, lat, lon)
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    df = pd.DataFrame(index=pd.to_datetime(ds[time_dim].values))
    df.index.name = "time"

    # Map dataset variable names to short labels
    var_map = {
        "tcco":     ("tcco",  "total_column_CO_kg_m2"),
        "tcno2":    ("tcno2", "total_column_NO2_kg_m2"),
        "tco3":     ("tco3",  "total_column_O3_kg_m2"),
        "tcso2":    ("tcso2", "total_column_SO2_kg_m2"),
        "aod550":   ("aod550",  "total_AOD_550nm"),
        "duaod550": ("duaod550","dust_AOD_550nm"),
        "omaod550": ("omaod550","organic_AOD_550nm"),
        "bcaod550": ("bcaod550","black_carbon_AOD_550nm"),
        "suaod550": ("suaod550","sulphate_AOD_550nm"),
        "pm2p5":    ("pm2p5",  "PM2p5_kg_m3"),
        "pm10":     ("pm10",   "PM10_kg_m3"),
    }

    for ds_var in ds.data_vars:
        key = str(ds_var)
        if key in var_map:
            col = var_map[key][1]
        else:
            col = key
        df[col] = ds[ds_var].values

    # Convert total column ozone to Dobson Units (1 DU = 2.1415e-5 kg m⁻²)
    if "total_column_O3_kg_m2" in df.columns:
        df["total_column_O3_DU"] = df["total_column_O3_kg_m2"] / 2.1415e-5

    # Convert PM from kg m⁻³ to μg m⁻³ (common unit)
    for col in ["PM2p5_kg_m3", "PM10_kg_m3"]:
        if col in df.columns:
            new_col = col.replace("kg_m3", "ug_m3")
            df[new_col] = df[col] * 1e9

    return df


def process_pressure_level(nc_path: Path, lat: float, lon: float) -> pd.DataFrame:
    ds = _open_and_select(nc_path, lat, lon)
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    df = pd.DataFrame(index=pd.to_datetime(ds[time_dim].values))
    df.index.name = "time"

    rename = {
        "o3":  "O3_surface_kg_kg",
        "no2": "NO2_surface_kg_kg",
        "co":  "CO_surface_kg_kg",
        "so2": "SO2_surface_kg_kg",
    }
    for ds_var in ds.data_vars:
        key = str(ds_var)
        col = rename.get(key, key + "_surface")
        df[col] = ds[ds_var].values

    # Convert mass mixing ratios to ppb (approximate, using dry-air MW = 28.97 g/mol)
    molar_mass = {"O3_surface_kg_kg": 48, "NO2_surface_kg_kg": 46,
                  "CO_surface_kg_kg": 28, "SO2_surface_kg_kg": 64}
    for col, mm in molar_mass.items():
        if col in df.columns:
            ppb_col = col.replace("_kg_kg", "_ppb")
            df[ppb_col] = df[col] * (28.97 / mm) * 1e9

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download CAMS EAC4 air quality data for a lat/lon point.")
    parser.add_argument("--lat",    type=float, default=51.5,
                        help="Latitude  (default: 51.5 – London)")
    parser.add_argument("--lon",    type=float, default=-0.12,
                        help="Longitude (default: -0.12 – London)")
    parser.add_argument("--start",  type=int,   default=2010,
                        help="First year, min 2003 (default: 2010)")
    parser.add_argument("--end",    type=int,   default=2023,
                        help="Last year, max 2024  (default: 2023)")
    parser.add_argument("--outdir", type=str,   default="cams_output",
                        help="Output directory (default: cams_output)")
    parser.add_argument("--chunk",  type=int,   default=3,
                        help="Years per CDS request (default: 3)")
    args = parser.parse_args()

    # CAMS EAC4 only goes back to 2003
    if args.start < 2003:
        print("WARNING: CAMS EAC4 starts in 2003. Adjusting --start to 2003.")
        args.start = 2003
    if args.end > 2024:
        print("WARNING: CAMS EAC4 ends in Dec 2024. Adjusting --end to 2024.")
        args.end = 2024

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCAMS EAC4 downloader")
    print(f"  Point  : lat={args.lat}, lon={args.lon}")
    print(f"  Period : {args.start} – {args.end}  "
          f"({args.end - args.start + 1} years)")
    print(f"  Output : {out_dir.resolve()}\n")

    # Note: CAMS uses the ADS endpoint, not the CDS endpoint.
    # Make sure ~/.adsapirc points to https://ads.atmosphere.copernicus.eu/api
    client = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api")

    chunks = build_year_chunks(args.start, args.end, chunk=args.chunk)

    sl_frames: list[pd.DataFrame] = []
    pl_frames: list[pd.DataFrame] = []

    for chunk_years in chunks:
        tag = f"{chunk_years[0]}-{chunk_years[-1]}"

        # ── Single-level ──────────────────────────────────────────────────
        sl_path = out_dir / f"cams_sl_{tag}.nc"
        if sl_path.exists():
            print(f"  {sl_path.name} already exists – skipping download.")
        else:
            download_single_level_chunk(client, chunk_years,
                                        args.lat, args.lon, sl_path)
        print(f"  Processing {sl_path.name} …")
        sl_frames.append(process_single_level(sl_path, args.lat, args.lon))

        # ── Pressure-level (surface gases) ────────────────────────────────
        pl_path = out_dir / f"cams_pl_{tag}.nc"
        if pl_path.exists():
            print(f"  {pl_path.name} already exists – skipping download.")
        else:
            download_pressure_level_chunk(client, chunk_years,
                                          args.lat, args.lon, pl_path)
        print(f"  Processing {pl_path.name} …")
        pl_frames.append(process_pressure_level(pl_path, args.lat, args.lon))

    # ── Merge all chunks ──────────────────────────────────────────────────
    print("\nMerging all chunks …")
    df_sl = pd.concat(sl_frames).sort_index()
    df_pl = pd.concat(pl_frames).sort_index()

    df_all = df_sl.join(df_pl, how="outer")
    df_all = df_all[~df_all.index.duplicated(keep="first")]

    # ── Save ──────────────────────────────────────────────────────────────
    tag_out = f"lat{args.lat}_lon{args.lon}_{args.start}-{args.end}"
    csv_path = out_dir / f"cams_airquality_{tag_out}.csv"
    df_all.to_csv(csv_path)
    print(f"\nMerged CSV saved → {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    priority_cols = [c for c in [
        "total_column_O3_DU",
        "NO2_surface_ppb",
        "CO_surface_ppb",
        "SO2_surface_ppb",
        "dust_AOD_550nm",
        "PM2p5_ug_m3",
        "PM10_ug_m3",
    ] if c in df_all.columns]

    print("\n── Summary statistics ──────────────────────────────────────────")
    if priority_cols:
        print(df_all[priority_cols].describe().round(4).to_string())
    else:
        print(df_all.describe().round(4).to_string())

    print("\nAll columns in output:")
    for col in df_all.columns:
        print(f"  {col}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
