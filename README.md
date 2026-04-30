# CAMS Atmospheric Composition Downloader

Downloads CAMS EAC4 reanalysis data for a single point (lat/lon).

Variables:
  - Ozone (O₃)                  – total column & surface
  - Nitrogen dioxide (NO₂)      – total column & surface
  - Carbon monoxide (CO)         – total column & surface
  - Sulphur dioxide (SO₂)        – surface
  - Dust aerosol optical depth   – 550 nm
  - PM2.5 and PM10               – particulate matter (surface)

## Data source:
  CAMS global reanalysis (EAC4) – cams-global-reanalysis-eac4
  Available: 2003 – Dec 2024 | Resolution: ~80 km | Frequency: 3-hourly / daily

## Requirements
------------
    pip install cdsapi xarray netCDF4 numpy pandas matplotlib cartopy rasterio

## CDS/ADS API key setup:
  1. Register at https://ads.atmosphere.copernicus.eu
  2. Create ~/.adsapirc  (or set env vars ADS_URL + ADS_API_KEY)
     url: https://ads.atmosphere.copernicus.eu/api
     key: <your-key>

## Usage
    python cams.py -o download       # fetch NetCDF from ADS
    python cams.py -o postprocess    # convert NetCDF → CSV
    python cams.py -o visualize      # plot CSV → PNG/PDF
    python cams.py -o all            # run all three in sequence
