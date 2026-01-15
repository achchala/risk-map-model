# Data Files

this directory should contain the following data files to be able to run run_risk_analysis.py

## Required Files

1. **Traffic_Collisions_Open_Data_2437597425626428496.xlsx**
   - Source: Toronto Police Open Data
   - Description: General traffic collision records
   - Size: ~79 MB

2. **TOTAL_KSI_6386614326836635957.csv**
   - Source: Toronto Police Open Data
   - Description: Killed or Seriously Injured (KSI) crash records
   - Size: ~7 MB

3. **Centreline - Version 2 - 4326.geojson**
   - Source: Toronto Open Data Portal
   - Description: Road network geometry (WGS84 coordinate system)
   - Size: ~90 MB

## Where to Download

### Toronto Police Open Data
- Visit: https://data.torontopolice.on.ca/
- Search for "Traffic Collisions" and "KSI" datasets
- Download the files and place them in this directory

### Toronto Open Data Portal
- Visit: https://open.toronto.ca/
- Search for "Centreline" or "Road Network"
- Download the GeoJSON version with EPSG:4326 coordinate system

## Note

These files are large and are excluded from git (see `.gitignore`). 
You need to download them manually and place them in this `data/` directory.

