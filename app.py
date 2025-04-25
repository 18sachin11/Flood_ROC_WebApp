import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from zipfile import ZipFile
import tempfile
import os

st.set_page_config(page_title="Flood Susceptibility ROC-AUC Analysis", layout="centered")
st.title("🌊 Flood Susceptibility ROC-AUC Analysis")

# Helper function to read zipped shapefiles
def read_zipped_shapefile(zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile(zip_file) as zf:
            zf.extractall(tmpdir)
            shapefile = [file for file in os.listdir(tmpdir) if file.endswith('.shp')][0]
            gdf = gpd.read_file(os.path.join(tmpdir, shapefile))
    return gdf

# File Uploads
flood_zip = st.file_uploader("Upload Flood Points (ZIP Shapefile)", type="zip")
nonflood_zip = st.file_uploader("Upload Non-Flood Points (ZIP Shapefile)", type="zip")
susceptibility_raster = st.file_uploader("Upload Flood Susceptibility Raster (GeoTIFF)", type="tif")

if flood_zip and nonflood_zip and susceptibility_raster:
    # Read shapefiles
    flood_points = read_zipped_shapefile(flood_zip)
    nonflood_points = read_zipped_shapefile(nonflood_zip)

    # Open raster
    raster = rasterio.open(susceptibility_raster)

    # Reproject points to raster CRS
    flood_points = flood_points.to_crs(raster.crs)
    nonflood_points = nonflood_points.to_crs(raster.crs)

    # Extract raster values at point locations
    flood_coords = [(geom.x, geom.y) for geom in flood_points.geometry]
    nonflood_coords = [(geom.x, geom.y) for geom in nonflood_points.geometry]

    flood_values = np.array([val[0] for val in raster.sample(flood_coords)])
    nonflood_values = np.array([val[0] for val in raster.sample(nonflood_coords)])

    # Prepare data for ROC
    y_true = np.concatenate([np.ones(len(flood_values)), np.zeros(len(nonflood_values))])
    y_scores = np.concatenate([flood_values, nonflood_values])

    # Remove NaNs
    mask = ~np.isnan(y_scores)
    y_true = y_true[mask]
    y_scores = y_scores[mask]

    # Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Display Results
    st.subheader(f"📊 ROC Curve (AUC = {roc_auc:.3f})")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Flood Susceptibility')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    st.pyplot(fig)

else:
    st.info("Please upload all three files to proceed.")

