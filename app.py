import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from zipfile import ZipFile
import tempfile
import os

# Page settings
st.set_page_config(page_title="Flood Susceptibility ROC-AUC App", page_icon="🌊", layout="centered")

# Sidebar - Author Info
with st.sidebar:
    st.markdown("### 👨‍🔬 Developed By")
    st.markdown("**Sachchidanand Singh**  \nScientist, WHRC  \nNational Institute of Hydrology, Jammu")
    st.markdown("---")
    st.markdown("📌 **Instructions**:")
    st.markdown("1. Upload **Flood Points** (as zipped shapefile).  \n2. Upload **Non-Flood Points**.  \n3. Upload **Flood Susceptibility Raster (GeoTIFF)**.  \n4. View AUC & ROC plot below.")
    st.markdown("---")

# Title
st.markdown("<h2 style='text-align: center; color: #0077b6;'>🌊 Flood Susceptibility ROC-AUC Analysis</h2>", unsafe_allow_html=True)

# Function to read zipped shapefiles
def read_zipped_shapefile(zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        with ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            shapefiles = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shapefiles:
                raise ValueError("⚠️ No .shp file found in the uploaded ZIP.")
            shapefile_path = os.path.join(tmpdir, shapefiles[0])
            return gpd.read_file(shapefile_path)

# Uploaders
st.subheader("📁 Upload Required Files")

flood_zip = st.file_uploader("1️⃣ Upload Flood Points (ZIP Shapefile)", type="zip")
nonflood_zip = st.file_uploader("2️⃣ Upload Non-Flood Points (ZIP Shapefile)", type="zip")
susceptibility_raster = st.file_uploader("3️⃣ Upload Flood Susceptibility Raster (GeoTIFF)", type="tif")

if flood_zip and nonflood_zip and susceptibility_raster:
    try:
        # Load data
        flood_points = read_zipped_shapefile(flood_zip)
        nonflood_points = read_zipped_shapefile(nonflood_zip)
        raster = rasterio.open(susceptibility_raster)

        # Project to raster CRS
        flood_points = flood_points.to_crs(raster.crs)
        nonflood_points = nonflood_points.to_crs(raster.crs)

        # Extract raster values
        flood_coords = [(geom.x, geom.y) for geom in flood_points.geometry]
        nonflood_coords = [(geom.x, geom.y) for geom in nonflood_points.geometry]

        flood_vals = np.array([val[0] for val in raster.sample(flood_coords)])
        nonflood_vals = np.array([val[0] for val in raster.sample(nonflood_coords)])

        y_true = np.concatenate([np.ones(len(flood_vals)), np.zeros(len(nonflood_vals))])
        y_scores = np.concatenate([flood_vals, nonflood_vals])

        # Remove NaNs
        valid = ~np.isnan(y_scores)
        y_true, y_scores = y_true[valid], y_scores[valid]

        # Compute ROC and AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Output
        st.markdown(f"<h3 style='text-align:center;'>📊 AUC Score: <span style='color:#0077b6;'>{roc_auc:.3f}</span></h3>", unsafe_allow_html=True)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='navy', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc='lower right')
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

else:
    st.info("⬆️ Please upload all three files to proceed with the analysis.")
