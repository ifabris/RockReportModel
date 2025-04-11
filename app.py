import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from scipy.interpolate import griddata
from io import BytesIO

st.set_page_config(page_title="Cu 3D Prediction Volume Generator", layout="wide")
st.title("🧠 Copper Prediction Volume Generator")

st.markdown("""
Upload your `collars.xlsx` and `assays.xlsx` files. The app will train a model and generate a volumetric 3D heatmap (ore body style) of predicted values for the selected element.

- Make sure column names include:
    - `Drillhole` or `BHID`
    - `From`, `To`, `Cu`, `Au`, etc.
    - `Easting`, `Northing`, `Elevation`, `Dip`, `Azimuth`
""")

collar_file = st.file_uploader("📂 Upload Collar File", type=["xlsx"])
assay_file = st.file_uploader("📂 Upload Assay File", type=["xlsx"])

if collar_file and assay_file:
    # Read files
    collars = pd.read_excel(collar_file)
    assays = pd.read_excel(assay_file)

    # Rename & clean
    collars = collars.rename(columns=lambda c: c.strip())
    assays = assays.rename(columns=lambda c: c.strip())

    # Normalize assay column names and clean units
    assays.columns = assays.columns.str.strip()
    clean_col_map = {col: col.lower().replace('(m)', '').strip() for col in assays.columns}

    # Clean up all object-type columns (e.g. Cu = "<0.01")
    for col in assays.columns:
        if assays[col].dtype == object:
            assays[col] = assays[col].astype(str).str.replace('<', '', regex=False)
            assays[col] = pd.to_numeric(assays[col], errors='coerce')

    # Map important fields
    column_map = {
        "from": next((col for col, clean in clean_col_map.items() if "from" in clean), None),
        "to": next((col for col, clean in clean_col_map.items() if "to" in clean and "total" not in clean), None),
        "bhid": next((col for col, clean in clean_col_map.items() if "bh" in clean and "id" in clean), None)
    }

    if not all(column_map.values()):
        st.error("❌ Couldn't detect 'From', 'To', or 'BHID' columns. Please check your assay file.")
        st.stop()

    # Rename to standardized
    assays = assays.rename(columns={
        column_map["from"]: "From",
        column_map["to"]: "To",
        column_map["bhid"]: "BHID"
    })

    # Get true element columns by filtering out structural ones
    excluded_keywords = ["from", "to", "length", "bhid"]
    element_options = [
        col for col in assays.select_dtypes(include=[np.number]).columns
        if all(kw not in col.lower() for kw in excluded_keywords)
    ]

    if not element_options:
        st.error("❌ No valid element columns found. Make sure your assay file has numeric data like 'Cu', 'Au', etc.")
        st.stop()

    selected_element = st.selectbox("🧪 Select element to predict", element_options)

    for col in ["From", "To", selected_element]:
        assays[col] = pd.to_numeric(assays[col], errors="coerce")

    assays = assays.dropna(subset=["From", "To", selected_element])
    assays["Length"] = assays["To"] - assays["From"]

    # Standardize collar columns
    rename_map = {
        "Elevation (mamsl)": "Elevation",
        "Dip        (°)": "Dip",
        "Azimuth       (°)": "Azimuth"
    }
    collars = collars.rename(columns=rename_map)

    if "Drillhole" in collars.columns:
        collars = collars.rename(columns={"Drillhole": "BHID"})

    # Merge
    merged = pd.merge(assays, collars, on="BHID")
    merged = merged.dropna(subset=["Easting", "Northing", "Elevation", "Dip", "Azimuth"])
    merged[f"log_{selected_element}"] = np.log1p(merged[selected_element])

    # Train model
    features = ["Easting", "Northing", "Elevation", "From", "To", "Length", "Dip", "Azimuth"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(merged[features], merged[f"log_{selected_element}"])

    # 3D volume prediction grid
    x_vals = np.linspace(merged["Easting"].min(), merged["Easting"].max(), 40)
    y_vals = np.linspace(merged["Northing"].min(), merged["Northing"].max(), 40)
    z_vals = np.linspace(merged["Elevation"].min(), merged["Elevation"].max(), 40)
    grid_points = [(x, y, z) for x in x_vals for y in y_vals for z in z_vals]
    grid_df = pd.DataFrame(grid_points, columns=["Easting", "Northing", "Elevation"])
    grid_df["From"] = 0
    grid_df["To"] = 1
    grid_df["Length"] = 1
    grid_df["Dip"] = 90
    grid_df["Azimuth"] = 0
    preds = model.predict(grid_df[features])
    grid_df[f"Predicted_{selected_element}"] = np.expm1(preds)

    # Use a percentile range for better color scaling
    vmin = np.percentile(grid_df[f"Predicted_{selected_element}"], 5)
    vmax = np.percentile(grid_df[f"Predicted_{selected_element}"], 95)

    st.success(f"Model trained and 3D volume generated for {selected_element}!")

    # Threshold slider
    threshold = st.slider("🎚 Minimum PPM Threshold to Display", float(vmin), float(vmax), float(vmin), step=10.0)

    # Isosurface volume
    fig = go.Figure()
    fig.add_trace(go.Isosurface(
        x=grid_df['Easting'],
        y=grid_df['Northing'],
        z=grid_df['Elevation'],
        value=grid_df[f"Predicted_{selected_element}"],
        isomin=threshold,
        isomax=vmax,
        surface_count=3,
        caps=dict(x_show=False, y_show=False),
        colorscale='Hot',
        colorbar=dict(title=f"{selected_element} (ppm)"),
        cmin=vmin,
        cmax=vmax,
        opacity=0.9
    ))

    # Drillhole orientation
    arrow_length = 25
    for _, row in collars.iterrows():
        dip_rad = np.radians(row['Dip'])
        az_rad = np.radians(row['Azimuth'])
        x0, y0, z0 = row['Easting'], row['Northing'], row['Elevation']
        x1 = x0 + arrow_length * np.cos(dip_rad) * np.sin(az_rad)
        y1 = y0 + arrow_length * np.cos(dip_rad) * np.cos(az_rad)
        z1 = z0 - arrow_length * np.sin(dip_rad)

        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines+markers+text',
            line=dict(color='red', width=3),
            marker=dict(size=3, color='blue'),
            name=row['BHID'],
            text=[row['BHID'], ''],
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f"🌍 3D {selected_element} Isosurface Volume with Drillhole Paths",
        scene=dict(
            xaxis_title="Easting",
            yaxis_title="Northing",
            zaxis_title="Elevation (m)",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # CSV Export
    csv = grid_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"📥 Download Predicted {selected_element} CSV",
        csv,
        f"{selected_element.lower()}_predicted_volume.csv",
        "text/csv"
    )
