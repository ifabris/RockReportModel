import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from scipy.interpolate import griddata
from sklearn.metrics import pairwise_distances_argmin_min
from io import BytesIO

st.set_page_config(page_title="3D Prediction Volume Generator", layout="wide")
st.title("Prediction Volume Generator")

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

    for col in assays.columns:
        if assays[col].dtype == object:
            assays[col] = assays[col].astype(str).str.replace('<', '', regex=False)
            assays[col] = pd.to_numeric(assays[col], errors='coerce')

    column_map = {
        "from": next((col for col, clean in clean_col_map.items() if "from" in clean), None),
        "to": next((col for col, clean in clean_col_map.items() if "to" in clean and "total" not in clean), None),
        "bhid": next((col for col, clean in clean_col_map.items() if "bh" in clean and "id" in clean), None)
    }

    if not all(column_map.values()):
        st.error("❌ Couldn't detect 'From', 'To', or 'BHID' columns. Please check your assay file.")
        st.stop()

    assays = assays.rename(columns={
        column_map["from"]: "From",
        column_map["to"]: "To",
        column_map["bhid"]: "BHID"
    })

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

    # ✅ Drillhole Intervals (Real Segments)
    def interval_to_xyz(row):
        dip_rad = np.radians(row["Dip"])
        azm_rad = np.radians(row["Azimuth"])
        x0 = row["Easting"] + row["From"] * np.cos(dip_rad) * np.sin(azm_rad)
        y0 = row["Northing"] + row["From"] * np.cos(dip_rad) * np.cos(azm_rad)
        z0 = row["Elevation"] - row["From"] * np.sin(dip_rad)
        x1 = row["Easting"] + row["To"] * np.cos(dip_rad) * np.sin(azm_rad)
        y1 = row["Northing"] + row["To"] * np.cos(dip_rad) * np.cos(azm_rad)
        z1 = row["Elevation"] - row["To"] * np.sin(dip_rad)
        return pd.Series([x0, y0, z0, x1, y1, z1])

    merged[["x0", "y0", "z0", "x1", "y1", "z1"]] = merged.apply(interval_to_xyz, axis=1)

    for _, row in merged.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row.x0, row.x1], y=[row.y0, row.y1], z=[row.z0, row.z1],
            mode='lines',
            line=dict(
                color=row[selected_element], width=3,
                colorscale='Hot',
                cmin=vmin, cmax=vmax
            ),
            hovertext=f"{row['BHID']}\n{selected_element}: {row[selected_element]:.1f}",
            hoverinfo='text',
            showlegend=False
        ))

    # ✅ Target Suggestion (Furthest high predictions)
    _, dists = pairwise_distances_argmin_min(grid_df[["Easting", "Northing", "Elevation"]], merged[["Easting", "Northing", "Elevation"]])
    grid_df['dist'] = dists
    targets = grid_df.sort_values(by=[f"Predicted_{selected_element}", 'dist'], ascending=[False, False]).head(10)

    fig.add_trace(go.Scatter3d(
        x=targets['Easting'], y=targets['Northing'], z=targets['Elevation'],
        mode='markers+text',
        marker=dict(size=6, color='cyan', symbol='diamond'),
        text=[f"Suggested Drill (ppm: {val:.1f})" for val in targets[f"Predicted_{selected_element}"]],
        name="Suggested Drill Points"
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

    st.markdown("### 📌 Suggested Drill Locations")
    for i, row in targets.iterrows():
        st.markdown(f"- **Easting:** {row['Easting']:.2f}, **Northing:** {row['Northing']:.2f}, **Elevation:** {row['Elevation']:.2f}, **{selected_element}:** {row[f'Predicted_{selected_element}']:.2f} ppm")

    csv = grid_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"📥 Download Predicted {selected_element} CSV",
        csv,
        f"{selected_element.lower()}_predicted_volume.csv",
        "text/csv"
    )
