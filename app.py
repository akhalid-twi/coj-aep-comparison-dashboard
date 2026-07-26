import json
import urllib.request
from io import BytesIO

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import FastMarkerCluster
from sklearn.neighbors import BallTree
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="COJ AEP Comparison Dashboard",
    layout="wide",
)

# -----------------------------
# Initialize Session State
# -----------------------------
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "map_center" not in st.session_state:
    st.session_state.map_center = [30.33218, -81.65565]  # Jacksonville, FL
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 10
if "map_render_key" not in st.session_state:
    st.session_state.map_render_key = 0


# -----------------------------
# Load Data from Master Parquet (Cached)
# -----------------------------
@st.cache_data
def load_data():
    url_main = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/sacs_aep_comparison_for_dashboard.parquet"

    with urllib.request.urlopen(url_main) as response:
        gdf = gpd.read_parquet(BytesIO(response.read()))

    gdf = gdf.dropna(subset=["lat", "lon"])
    gdf = gdf[np.isfinite(gdf["lat"]) & np.isfinite(gdf["lon"])]
    return gdf


gdf = load_data()


# -----------------------------
# Build BallTree for Spatial Clicks
# -----------------------------
# Change the function signature to accept raw numpy arrays
@st.cache_resource
def build_spatial_tree(lats: np.ndarray, lons: np.ndarray):
    coords_rad = np.radians(np.column_stack([lats, lons]))
    return BallTree(coords_rad, metric="haversine")


# Call it using array values from the GeoDataFrame
tree = build_spatial_tree(gdf["lat"].values, gdf["lon"].values)


# -----------------------------
# Load FEMA GeoParquet Layer (Cached)
# -----------------------------
@st.cache_data
def load_fema_layer():
    url_fema = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/fema_zones.parquet"
    try:
        with urllib.request.urlopen(url_fema) as response:
            fema_gdf = gpd.read_parquet(BytesIO(response.read()))

        # Ensure WGS84 projection for Leaflet
        if fema_gdf.crs != "EPSG:4326":
            fema_gdf = fema_gdf.to_crs(epsg=4326)

        # Convert GeoDataFrame directly to GeoJSON dict structure for Folium
        return json.loads(fema_gdf.to_json())
    except Exception as e:
        st.warning(f"Could not load FEMA zones parquet: {e}")
        return None


fema_geojson = load_fema_layer()


# -----------------------------
# Global Scenario Controls
# -----------------------------
col_header, col_ctrl = st.columns([6, 2])
with col_header:
    st.title("COJ AEP Comparison Dashboard")
with col_ctrl:
    scenario_option = st.selectbox("Scenario", ["All", "Base", "SLR1", "SLR4"])

# Create the two main side-by-side layout columns
col_map, col_plot = st.columns([3, 2])
# ==============================================================================
# INTERACTIVITY CHECK (Placed BEFORE Map Building)
# ==============================================================================
# We check if there's a click in st.session_state from the previous run
# and update selected_idx BEFORE building the map object m.

if "coj_interactive_map" in st.session_state and st.session_state["coj_interactive_map"]:
    last_click = st.session_state["coj_interactive_map"].get("last_clicked")
    if last_click:
        lat_click = last_click["lat"]
        lon_click = last_click["lng"]

        lat_click_rad = np.radians(lat_click)
        lon_click_rad = np.radians(lon_click)
        dist, idx = tree.query([[lat_click_rad, lon_click_rad]], k=1)
        nearest_idx = idx[0][0]

        distance_m = dist[0][0] * 6371000  # Earth radius in meters

        if distance_m < 2500 and st.session_state.selected_idx != nearest_idx:
            st.session_state.selected_idx = nearest_idx
            clicked_row = gdf.iloc[nearest_idx]
            st.session_state.map_center = [
                float(clicked_row["lat"]),
                float(clicked_row["lon"]),
            ]
            st.session_state.map_zoom = 14


# ==============================================================================
# MAP COMPONENT (Left Column)
# ==============================================================================
with col_map:
    # 1. Static base map
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="cartodbpositron",
    )

    # 2. Add static FEMA layer
    if fema_geojson:
        def style_fema(feature):
            zone = str(feature["properties"].get("FLD_ZONE", ""))
            color = "#8E24AA" if "V" in zone else "#0288D1"
            return {
                "fillColor": color,
                "color": color,
                "weight": 1,
                "fillOpacity": 0.25,
            }

        folium.GeoJson(
            fema_geojson,
            name="FEMA Flood Zones",
            style_function=style_fema,
            tooltip=folium.GeoJsonTooltip(
                fields=["FLD_ZONE", "BFE"],
                aliases=["Zone:", "BFE (ft):"],
                localize=True,
            ),
        ).add_to(m)

    # 3. Add static Cluster layer
    data_points = [[float(lat), float(lon)] for lat, lon in zip(gdf["lat"], gdf["lon"])]
    callback_js = """
    function (row) {
        var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
            radius: 3,
            fillColor: '#000000',
            color: '#000000',
            weight: 1,
            fillOpacity: 0.6
        });
        return marker;
    };
    """
    FastMarkerCluster(data=data_points, callback=callback_js).add_to(m)

    # 4. Create a DYNAMIC FeatureGroup for the selected marker
    fg_selected = folium.FeatureGroup(name="Selected Cell Marker")
    selected_row = gdf.iloc[st.session_state.selected_idx]
    
    folium.Marker(
        location=[float(selected_row["lat"]), float(selected_row["lon"])],
        popup=f"Cell: {selected_row.get('cell_id', st.session_state.selected_idx)}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(fg_selected)

    folium.LayerControl().add_to(m)

    # 5. Pass feature_group_to_add, center, and zoom directly to st_folium
    map_data = st_folium(
        m,
        center=st.session_state.map_center,
        zoom=st.session_state.map_zoom,
        feature_group_to_add=fg_selected,
        use_container_width=True,
        height=650,
        returned_objects=["last_clicked"],
        key="coj_interactive_map",
    )
# ==============================================================================
# INTERACTIVITY & STATE UPDATES
# ==============================================================================
if map_data and map_data.get("last_clicked"):
    lat_click = map_data["last_clicked"]["lat"]
    lon_click = map_data["last_clicked"]["lng"]

    # Spatial Lookup
    lat_click_rad = np.radians(lat_click)
    lon_click_rad = np.radians(lon_click)
    dist, idx = tree.query([[lat_click_rad, lon_click_rad]], k=1)
    nearest_idx = idx[0][0]

    earth_radius = 6371000  # meters
    distance_m = dist[0][0] * earth_radius

    # 2500m tolerance threshold
    if distance_m < 2500 and st.session_state.selected_idx != nearest_idx:
        st.session_state.selected_idx = nearest_idx

        clicked_row = gdf.iloc[nearest_idx]
        st.session_state.map_center = [
            float(clicked_row["lat"]),
            float(clicked_row["lon"]),
        ]
        st.session_state.map_zoom = 14
        
        # FIX 2 & 3: Removed map_render_key increment and st.rerun()

# ==============================================================================
# PLOT COMPONENT (Right Column)
# ==============================================================================
selected_row = gdf.iloc[st.session_state.selected_idx]
aep_raw = selected_row["aep"]
aep_data = json.loads(aep_raw) if isinstance(aep_raw, str) else aep_raw


def filter_aep(aep_dict, option):
    if option == "All":
        return aep_dict

    filtered = {}
    for k, v in aep_dict.items():
        # Keep universal benchmarks always
        if k in ["SACS", "SACS_RAS", "SWE"]:
            filtered[k] = v

        # Filter bias-corrected scenarios dynamically
        elif k == "Combined-BiasCorrected" and option == "Base":
            filtered[k] = v
        elif k == "Combined-SLR1-BiasCorrected" and option == "SLR1":
            filtered[k] = v
        elif k == "Combined-SLR4-BiasCorrected" and option == "SLR4":
            filtered[k] = v

        # Filter standard uncorrected scenarios
        elif option in k and "BiasCorrected" not in k:
            filtered[k] = v

    return filtered


aep_filtered = filter_aep(aep_data, scenario_option)


COLOR_MAP = {
    "SACS": dict(color="#000000", dash="solid", width=3),
    "SACS_RAS": dict(color="#666666", dash="solid", width=3),
    "NTC-Syn-Base": dict(color="#4CAF50", dash="dot", width=2),
    "NTC-Syn-SLR1": dict(color="#2E7D32", dash="dot", width=3),
    "NTC-Syn-SLR4": dict(color="#1B5E20", dash="dot", width=4),
    "TC-OS-Base": dict(color="#42A5F5", dash="dash", width=2),
    "TC-OS-SLR1": dict(color="#1E88E5", dash="dash", width=3),
    "TC-OS-SLR4": dict(color="#0D47A1", dash="dash", width=4),
    "Combined-Base": dict(color="#FFB74D", dash="solid", width=2),
    "Combined-SLR1": dict(color="#F57C00", dash="solid", width=3),
    "Combined-SLR4": dict(color="#D84315", dash="solid", width=4),
    "Combined-BiasCorrected": dict(color="#D32F2F", dash="solid", width=4),
    "Combined-SLR1-BiasCorrected": dict(color="#E65100", dash="solid", width=4),
    "Combined-SLR4-BiasCorrected": dict(color="#880E4F", dash="solid", width=4),
}

LABEL_MAP = {
    "SACS": "SACS_ADCIRC_CC_Full_set",
    "SACS_RAS": "SACS_RAS_TC_506_storms",
    "SWE": "SWE Baseline",
    "Combined-BiasCorrected": "Combined-Base (Bias Corrected)",
    "Combined-SLR1-BiasCorrected": "Combined-SLR1 (Bias Corrected)",
    "Combined-SLR4-BiasCorrected": "Combined-SLR4 (Bias Corrected)",
}

with col_plot:
    # Safely Extract and Parse FEMA 100-Year BFE
    raw_bfe = selected_row.get("fema_bfe")

    try:
        if raw_bfe is not None and not pd.isna(raw_bfe):
            fema_bfe = float(raw_bfe)
        else:
            fema_bfe = None
    except (ValueError, TypeError):
        fema_bfe = None

    bfe_str = f"{fema_bfe:.2f} ft" if fema_bfe is not None else "N/A"
    cell_id_val = selected_row.get("cell_id", st.session_state.selected_idx)
    sacs_id_val = selected_row.get("sacs_id", "N/A")

    st.markdown(
        f"**Cell:** {cell_id_val} | **SACS ID:** {sacs_id_val} | **FEMA BFE (100yr):** {bfe_str}"
    )

    fig = go.Figure()

    # Plot Return Period Curves
    for label, data in aep_filtered.items():
        if not data:
            continue

        display_label = LABEL_MAP.get(label, label)

        parsed_points = [
            (float(k), float(v))
            for k, v in data.items()
            if v is not None and not pd.isna(v)
        ]

        if not parsed_points:
            continue

        parsed_points.sort(key=lambda item: item[0])
        x = [pt[0] for pt in parsed_points]
        y = [pt[1] for pt in parsed_points]

        style = COLOR_MAP.get(
            label, dict(color="gray", dash="solid", width=2)
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=display_label,
                line=dict(
                    color=style["color"],
                    dash=style["dash"],
                    width=style.get("width", 2),
                ),
                marker=dict(size=4),
            )
        )

    # Add FEMA 100-Year BFE Horizontal Reference Line
    if fema_bfe is not None:
        fig.add_hline(
            y=fema_bfe,
            line_dash="dashdot",
            line_color="#9C27B0",
            line_width=2.5,
            annotation_text=f"FEMA 100-Yr BFE ({fema_bfe:.2f} ft)",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#9C27B0"),
        )

    # Add vertical benchmark lines for key return periods
    for rp in [10, 100, 500, 1000]:
        fig.add_vline(
            x=rp, line_dash="dash", line_color="gray", opacity=0.4
        )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FAFBFC",
        paper_bgcolor="#F5F7FA",
        xaxis=dict(
            type="log",
            title="Return Period (years)",
            range=[np.log10(1), np.log10(10000)],
            tickvals=[
                2,
                5,
                10,
                25,
                50,
                100,
                250,
                500,
                1000,
                2000,
                5000,
                10000,
            ],
            ticktext=[
                "2",
                "5",
                "10",
                "25",
                "50",
                "100",
                "250",
                "500",
                "1000",
                "2000",
                "5000",
                "10000",
            ],
            gridcolor="#E0E6ED",
        ),
        yaxis=dict(title="WSE (ft, NAVD88)", gridcolor="#E0E6ED"),
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            title="Scenario",
            orientation="h",
            y=1.02,
            x=0.0,
            xanchor="left",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
