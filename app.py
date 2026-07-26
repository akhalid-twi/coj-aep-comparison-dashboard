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
if "last_processed_click" not in st.session_state:
    st.session_state.last_processed_click = None


# -----------------------------
# Load Data & Tree (Cached)
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


@st.cache_resource
def build_spatial_tree(lats: np.ndarray, lons: np.ndarray):
    coords_rad = np.radians(np.column_stack([lats, lons]))
    return BallTree(coords_rad, metric="haversine")


tree = build_spatial_tree(gdf["lat"].values, gdf["lon"].values)


@st.cache_data
def load_fema_layer():
    url_fema = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/fema_zones.parquet"
    try:
        with urllib.request.urlopen(url_fema) as response:
            fema_gdf = gpd.read_parquet(BytesIO(response.read()))
        if fema_gdf.crs != "EPSG:4326":
            fema_gdf = fema_gdf.to_crs(epsg=4326)
        return json.loads(fema_gdf.to_json())
    except Exception as e:
        st.warning(f"Could not load FEMA zones parquet: {e}")
        return None


fema_geojson = load_fema_layer()


# -----------------------------
# Cached Base Map Initialization
# -----------------------------
@st.cache_resource
def get_base_map():
    # Base map centered statically on Jacksonville
    m = folium.Map(
        location=[30.33218, -81.65565],
        zoom_start=10,
        tiles="cartodbpositron",
    )

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

    data_points = [
        [float(lat), float(lon)] for lat, lon in zip(gdf["lat"], gdf["lon"])
    ]

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
    folium.LayerControl().add_to(m)
    return m


# -----------------------------
# Layout
# -----------------------------
col_header, col_ctrl = st.columns([6, 2])
with col_header:
    st.title("COJ AEP Comparison Dashboard")
with col_ctrl:
    scenario_option = st.selectbox("Scenario", ["All", "Base", "SLR1", "SLR4"])

col_map, col_plot = st.columns([3, 2])

# ==============================================================================
# MAP COMPONENT
# ==============================================================================
with col_map:
    # Build isolated FeatureGroup for red active marker
    selected_row = gdf.iloc[st.session_state.selected_idx]
    sel_lat = float(selected_row["lat"])
    sel_lon = float(selected_row["lon"])

    marker_fg = folium.FeatureGroup(name="Selected Cell Marker")
    folium.Marker(
        location=[sel_lat, sel_lon],
        popup=f"Cell: {selected_row.get('cell_id', st.session_state.selected_idx)}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(marker_fg)

    # Base map loaded from cache (Static structure)
    base_map = get_base_map()

    map_data = st_folium(
        base_map,
        feature_group_to_add=marker_fg,  # In-place dynamic JavaScript update
        use_container_width=True,
        height=650,
        returned_objects=["last_clicked"],
        key="main_map_static_key",
    )

    # Click handling
    if map_data and map_data.get("last_clicked"):
        current_click = map_data["last_clicked"]
        if current_click != st.session_state.last_processed_click:
            st.session_state.last_processed_click = current_click

            lat_click_rad = np.radians(current_click["lat"])
            lon_click_rad = np.radians(current_click["lng"])
            dist, idx = tree.query([[lat_click_rad, lon_click_rad]], k=1)
            nearest_idx = idx[0][0]

            distance_m = dist[0][0] * 6371000

            if distance_m < 2500 and st.session_state.selected_idx != nearest_idx:
                st.session_state.selected_idx = nearest_idx
                st.rerun()

# ==============================================================================
# PLOT COMPONENT
# ==============================================================================
with col_plot:
    selected_row = gdf.iloc[st.session_state.selected_idx]
    aep_raw = selected_row["aep"]
    aep_data = json.loads(aep_raw) if isinstance(aep_raw, str) else aep_raw

    def filter_aep(aep_dict, option):
        if option == "All":
            return aep_dict
        filtered = {}
        for k, v in aep_dict.items():
            if k in ["SACS", "SACS_RAS", "SWE"]:
                filtered[k] = v
            elif k == "Combined-BiasCorrected" and option == "Base":
                filtered[k] = v
            elif k == "Combined-SLR1-BiasCorrected" and option == "SLR1":
                filtered[k] = v
            elif k == "Combined-SLR4-BiasCorrected" and option == "SLR4":
                filtered[k] = v
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

    raw_bfe = selected_row.get("fema_bfe")
    try:
        fema_bfe = float(raw_bfe) if raw_bfe is not None and not pd.isna(raw_bfe) else None
    except (ValueError, TypeError):
        fema_bfe = None

    bfe_str = f"{fema_bfe:.2f} ft" if fema_bfe is not None else "N/A"
    cell_id_val = selected_row.get("cell_id", st.session_state.selected_idx)
    sacs_id_val = selected_row.get("sacs_id", "N/A")

    st.markdown(
        f"**Cell:** {cell_id_val} | **SACS ID:** {sacs_id_val} | **FEMA BFE (100yr):** {bfe_str}"
    )

    fig = go.Figure()

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

        style = COLOR_MAP.get(label, dict(color="gray", dash="solid", width=2))

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

    for rp in [10, 100, 500, 1000]:
        fig.add_vline(x=rp, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FAFBFC",
        paper_bgcolor="#F5F7FA",
        xaxis=dict(
            type="log",
            title="Return Period (years)",
            range=[np.log10(1), np.log10(10000)],
            tickvals=[2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
            ticktext=["2", "5", "10", "25", "50", "100", "250", "500", "1000", "2000", "5000", "10000"],
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
