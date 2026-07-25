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


# -----------------------------
# Load Data from Master Parquet (Cached)
# -----------------------------
@st.cache_data
def load_data():
    url_main = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/sacs_aep_comparison_for_dashboard.parquet"

    with urllib.request.urlopen(url_main) as response:
        gdf = gpd.read_parquet(BytesIO(response.read()))

    # Clean missing or non-finite coordinates to prevent Leaflet / Folium crashes
    gdf = gdf.dropna(subset=["lat", "lon"])
    gdf = gdf[np.isfinite(gdf["lat"]) & np.isfinite(gdf["lon"])]

    return gdf


gdf = load_data()


# -----------------------------
# Spatial Index Tree (Cached)
# -----------------------------
@st.cache_resource
def get_ball_tree(_df):
    coords_rad = np.radians(np.vstack([_df["lat"], _df["lon"]]).T)
    return BallTree(coords_rad, metric="haversine")


tree = get_ball_tree(gdf)

# -----------------------------
# App Setup & Styles
# -----------------------------
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background-color: #EEF2F6; }
[data-testid="stMainBlockContainer"] {
    background-color: #F5F7FA;
    padding-top: 2rem;
    padding-bottom: 2rem;
    border-radius: 10px;
}
body { color: #1F2937; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h2 style='text-align: center;'>COJ AEP Comparison Dashboard</h2>",
    unsafe_allow_html=True,
)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
jack_lat = float(gdf["lat"].mean())
jack_lon = float(gdf["lon"].mean())

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

if "map_center" not in st.session_state:
    st.session_state.map_center = [jack_lat, jack_lon]

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 10

if "map_render_key" not in st.session_state:
    st.session_state.map_render_key = 0

# -----------------------------
# Global Scenario Controls
# -----------------------------
col1, col2 = st.columns([6, 1])
with col2:
    scenario_option = st.selectbox("Scenario", ["All", "Base", "SLR1", "SLR4"])

# Layout setup
col_map, col_plot = st.columns([3, 2])


# ==============================================================================
# MAP COMPONENT (Left Column)
# ==============================================================================
with col_map:
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="cartodbpositron",
    )

    # Convert coordinates to explicit Python floats to keep Leaflet fast
    data_points = [
        [float(lat), float(lon)] for lat, lon in zip(gdf["lat"], gdf["lon"])
    ]

    callback_js = """
    function (row) {
        var marker = L.circleMarker(new L.LatLng(row[0], row[1]), {
            radius: 3,
            fillColor: '#0072B2',
            color: '#0072B2',
            weight: 1,
            fillOpacity: 0.6
        });
        return marker;
    };
    """

    FastMarkerCluster(data=data_points, callback=callback_js).add_to(m)

    # Highlight active selected cell
    selected_row = gdf.iloc[st.session_state.selected_idx]

    sel_lat = float(selected_row["lat"])
    sel_lon = float(selected_row["lon"])

    folium.Marker(
        location=[sel_lat, sel_lon],
        popup=f"Cell: {selected_row.cell_id}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    map_data = st_folium(
        m,
        center=st.session_state.map_center,
        zoom=st.session_state.map_zoom,
        use_container_width=True,
        height=650,
        returned_objects=["last_clicked"],
        key=f"map_instance_{st.session_state.map_render_key}",
    )

# ==============================================================================
# INTERACTIVITY & DISCRETE RERUN SIGNALING
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

    # 2500m tolerance threshold to make clicking comfortable when zoomed out
    if distance_m < 2500 and st.session_state.selected_idx != nearest_idx:
        st.session_state.selected_idx = nearest_idx

        clicked_row = gdf.iloc[nearest_idx]
        st.session_state.map_center = [
            float(clicked_row["lat"]),
            float(clicked_row["lon"]),
        ]
        st.session_state.map_zoom = 14
        st.session_state.map_render_key += 1
        st.rerun()

# ==============================================================================
# PLOT COMPONENT (Right Column)
# ==============================================================================
selected_row = gdf.iloc[st.session_state.selected_idx]
aep_data = json.loads(selected_row["aep"])


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
    #"SWE": dict(color="#9E9E9E", dash="dashdot", width=2),
    # NTC (green family)
    "NTC-Syn-Base": dict(color="#4CAF50", dash="dot", width=2),
    "NTC-Syn-SLR1": dict(color="#2E7D32", dash="dot", width=3),
    "NTC-Syn-SLR4": dict(color="#1B5E20", dash="dot", width=4),
    # TC (blue family)
    "TC-OS-Base": dict(color="#42A5F5", dash="dash", width=2),
    "TC-OS-SLR1": dict(color="#1E88E5", dash="dash", width=3),
    "TC-OS-SLR4": dict(color="#0D47A1", dash="dash", width=4),
    # Combined Uncorrected (orange/red family)
    "Combined-Base": dict(color="#FFB74D", dash="solid", width=2),
    "Combined-SLR1": dict(color="#F57C00", dash="solid", width=3),
    "Combined-SLR4": dict(color="#D84315", dash="solid", width=4),
    # Combined Bias Corrected (distinct bold long-dashed lines)
    "Combined-BiasCorrected": dict(
        color="#D32F2F", dash="solid", width=4
    ),  # Red
    "Combined-SLR1-BiasCorrected": dict(
        color="#E65100", dash="solid", width=4
    ),  # Dark Orange
    "Combined-SLR4-BiasCorrected": dict(
        color="#880E4F", dash="solid", width=4
    ),  # Deep Crimson
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
    st.markdown(
        f"**Cell:** {selected_row.cell_id}  \n**SACS ID:** {selected_row.sacs_id}"
    )

    fig = go.Figure()

    for label, data in aep_filtered.items():
        if not data:
            continue

        display_label = LABEL_MAP.get(label, label)

        # Parse return periods (x) and WSE (y) while filtering out nulls/NaNs
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
