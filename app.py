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
# Load Data from GitHub (Cached)
# -----------------------------
@st.cache_data
def load_data():
    url_main = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/sacs_aep_comparison_for_dashboard.parquet"
    url_ras = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/sacs_ras_tc_aep.parquet"
    url_bc = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/combined_bias_corrected_aep.parquet"

    # --- load main dataset ---
    with urllib.request.urlopen(url_main) as response:
        gdf_main = gpd.read_parquet(BytesIO(response.read()))

    # --- load bias corrected dataset ---
    with urllib.request.urlopen(url_bc) as response:
        gdf_bc = gpd.read_parquet(BytesIO(response.read()))

    # --- load RAS dataset ---
    with urllib.request.urlopen(url_ras) as response:
        gdf_ras = gpd.read_parquet(BytesIO(response.read()))

    return gdf_main, gdf_ras, gdf_bc


gdf_main, gdf_ras, gdf_bc = load_data()


# -----------------------------
# Merge dicts helper
# -----------------------------
def merge_aep(main_json, extra_json):
    if isinstance(main_json, str):
        aep_main = json.loads(main_json)
    else:
        aep_main = main_json if main_json is not None else {}

    if pd.isna(extra_json) or extra_json is None:
        return json.dumps(aep_main)

    try:
        if isinstance(extra_json, str):
            aep_extra = json.loads(extra_json)
        else:
            aep_extra = extra_json
        aep_main.update(aep_extra)
    except Exception:
        pass

    return json.dumps(aep_main)

# -----------------------------
# Merge on cell_id / sacs_id
# -----------------------------

# Create lookup dictionaries with integer-normalized string keys
def build_lookup_dict(df, id_col):
    lookup = {}
    for _, r in df.iterrows():
        raw_id = r[id_col]
        if pd.notna(raw_id):
            # Clean floats like 308214.0 -> '308214'
            try:
                key_str = str(int(float(raw_id)))
            except ValueError:
                key_str = str(raw_id).strip()
            lookup[key_str] = r["aep"]
    return lookup

gdf_ras_lookup = build_lookup_dict(gdf_ras, "sacs_id")
gdf_bc_lookup = build_lookup_dict(gdf_bc, "point_id")

merged_aep = []

for idx, row in gdf_main.iterrows():
    # Normalize current row keys to integer strings
    try:
        sacs_id = str(int(float(row.sacs_id))) if pd.notna(row.sacs_id) else ""
    except ValueError:
        sacs_id = str(row.sacs_id).strip()

    try:
        cell_id = str(int(float(row.cell_id))) if pd.notna(row.cell_id) else ""
    except ValueError:
        cell_id = str(row.cell_id).strip()

    aep_json = row["aep"]

    # 1. Merge RAS TC dataset on sacs_id
    if sacs_id in gdf_ras_lookup:
        aep_json = merge_aep(aep_json, gdf_ras_lookup[sacs_id])

    # 2. Merge bias corrected dataset on cell_id (point_id)
    if cell_id in gdf_bc_lookup:
        aep_json = merge_aep(aep_json, gdf_bc_lookup[cell_id])

    merged_aep.append(aep_json)

gdf_main["aep"] = merged_aep
gdf = gdf_main

# -----------------------------
# Spatial Index Tree (Cached)
# -----------------------------
@st.cache_resource
def get_ball_tree(_df):
    coords_rad = np.radians(np.vstack([_df["lat"], _df["lon"]]).T)
    return BallTree(coords_rad, metric="haversine")

popup=f"Cell: {selected_row['cell_id']}"
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

with col_plot:
    st.markdown(f"**Cell:** {selected_row['cell_id']}  \n**SACS ID:** {selected_row['sacs_id']}")
    
    # DIAGNOSTIC PRINT: Check exact key names inside the merged JSON
    st.write("Keys found in cell AEP:", list(aep_data.keys()))
# ==============================================================================
# MAP COMPONENT (Left Column)
# ==============================================================================
with col_map:
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="cartodbpositron",
    )

    data_points = gdf[["lat", "lon"]].to_numpy().tolist()

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

    if distance_m < 500 and st.session_state.selected_idx != nearest_idx:
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
        if k in ["SACS", "SACS_RAS"]:
            filtered[k] = v
        elif k == "Combined-BiasCorrected":
            if option == "Base":
                filtered[k] = v
        elif option in k:
            filtered[k] = v

    return filtered


aep_filtered = filter_aep(aep_data, scenario_option)

COLOR_MAP = {
    "SACS": dict(color="#000000", dash="solid", width=3),
    "SACS_RAS": dict(color="#666666", dash="solid", width=3),
    # NTC
    "NTC-Syn-Base": dict(color="#4CAF50", dash="dot", width=2),
    "NTC-Syn-SLR1": dict(color="#2E7D32", dash="dot", width=3),
    "NTC-Syn-SLR4": dict(color="#1B5E20", dash="dot", width=4),
    # TC
    "TC-OS-Base": dict(color="#42A5F5", dash="dash", width=2),
    "TC-OS-SLR1": dict(color="#1E88E5", dash="dash", width=3),
    "TC-OS-SLR4": dict(color="#0D47A1", dash="dash", width=4),
    # Combined
    "Combined-Base": dict(color="#FFB74D", dash="solid", width=2),
    "Combined-SLR1": dict(color="#F57C00", dash="solid", width=3),
    "Combined-SLR4": dict(color="#D84315", dash="solid", width=4),
    # Bias corrected
    "Combined-BiasCorrected": dict(
        color="#FF0000", dash="longdash", width=4
    ),
}

LABEL_MAP = {
    "SACS": "SACS_ADCIRC_CC_Full_set",
    "SACS_RAS": "SACS_RAS_TC_506_storms",
    "Combined-BiasCorrected": "Combined-Base (Bias Corrected)",
}

with col_plot:
    st.markdown(
        f"**Cell:** {selected_row.cell_id}  \n**SACS ID:** {selected_row.sacs_id}"
    )

    fig = go.Figure()
    for label, data in aep_filtered.items():
        display_label = LABEL_MAP.get(label, label)

        # Robust extraction for float and string keys
        x = sorted([float(k) for k in data.keys()])
        y = []
        for val_x in x:
            str_key_int = str(int(val_x)) if val_x.is_integer() else str(val_x)
            str_key_float = str(val_x)

            if str_key_int in data:
                y.append(float(data[str_key_int]))
            elif str_key_float in data:
                y.append(float(data[str_key_float]))
            elif val_x in data:
                y.append(float(data[val_x]))

        style = COLOR_MAP.get(label, dict(color="gray", dash="solid", width=2))

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=display_label,
                line=dict(
                    color=style["color"],
                    dash=style["dash"],
                    width=style.get("width", 2),
                ),
            )
        )

    for rp in [10, 100, 500, 1000]:
        fig.add_vline(x=rp, line_dash="dash", line_color="gray", opacity=0.6)

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FAFBFC",
        paper_bgcolor="#F5F7FA",
        xaxis=dict(
            type="log",
            title="Return Period (years)",
            range=[np.log10(1), np.log10(10000)],
            tickvals=[2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
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
            title="Scenario", orientation="h", y=1.02, x=0.1, xanchor="left"
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
