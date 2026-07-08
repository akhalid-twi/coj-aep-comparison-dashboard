import streamlit as st
import geopandas as gpd
import pandas as pd
import json
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import numpy as np
from sklearn.neighbors import BallTree
from folium.plugins import FastMarkerCluster
import urllib.request
from io import BytesIO


# -----------------------------
# Load Data from GitHub (Cached)
# -----------------------------
@st.cache_data
def load_data():

    url_main = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/sacs_aep_comparison_for_dashboard.parquet"

    url_ras = "https://github.com/akhalid-twi/coj-aep-comparison-dashboard/raw/main/assets/sacs_ras_tc_aep.parquet"

    # --- load main dataset ---
    with urllib.request.urlopen(url_main) as response:
        gdf_main = gpd.read_parquet(BytesIO(response.read()))

    # --- load RAS dataset ---
    with urllib.request.urlopen(url_ras) as response:
        gdf_ras = gpd.read_parquet(BytesIO(response.read()))

    return gdf_main, gdf_ras

gdf_main, gdf_ras = load_data()

# -----------------------------
# Merge dicts
# -----------------------------

def merge_aep(main_json, ras_json):
    aep_main = json.loads(main_json)

    if pd.isna(ras_json):
        return json.dumps(aep_main)

    try:
        aep_ras = json.loads(ras_json)
        aep_main.update(aep_ras)
    except:
        pass

    return json.dumps(aep_main)

# -----------------------------
# Merge on cell_id
# -----------------------------

gdf_ras_lookup = gdf_ras.set_index("sacs_id")

merged_aep = []

for idx, row in gdf_main.iterrows():
    sacs_id = str(row.sacs_id)
    
    if sacs_id in gdf_ras_lookup.index:
        ras_json = gdf_ras_lookup.loc[sacs_id]["aep"]
    else:
        ras_json = None

    merged_aep.append(merge_aep(row["aep"], ras_json))

gdf_main["aep"] = merged_aep

# This becomes your working dataset
gdf = gdf_main


# -----------------------------
# 2. Spatial Index Tree (Cached)
# -----------------------------
@st.cache_resource
def get_ball_tree(_df):
    coords_rad = np.radians(np.vstack([_df["lat"], _df["lon"]]).T)
    return BallTree(coords_rad, metric="haversine")

tree = get_ball_tree(gdf)

# -----------------------------
# 3. App Setup & Styles
# -----------------------------
st.set_page_config(layout="wide")

st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>COJ AEP Comparison Dashboard</h2>", unsafe_allow_html=True)

# -----------------------------
# 4. Session State Initialization
# -----------------------------
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

# Store structural configurations to drive redrawing only when forced
if "map_center" not in st.session_state:
    st.session_state.map_center = [float(gdf["lat"].mean()), float(gdf["lon"].mean())]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 10
if "map_render_key" not in st.session_state:
    st.session_state.map_render_key = 0

# -----------------------------
# 5. Global Scenario Controls
# -----------------------------
col1, col2 = st.columns([6, 1])
with col2:
    scenario_option = st.selectbox("Scenario", ["All", "Base", "SLR1", "SLR4"])

# Layout setup
col_map, col_plot = st.columns([3, 2])

# ==============================================================================
# 6. MAP COMPONENT (Left Column)
# ==============================================================================
with col_map:
    # We pass fixed values here so native browser panning never triggers Python loop changes
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="cartodbpositron"
    )

    # Canvas-based quick performance rendering
    data_points = list(zip(gdf["lat"], gdf["lon"]))
    callback_js = """
        function (row) {
            var circle = L.circleMarker(new L.LatLng(row[0], row[1]), {
                radius: 2,
                color: '#0072B2',
                fillOpacity: 0.5
            });
            return circle;
        };
    """
    FastMarkerCluster(data=data_points, callback=callback_js).add_to(m)

    # Draw highlighted marker over current point
    selected_row = gdf.iloc[st.session_state.selected_idx]
    folium.Marker(
        location=[selected_row["lat"], selected_row["lon"]],
        popup=f"Cell: {selected_row.cell_id}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Render mapping instance using exact keyword control parameters
    map_data = st_folium(
        m,
        center=st.session_state.map_center,
        zoom=st.session_state.map_zoom,
        use_container_width=True,
        height=650,
        key=f"map_instance_{st.session_state.map_render_key}" # Dynamic key locks rendering loops
    )

# ==============================================================================
# 7. INTERACTIVITY & DISCRETE RERUN SIGNALING
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

    # ONLY adjust maps and invoke execution reruns if a new point is targeted
    if distance_m < 500 and st.session_state.selected_idx != nearest_idx:
        st.session_state.selected_idx = nearest_idx
        
        # Center the map precisely on the coordinates of the selected feature 
        clicked_row = gdf.iloc[nearest_idx]
        st.session_state.map_center = [float(clicked_row["lat"]), float(clicked_row["lon"])]
        
        # Pull zoom setting cleanly up to a comfortable close-up look
        st.session_state.map_zoom = 14
        
        # Incrementing the key instructs Streamlit to cleanly build the marker close-up smoothly
        st.session_state.map_render_key += 1
        st.rerun()

# ==============================================================================
# 8. PLOT COMPONENT (Right Column)
# ==============================================================================
selected_row = gdf.iloc[st.session_state.selected_idx]
aep_data = json.loads(selected_row["aep"])


def filter_aep(aep_dict, option):
    if option == "All":
        return aep_dict

    return {
        k: v for k, v in aep_dict.items()
        if k in ["SACS", "SACS_RAS"] or option in k
    }

aep_filtered = filter_aep(aep_data, scenario_option)

COLOR_MAP = {
    "SACS": dict(color="#000000", dash="solid", width=4, marker="circle"),
    "SACS_RAS": dict(color="#666666", dash="solid", width=3, marker="x"),

    # NTC (light green family)
    "NTC-Syn-Base": dict(color="#D9F0D3", dash="solid", marker="circle"),
    "NTC-Syn-SLR1": dict(color="#A6DBA0", dash="solid", marker="square"),
    "NTC-Syn-SLR4": dict(color="#5AAE61", dash="solid", marker="diamond"),

    # TC (blue family)
    "TC-OS-Base": dict(color="#C6DBEF", dash="solid", marker="circle"),
    "TC-OS-SLR1": dict(color="#6BAED6", dash="solid", marker="square"),
    "TC-OS-SLR4": dict(color="#2171B5", dash="solid", marker="diamond"),

    # Combined (strong orange/red family)
    "Combined-Base": dict(color="#FDBB84", dash="solid", marker="circle"),
    "Combined-SLR1": dict(color="#FC8D59", dash="solid", marker="square"),
    "Combined-SLR4": dict(color="#D7301F", dash="solid", marker="diamond"),
}

LABEL_MAP = {
    "SACS": "SACS_ADCIRC_CC_Full_set",
    "SACS_RAS": "SACS_RAS_TC_506_storms"
}


with col_plot:
    st.markdown(f"**Cell:** {selected_row.cell_id}  \n**SACS ID:** {selected_row.sacs_id}")

    fig = go.Figure()
    for label, data in aep_filtered.items():
        
        display_label = LABEL_MAP.get(label, label)
        
        x = sorted([float(k) for k in data.keys()])
        y = [float(data[str(k)]) if str(k) in data else float(data[k]) for k in x]
        style = COLOR_MAP.get(label, dict(color="gray", dash="solid", marker="circle"))

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=display_label,
            line=dict(color=style["color"], dash=style["dash"], width=style.get("width", 2)),
            marker=dict(size=5, symbol=style["marker"])
        ))

    for rp in [10, 100, 500, 1000]:
        fig.add_vline(x=rp, line_dash="dash", line_color="gray", opacity=0.6)

    fig.update_layout(
        template="plotly_white", plot_bgcolor="#FAFBFC", paper_bgcolor="#F5F7FA",
        xaxis=dict(type="log", title="Return Period (years)", range=[np.log10(2), np.log10(2000)],
                  tickvals=[2,5,10,25,50,100,250,500,1000,2000], gridcolor="#E0E6ED"),
        yaxis=dict(title="WSE (ft)", gridcolor="#E0E6ED"),
        height=600, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(title="Scenario", orientation="h", y=1.02, x=0.1, xanchor="left"),
    )

    st.plotly_chart(fig, use_container_width=True)
