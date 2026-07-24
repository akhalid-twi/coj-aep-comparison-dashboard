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

gdf_main, gdf_ras, gdf_bc  = load_data()

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
gdf_bc_lookup = gdf_bc.set_index("point_id")

merged_aep = []

for idx, row in gdf_main.iterrows():

    sacs_id = str(row.sacs_id)

    aep_json = row["aep"]

    # Merge RAS TC dataset
    if sacs_id in gdf_ras_lookup.index:
        aep_json = merge_aep(
            aep_json,
            gdf_ras_lookup.loc[sacs_id]["aep"]
        )

    # Merge bias corrected dataset
    if sacs_id in gdf_bc_lookup.index:
        aep_json = merge_aep(
            aep_json,
            gdf_bc_lookup.loc[sacs_id]["aep"]
        )

    merged_aep.append(aep_json)

gdf_main["aep"] = merged_aep

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


# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
# Calculate exact mean lat/lon for Jacksonville center
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
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="cartodbpositron"
    )

    # STRICTLY extract as [[latitude, longitude], ...]
    # Using .to_numpy() / .tolist() avoids any named tuple index offset bugs
    data_points = gdf[["lat", "lon"]].to_numpy().tolist()

    # JS Callback explicitly assigning row[0] -> Lat, row[1] -> Lng
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
    
    # Explicitly pull lat and lon as floats
    sel_lat = float(selected_row["lat"])
    sel_lon = float(selected_row["lon"])

    folium.Marker(
        location=[sel_lat, sel_lon],
        popup=f"Cell: {selected_row.cell_id}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    map_data = st_folium(
        m,
        center=st.session_state.map_center,
        zoom=st.session_state.map_zoom,
        use_container_width=True,
        height=650,
        key=f"map_instance_{st.session_state.map_render_key}"
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

    # Show everything
    if option == "All":
        return aep_dict

    filtered = {}

    for k, v in aep_dict.items():

        # Always keep reference datasets
        if k in ["SACS", "SACS_RAS"]:
            filtered[k] = v

        # Only show bias-corrected data for Base view
        elif k == "Combined-BiasCorrected":
            if option == "Base":
                filtered[k] = v

        # Normal scenario filtering
        elif option in k:
            filtered[k] = v

    return filtered


aep_filtered = filter_aep(aep_data, scenario_option)


COLOR_MAP = {
    "SACS": dict(color="#000000", dash="solid", width=3,marker=None),
    "SACS_RAS": dict(color="#666666", dash="solid", width=3,marker=None),

    # NTC (green family)
    "NTC-Syn-Base": dict(color="#4CAF50", dash="dot", width=2,marker=None),
    "NTC-Syn-SLR1": dict(color="#2E7D32", dash="dot", width=3,marker=None),
    "NTC-Syn-SLR4": dict(color="#1B5E20", dash="dot", width=4,marker=None),

    # TC (blue family)
    "TC-OS-Base": dict(color="#42A5F5", dash="dash", width=2,marker=None),
    "TC-OS-SLR1": dict(color="#1E88E5", dash="dash", width=3,marker=None),
    "TC-OS-SLR4": dict(color="#0D47A1", dash="dash", width=4,marker=None),

    # Combined (orange/red family)
    "Combined-Base": dict(color="#FFB74D", dash="solid", width=2,marker=None),
    "Combined-SLR1": dict(color="#F57C00", dash="solid", width=3,marker=None),
    "Combined-SLR4": dict(color="#D84315", dash="solid", width=4,marker=None),

        
    # Bias corrected
    "Combined-BiasCorrected": dict(
        color="#FF0000",
        dash="longdash",
        width=5
    ),

}



LABEL_MAP = {
    "SACS": "SACS_ADCIRC_CC_Full_set",
    "SACS_RAS": "SACS_RAS_TC_506_storms",
    "Combined-BiasCorrected": "Combined-Base (Bias Corrected)"
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
            x=x,
            y=y,
            mode="lines",
            name=display_label,
            line=dict(
                color=style["color"],
                dash=style["dash"],
                width=style.get("width", 2)
            )
        ))

    for rp in [10, 100, 500, 1000]:
        fig.add_vline(x=rp, line_dash="dash", line_color="gray", opacity=0.6)

    fig.update_layout(
        template="plotly_white", plot_bgcolor="#FAFBFC", paper_bgcolor="#F5F7FA",
        
        # xaxis=dict(type="log", title="Return Period (years)", range=[np.log10(2), np.log10(2000)],
        #           tickvals=[2,5,10,25,50,100,250,500,1000,2000], gridcolor="#E0E6ED"),
        xaxis=dict(
            type="log",
            title="Return Period (years)",
            range=[np.log10(1), np.log10(10000)],
            tickvals=[2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
            ticktext=["2", "5", "10", "25", "50", "100", "250", "500", "1000", "2000", "5000", "10000"],
            gridcolor="#E0E6ED"
        ),

        
        yaxis=dict(title="WSE (ft, NAVD88)", gridcolor="#E0E6ED"),
        height=600, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(title="Scenario", orientation="h", y=1.02, x=0.1, xanchor="left"),
    )

    st.plotly_chart(fig, use_container_width=True)
