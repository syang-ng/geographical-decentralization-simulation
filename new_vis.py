import argparse
import dash
import json
import math
import pandas as pd
import plotly.graph_objects as go

from dash import State, dcc, html
from dash.dependencies import Input, Output
from sklearn.neighbors import NearestNeighbors

# Assuming these are available in your environment
from distribution import *
from measure import *

from dash import callback_context
import urllib.request
import os


space = SphericalSpace()


# --- Helper Function for Distance ---
# We need this for the density calculation.
def geodesic_distance(p1, p2):
    """Calculates the geodesic distance between two points on a unit sphere."""
    dotp = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
    dotp = max(-1.0, min(1.0, dotp))  # Clamp for numerical safety
    return math.acos(dotp)


# --- Pre-load Data ---
def load_data(file_path):
    """Load the slot data from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run the simulation first.")
        return []


def latlon_to_xyz(lat, lon):
    """Convert in‐place, as before."""
    phi = math.radians(lat)
    theta = math.radians(lon)
    return (
        math.cos(phi) * math.cos(theta),
        math.cos(phi) * math.sin(theta),
        math.sin(phi),
    )


def create_app(
    all_slot_data, relay_data, mev_series, attest_series, proposal_time_series
):
    n_slots = len(all_slot_data)

    # ---------------------------------------------------------
    #  1.  Pre-compute per-slot metric history (once)
    # ---------------------------------------------------------
    (
        clusters_hist,
        total_dist_hist,
        avg_nnd_hist,
        nni_hist,
        mev_hist,
        attest_hist,
        proposal_time_hist,
    ) = ([], [], [], [], [], [], [])

    granularity = 10
    # placeholders for “last seen” metrics
    last_c = last_t = last_a = last_nni = last_mev = last_attest = (
        last_proposal_time
    ) = 0.0

    for i, pts in enumerate(all_slot_data):
        if i % granularity == 0:
            # do the full n×n compute only on multiples of `granularity`
            dm = init_distance_matrix(
                pts, space  # , gcp_zones=gcp_zones, gcp_latency=gcp_latency
            )
            if len(pts) > 1:
                last_c = cluster_matrix(dm)
                last_t = total_distance(dm)
                last_a = average_nearest_neighbor_distance(dm)
                last_nni = nearest_neighbor_index_spherical(dm, space)[0]
                last_mev = sum(mev_series[i]) if mev_series else 0.0
                last_attest = sum(attest_series[i])
                last_proposal_time = (
                    sum(t for t in proposal_time_series[i] if t > 0)
                    if proposal_time_series[i]
                    else 0.0
                )

            else:
                last_c = 0
                last_t = last_a = last_nni = last_mev = last_attest = (
                    last_proposal_time
                ) = 0.0

        # *every* slot, append whatever the “last computed” values are
        clusters_hist.append(last_c)
        total_dist_hist.append(last_t)
        avg_nnd_hist.append(last_a)
        nni_hist.append(last_nni)
        mev_hist.append(last_mev)
        attest_hist.append(last_attest)
        proposal_time_hist.append(last_proposal_time)

    # ---------------------------------------------------------
    #  2.  Helper: build the 3-D density + relay figure
    # ---------------------------------------------------------
    def build_density_fig(points, relay_pts):
        # ----- local density via fixed-radius neighbours -----
        if points:
            radius = 0.2
            nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree").fit(points)
            neigh = nbrs.radius_neighbors(points, return_distance=False)
            density = np.array([len(n) - 1 for n in neigh], dtype=float)
            density = np.clip(density, 0, None)
            x, y, z = zip(*points)
        else:  # shouldn’t happen, but be safe
            density = []
            x = y = z = []

        fig = go.Figure(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=4,
                    color=density,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title="Local Density",
                        # push the colorbar out of the way:
                        x=1.15,  # → 115% of the plot width
                        y=0.5,  # center vertically
                        len=0.7,  # 70% of the plot height
                    ),
                ),
                name="Validators",
            )
        )

        # now – add a low-opacity wireframe sphere
        phi = np.linspace(0, np.pi, 40)
        theta = np.linspace(0, 2 * np.pi, 80)
        phi, theta = np.meshgrid(phi, theta)

        # unit‐sphere coordinates
        xs = np.sin(phi) * np.cos(theta)
        ys = np.sin(phi) * np.sin(theta)
        zs = np.cos(phi)

        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                showscale=False,
                opacity=0.2,
                colorscale=[[0, "lightblue"], [1, "lightblue"]],  # uniform light‐blue
                name="Earth",
                hoverinfo="skip",
            )
        )
        # Load GeoJSON of countries (low‐res version for speed)
        if not os.path.exists("./data/world_countries.geo.json"):
            url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
            with urllib.request.urlopen(url) as resp:
                world = json.load(resp)
                # Save the world data to a local file for future use
                with open("./data/world_countries.geo.json", "w") as f:
                    json.dump(world, f)
        else:
            with open("./data/world_countries.geo.json", "r") as f:
                world = json.load(f)

        # add country boundary lines
        for feature in world["features"]:
            geom = feature["geometry"]
            # handle both Polygons and MultiPolygons
            polygons = geom["coordinates"]
            if geom["type"] == "Polygon":
                polygons = [polygons]
            for poly in polygons:
                # each poly is a list of rings; we only need the outer ring (first)
                ring = poly[0]
                lons, lats = zip(*ring)
                # convert to xyz
                xyz = [latlon_to_xyz(lat, lon) for lat, lon in zip(lats, lons)]
                xs, ys, zs = map(list, zip(*xyz))
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(color="white", width=1),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # ----- overlay relay position(s) ---------------------
        if relay_pts:
            # promote single tuple to list-of-tuple
            if isinstance(relay_pts[0], (int, float)):
                relay_pts = [relay_pts]
            rx, ry, rz = zip(*relay_pts)
            fig.add_trace(
                go.Scatter3d(
                    x=rx,
                    y=ry,
                    z=rz,
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color="red",
                        symbol="diamond",
                        line=dict(color="black", width=1),
                    ),
                    text=["Relay"] * len(rx),
                    textposition="top center",
                    name="Relay",
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[-1, 1]),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=160, b=0, t=30),  # make room on the right
            legend=dict(
                x=0.02,  # near left edge of the plotting area
                y=0.98,  # top
                bgcolor="rgba(255,255,255,0.7)",
            ),
        )
        return fig

    # ---------------------------------------------------------
    #  3.  Build the Dash app & layout
    # ---------------------------------------------------------
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            # -------- control strip --------------------------
            html.Div(
                [
                    html.Button(
                        "⏵ Play",
                        id="play-btn",
                        n_clicks=0,
                        style={"marginRight": "6px"},
                    ),
                    html.Button(
                        "⏸ Pause",
                        id="pause-btn",
                        n_clicks=0,
                        style={"marginRight": "18px"},
                    ),
                    dcc.Dropdown(
                        id="step-size",
                        options=[
                            {"label": f"{k} slot", "value": k} for k in (1, 10, 50)
                        ],
                        value=1,
                        clearable=False,
                        style={
                            "width": 110,
                            "display": "inline-block",
                            "marginRight": "18px",
                        },
                    ),
                    # wrap the slider in a div; give *that* div flex-1
                    html.Div(
                        dcc.Slider(
                            id="slot-slider",
                            min=0,
                            max=max(n_slots - 1, 0),
                            value=0,
                            step=1,
                            marks={
                                i: str(i)
                                for i in range(0, n_slots, max(1, n_slots // 10))
                            },
                            tooltip={"placement": "bottom"},
                        ),
                        style={"flex": 1},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": "8px 14px",
                },
            ),
            # --- full-width info bar -------------------------
            html.Div(
                id="slot-info-display",
                className="card",
                style={
                    # "width": "100%",
                    "padding": "12px 24px",
                    "margin": "8px 16px",
                    # "backgroundColor": "#f5f5f5",
                    "borderRadius": "8px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",
                    "display": "flex",
                    "justifyContent": "space-around",
                    "alignItems": "center",
                    "fontFamily": "Arial, sans-serif",
                },
            ),
            # -------- card grid ------------------------------
            # --- first row: full-width globe  -----------------------
            html.Div(
                dcc.Graph(
                    id="density-graph",
                ),
                style={
                    "padding": "0 16px",
                    "gridColumn": "1 / -1",  # span all columns
                    "height": "600px",  # make it taller if you like
                },
                className="card",
            ),
            # --- second row: plots ------------
            html.Div(
                [
                    html.Div(dcc.Graph(id="clusters-line"), className="card"),
                    html.Div(dcc.Graph(id="totaldist-line"), className="card"),
                    html.Div(dcc.Graph(id="avg-nnd-line"), className="card"),
                    html.Div(dcc.Graph(id="nni-line"), className="card"),
                    html.Div(dcc.Graph(id="mev-line"), className="card"),
                    html.Div(dcc.Graph(id="attest-line"), className="card"),
                    html.Div(dcc.Graph(id="proposal-time-line"), className="card"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                    "gap": "12px",
                    "padding": "20px 16px",
                },
            ),
            # -------- hidden helpers -------------------------
            dcc.Interval(id="play-interval", interval=500, disabled=True),
            dcc.Store(id="movie-state", data={"slot": 0, "playing": False}),
        ]
    )

    # ---------------------------------------------------------
    #  4.  Animation controller (Play / Pause / slider / step)
    # ---------------------------------------------------------
    @app.callback(
        Output("movie-state", "data"),
        Output("slot-slider", "value"),
        Output("play-interval", "disabled"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        Input("slot-slider", "value"),
        Input("play-interval", "n_intervals"),
        Input("step-size", "value"),
        State("movie-state", "data"),
        prevent_initial_call=True,
    )
    def movie_controller(
        play_clicks, pause_clicks, slider_val, n_ticks, step_sz, state
    ):

        trig = dash.callback_context.triggered_id

        if trig == "play-btn":
            state["playing"] = True
        elif trig == "pause-btn":
            state["playing"] = False
        elif trig == "slot-slider":
            state["slot"] = slider_val
            state["playing"] = False  # dragging pauses
        elif trig == "play-interval" and state["playing"]:
            state["slot"] = (state["slot"] + step_sz) % n_slots

        return state, state["slot"], not state["playing"]

    # ---------------------------------------------------------
    #  5.  Redraw everything when slot changes
    # ---------------------------------------------------------
    @app.callback(
        Output("density-graph", "figure"),
        Output("clusters-line", "figure"),
        Output("totaldist-line", "figure"),
        Output("avg-nnd-line", "figure"),
        Output("nni-line", "figure"),
        Output("mev-line", "figure"),
        Output("attest-line", "figure"),
        Output("proposal-time-line", "figure"),
        Output("slot-info-display", "children"),
        Input("movie-state", "data"),
    )
    def redraw(state):
        idx = state["slot"]
        x = list(range(idx + 1))

        # -- 3-D view (card 1) --
        fig3d = build_density_fig(all_slot_data[idx], relay_data[idx])
        fig3d.update_layout(title=f"Geo Tracker", margin=dict(l=15, r=10, b=20, t=40))

        # spatial metrics
        def mkline(data, title):
            f = go.Figure(go.Scatter(x=x, y=data[: idx + 1], mode="lines"))
            f.update_layout(title=title, margin=dict(l=10, r=10, t=30, b=20))
            return f

        fig_c = mkline(clusters_hist, "Clusters")
        fig_t = mkline(total_dist_hist, "Total Distance")
        fig_a = mkline(avg_nnd_hist, "Avg NND")
        fig_n = mkline(nni_hist, "NNI")
        fig_mev = mkline(mev_hist, "MEV Earned")
        fig_attest = mkline(attest_hist, "Attestation Rate %")
        fig_proposal_time = mkline(proposal_time_hist, "Proposal Time (s)")

        info = html.Div(
            [
                html.Span(
                    f"Slot {idx+1}", style={"fontWeight": "bold", "marginRight": "24px"}
                ),
                html.Span(
                    f"Clusters: {clusters_hist[idx]}", style={"marginRight": "24px"}
                ),
                html.Span(
                    f"Total Distance: {total_dist_hist[idx]:.4f}",
                    style={"marginRight": "24px"},
                ),
                html.Span(
                    f"Avg NND: {avg_nnd_hist[idx]:.4f}", style={"marginRight": "24px"}
                ),
                html.Span(f"NNI: {nni_hist[idx]:.4f}", style={"marginRight": "24px"}),
                html.Span(
                    f"MEV Earned: {mev_hist[idx]:.4f}", style={"marginRight": "24px"}
                ),
                html.Span(
                    f"Attestation Rate: {attest_hist[idx]:.2f}%",
                    style={"marginRight": "24px"},
                ),
                html.Span(
                    f"Proposal Time: {proposal_time_hist[idx]:.2f} ms",
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "justifyContent": "space-around",
                "alignItems": "center",
                "fontFamily": "Arial, sans-serif",
            },
        )

        return (
            fig3d,
            fig_c,
            fig_t,
            fig_a,
            fig_n,
            fig_mev,
            fig_attest,
            fig_proposal_time,
            info,
        )

    return app


# --- Main Execution ---
if __name__ == "__main__":
    dir = "output"
    parser = argparse.ArgumentParser(description="Run the simulation viewer.")
    parser.add_argument(
        "--data",
        type=str,
        default=f"{dir}/data.json",
        help="Path to the data file (default: data.json)",
    )
    args = parser.parse_args()

    data_path = args.data
    all_slot_data = load_data(data_path)
    relay_data = load_data(f"{dir}/relay_data.json")
    mev_series = load_data(f"{dir}/mev_by_slot.json")
    attest_series = load_data(f"{dir}/attest_by_slot.json")
    proposal_time_series = load_data(f"{dir}/proposal_time_by_slot.json")

    if not all_slot_data:
        print("Application cannot start because data is missing.")
        exit(1)
    else:
        app = create_app(
            all_slot_data, relay_data, mev_series, attest_series, proposal_time_series
        )
        app.run(debug=True)
