import argparse
import dash
import json
import math
import plotly.graph_objects as go

from dash import State, dcc, html
from dash.dependencies import Input, Output
from sklearn.neighbors import NearestNeighbors

# Assuming these are available in your environment
from distribution import *
from measure import *

from dash import callback_context

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


def create_app(all_slot_data, relay_data, mev_series, attest_series):
    n_slots = len(all_slot_data)

    # ---------------------------------------------------------
    #  1.  Pre-compute per-slot metric history (once)
    # ---------------------------------------------------------
    clusters_hist, total_dist_hist, avg_nnd_hist, nni_hist, mev_hist, attest_hist = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    granularity = 10
    # placeholders for “last seen” metrics
    last_c = last_t = last_a = last_nni = last_mev = last_attest = 0.0

    for i, pts in enumerate(all_slot_data):
        if i % granularity == 0:
            # do the full n×n compute only on multiples of `granularity`
            dm = init_distance_matrix(pts, space)
            if len(pts) > 1:
                last_c = cluster_matrix(dm)
                last_t = total_distance(dm)
                last_a = average_nearest_neighbor_distance(dm)
                last_nni = nearest_neighbor_index_spherical(dm, space)[0]
                last_mev = sum(mev_series[i]) if mev_series else 0.0
                last_attest = sum(attest_series[i])
            else:
                last_c = 0
                last_t = last_a = last_nni = last_mev = last_attest = 0.0

        # *every* slot, append whatever the “last computed” values are
        clusters_hist.append(last_c)
        total_dist_hist.append(last_t)
        avg_nnd_hist.append(last_a)
        nni_hist.append(last_nni)
        mev_hist.append(last_mev)
        attest_hist.append(last_attest)

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
            # -------- card grid ------------------------------
            html.Div(
                [
                    html.Div(dcc.Graph(id="density-graph"), className="card"),
                    html.Div(dcc.Graph(id="clusters-line"), className="card"),
                    html.Div(dcc.Graph(id="totaldist-line"), className="card"),
                    html.Div(dcc.Graph(id="avg-nnd-line"), className="card"),
                    html.Div(dcc.Graph(id="nni-line"), className="card"),
                    html.Div(dcc.Graph(id="mev-line"), className="card"),
                    html.Div(dcc.Graph(id="attest-line"), className="card"),
                    html.Div(id="slot-info-display", className="card"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                    "gap": "12px",
                    "padding": "0 16px 20px",
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

        # new MEV & supermaj
        fig_mev = mkline(mev_hist, "MEV Earned")
        fig_attest = mkline(attest_hist, "Attestation Rate %")

        info = html.Div(
            [
                html.H4(f"Slot {idx+1}", style={"marginTop": 0}),
                html.P(f"Clusters: {clusters_hist[idx]}"),
                html.P(f"Total dist: {total_dist_hist[idx]:.4f}"),
                html.P(f"Avg NND: {avg_nnd_hist[idx]:.4f}"),
                html.P(f"NNI: {nni_hist[idx]:.4f}"),
                html.P(f"MEV: {mev_hist[idx]:.4f}"),
                html.P(f"Attestation %: {attest_hist[idx]:.2f}"),
            ],
            style={"padding": "12px"},
        )

        return fig3d, fig_c, fig_t, fig_a, fig_n, fig_mev, fig_attest, info

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

    if not all_slot_data:
        print("Application cannot start because data is missing.")
        exit(1)
    else:
        app = create_app(all_slot_data, relay_data, mev_series, attest_series)
        app.run(debug=True)
