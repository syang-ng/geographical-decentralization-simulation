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


def create_app(all_slot_data, relay_data):
    """
    Build and return a Dash app that animates the simulation slot-by-slot.

    Parameters
    ----------
    all_slot_data : list[list[tuple[float,float,float]]]
        Outer list indexed by slot; each inner list holds validator (x,y,z) points.
    relay_data : list[list[list[float,float,float]]]
        Same length as all_slot_data; each item is **a list of relay points** for that slot.
        If there is only one relay that never moves, each element should be [[x,y,z]].
    """
    n_slots = len(all_slot_data)

    app = dash.Dash(__name__)

    # ---------------  layout  ---------------------------------
    app.layout = html.Div(
        [
            html.H1("Simulation Viewer"),
            html.Div(
                [
                    html.Button(
                        "⏵ Play",
                        id="play-btn",
                        n_clicks=0,
                        style={"marginRight": "8px"},
                    ),
                    html.Button(
                        "⏸ Pause",
                        id="pause-btn",
                        n_clicks=0,
                        style={"marginRight": "16px"},
                    ),
                    # NEW: choose increment
                    dcc.Dropdown(
                        id="step-size",
                        options=[
                            {"label": "1 slot", "value": 1},
                            {"label": "10 slots", "value": 10},
                            {"label": "50 slots", "value": 50},
                        ],
                        value=1,  # default
                        clearable=False,
                        style={"width": "120px", "display": "inline-block"},
                    ),
                ],
                style={"textAlign": "center", "marginBottom": "10px"},
            ),
            dcc.Graph(id="density-playback-graph"),
            dcc.Slider(
                id="slot-slider",
                min=0,
                max=max(n_slots - 1, 0),
                value=0,
                step=1,
                marks={i: str(i) for i in range(0, n_slots, max(1, n_slots // 10))},
            ),
            html.Div(
                id="slot-info-display",
                style={
                    "textAlign": "center",
                    "fontWeight": "bold",
                    "marginTop": "12px",
                },
            ),
            # hidden components for animation state
            dcc.Interval(id="play-interval", interval=500, disabled=True),  # ms
            dcc.Store(id="movie-state", data={"slot": 0, "playing": False}),
        ]
    )

    # ---------------  callbacks  ------------------------------
    @app.callback(
        Output("movie-state", "data"),
        Output("slot-slider", "value"),
        Output("play-interval", "disabled"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        Input("slot-slider", "value"),
        Input("play-interval", "n_intervals"),
        Input("step-size", "value"),  #  ← NEW
        State("movie-state", "data"),
        prevent_initial_call=True,
    )
    def movie_controller(
        play_clicks, pause_clicks, slider_value, n_ticks, step_size, state
    ):

        trig = dash.callback_context.triggered_id

        if trig == "play-btn":
            state["playing"] = True

        elif trig == "pause-btn":
            state["playing"] = False

        elif trig == "slot-slider":
            state["slot"] = slider_value
            state["playing"] = False  # dragging pauses playback

        elif trig == "play-interval" and state["playing"]:
            state["slot"] = (state["slot"] + step_size) % n_slots

        # if the user just changed the dropdown while paused,
        # nothing else to do – we only use step_size on the next tick

        disable_interval = not state["playing"]
        return state, state["slot"], disable_interval

    # -- Main redraw of the 3-D view & metrics -----------------
    @app.callback(
        Output("density-playback-graph", "figure"),
        Output("slot-info-display", "children"),
        Input("movie-state", "data"),
    )
    def update_density_view(state):
        idx = state["slot"]
        points = all_slot_data[idx]

        # ---- metrics (distance matrix etc.) ------------------
        dist_matrix = init_distance_matrix(points, space)

        if len(points) > 1:
            n_clusters = cluster_matrix(dist_matrix)
            total_dist = total_distance(dist_matrix)
            avg_nnd = average_nearest_neighbor_distance(dist_matrix)
            nni, _, _ = nearest_neighbor_index_spherical(dist_matrix, space)
        else:
            n_clusters = 0
            total_dist = avg_nnd = nni = 0.0

        # ---- local density colouring -------------------------
        radius = 0.2
        nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree").fit(points)
        neigh = nbrs.radius_neighbors(points, return_distance=False)
        density = np.array([len(n) - 1 for n in neigh], dtype=float)
        density = np.clip(density, 0, None)

        # ---- validator scatter trace -------------------------
        x, y, z = zip(*points)
        fig = go.Figure(
            data=[
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
                        colorbar=dict(title="Local Density"),
                    ),
                    name="Validators",
                )
            ]
        )

        # ---- relay trace(s) ---------------------------------
        if relay_data and idx < len(relay_data):
            r_pts = relay_data[idx]
            # promote single point to list-of-points if needed
            if isinstance(r_pts[0], (int, float)):
                r_pts = [r_pts]
            rx, ry, rz = zip(*r_pts)
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

        # ---- layout tweaks -----------------------------------
        fig.update_layout(
            title=f"Slot {idx+1} · {len(points)} validators",
            scene=dict(
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[-1, 1]),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(title=None),
        )

        # ---- legend tweaks -----------------------------------
        fig.update_layout(
            legend=dict(
                x=0.92,
                y=1,
                bgcolor="rgba(255,255,255,0.6)",
            )
        )

        # ---- metric panel text -------------------------------
        info = html.Div(
            [
                html.P(f"Slot {idx+1}"),
                html.P(f"Clusters: {n_clusters}"),
                html.P(f"Total distance: {total_dist:,.4f}"),
                html.P(f"Avg NND: {avg_nnd:,.4f}"),
                html.P(f"NNI: {nni:,.4f}"),
            ]
        )

        return fig, info

    # ---------------------------------------------------------
    return app


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the simulation viewer.")
    parser.add_argument(
        "--data",
        type=str,
        default="data.json",
        help="Path to the data file (default: data.json)",
    )
    args = parser.parse_args()

    data_path = args.data
    all_slot_data = load_data(data_path)
    relay_data = load_data("relay_data.json")
    if not all_slot_data:
        print("Application cannot start because data is missing.")
        exit(1)
    else:
        app = create_app(all_slot_data, relay_data)
        app.run(debug=True)
