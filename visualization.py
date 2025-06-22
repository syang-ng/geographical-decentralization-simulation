import argparse
import dash
import json
import math
import plotly.graph_objects as go

from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.neighbors import NearestNeighbors

# Assuming these are available in your environment
from distribution import *
from measure import *

space = SphericalSpace()

# --- Helper Function for Distance ---
# We need this for the density calculation.
def geodesic_distance(p1, p2):
    """Calculates the geodesic distance between two points on a unit sphere."""
    dotp = p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]
    dotp = max(-1.0, min(1.0, dotp)) # Clamp for numerical safety
    return math.acos(dotp)

# --- Pre-load Data ---
def load_data(file_path):
    """Load the slot data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run the simulation first.")
        return []

# --- Dash App ---
def create_app(all_slot_data):
    """Create and return the Dash app instance."""
    app = dash.Dash(__name__)

    number_of_slots = len(all_slot_data)

    app.layout = html.Div([
        html.H1("Simulation Viewer"),
        html.P("Drag the slider to select a slot. Validators are colored by local density."),
        
        dcc.Graph(id='density-playback-graph'),
        
        # This div will now display the slot and all the calculated metrics
        html.Div(id='slot-info-display', style={'textAlign': 'center', 'fontSize': 16, 'fontWeight': 'bold', 'margin-top': '20px'}),
        
        dcc.Slider(
            id='slot-slider',
            min=0,
            max=number_of_slots - 1 if number_of_slots > 0 else 0,
            value=0, # Start at the first slot
            marks={i: str(i) for i in range(0, number_of_slots, 100)}, # Marks every 100 slots
            step=1,
        ),
    ])


    @app.callback(
        Output('density-playback-graph', 'figure'),
        Output('slot-info-display', 'children'), # Updated Output ID
        Input('slot-slider', 'value')
    )
    def update_density_view(selected_slot):
        """
        This callback updates the view for the selected slot, including density calculation and metric display.
        """
        # 1. Get the 1000 points for the currently selected slot
        points = all_slot_data[selected_slot]
        
        # 2. Calculate the required metrics
        # Ensure your 'measure.py' functions are robust to small numbers of points if that's possible.
        dist_matrix = init_distance_matrix(points, space)

        # Handle cases where dist_matrix might be empty or too small for certain calculations
        if len(points) > 1:
            number_of_clusters = cluster_matrix(dist_matrix)
            total_distance_value = total_distance(dist_matrix)
            avg_nnd = average_nearest_neighbor_distance(dist_matrix)
            nni, _, _ = nearest_neighbor_index_spherical(dist_matrix, space)
        else: # Default values if not enough points for meaningful calculation
            number_of_clusters = 0
            total_distance_value = 0.0
            avg_nnd = 0.0
            nni = 0.0


        # 3. Calculate density for each point within this slot
        radius = 0.2

        nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(points)
        neighbors_within_radius = nbrs.radius_neighbors(points, return_distance=False)

        density = np.array([len(neigh) - 1 for neigh in neighbors_within_radius])

        density = np.clip(density, 0, None)
        # density_normalized = (density - density.min()) / (density.max() - density.min() + 1e-8)

        # 4. Unzip points for plotting
        x_coords, y_coords, z_coords = zip(*points)

        # 5. Create the figure with density coloring
        fig = go.Figure(data=[go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=4,
                color=density,   # Use the calculated scores for color
                colorscale='Viridis',   # A nice color scale for density
                showscale=True,
                colorbar=dict(title='Local Density')
            )
        )])
        
        # 6. Update layout
        fig.update_layout(
            title=f"Slot {selected_slot}: {len(points)} Validators with Density Coloring",
            scene=dict(
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[-1, 1]),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # 7. Create the display text for all metrics
        display_text = html.Div([
            html.P(f"Currently showing Slot: {selected_slot}"),
            html.P(f"Number of Clusters: {number_of_clusters}"),
            html.P(f"Total Distance: {total_distance_value:.4f}"),
            html.P(f"Average Nearest Neighbor Distance: {avg_nnd:.4f}"),
            html.P(f"Nearest Neighbor Index (NNI): {nni:.4f}")
        ])
        
        return fig, display_text
    
    return app


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the simulation viewer.')
    parser.add_argument('--data', type=str, default='data.json', help='Path to the data file (default: data.json)')
    args = parser.parse_args()

    data_path = args.data
    all_slot_data = load_data(data_path)
    if not all_slot_data:
        print("Application cannot start because data is missing.")
        exit(1)
    else:
        app = create_app(all_slot_data)
        app.run(debug=True)
