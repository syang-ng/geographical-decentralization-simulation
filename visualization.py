# playback_app.py
import json
import math
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# --- Helper Function for Distance ---
# We need this for the density calculation.
def geodesic_distance(p1, p2):
    """Calculates the geodesic distance between two points on a unit sphere."""
    dotp = p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]
    dotp = max(-1.0, min(1.0, dotp)) # Clamp for numerical safety
    return math.acos(dotp)

# --- Pre-load Data ---
try:
    with open('data.json', 'r') as f:
        all_slot_data = json.load(f)
    NUM_SLOTS = len(all_slot_data)
except FileNotFoundError:
    print("Error: data.json not found. Please run simulation first.")
    all_slot_data = []
    NUM_SLOTS = 0

# --- Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Simulation Viewer"),
    html.P("Drag the slider to select a slot. Validators are colored by local density."),
    
    dcc.Graph(id='density-playback-graph'),
    
    html.Div(id='slot-display', style={'textAlign': 'center', 'fontSize': 20, 'fontWeight': 'bold'}),
    
    dcc.Slider(
        id='slot-slider',
        min=0,
        max=NUM_SLOTS - 1 if NUM_SLOTS > 0 else 0,
        value=0, # Start at the first slot
        marks={i: str(i) for i in range(0, NUM_SLOTS, 10)}, # Marks every 10 slots
        step=1,
    ),
])

@app.callback(
    Output('density-playback-graph', 'figure'),
    Output('slot-display', 'children'),
    Input('slot-slider', 'value')
)
def update_density_view(selected_slot):
    """
    This callback updates the view for the selected slot, including density calculation.
    """
    # 1. Get the 1000 points for the currently selected slot
    points = all_slot_data[selected_slot]
    
    # 2. Calculate density for each point within this slot
    # Note: This calculation is intensive and may cause a slight delay when moving the slider.
    K_NEIGHBORS = 10
    density_scores = []
    for i, p1 in enumerate(points):
        distances = [geodesic_distance(p1, p2) for j, p2 in enumerate(points) if i != j]
        distances.sort()
        avg_dist_to_neighbors = sum(distances[:K_NEIGHBORS]) / K_NEIGHBORS
        density = 1.0 / (avg_dist_to_neighbors + 1e-9)
        density_scores.append(density)

    # 3. Unzip points for plotting
    x_coords, y_coords, z_coords = zip(*points)

    # 4. Create the figure with density coloring
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=4,
            color=density_scores,   # Use the calculated scores for color
            colorscale='Viridis',   # A nice color scale for density
            showscale=True,
            colorbar=dict(title='Local Density')
        )
    )])
    
    # 5. Update layout
    fig.update_layout(
        title=f"Slot {selected_slot}: 1000 Validators with Density Coloring",
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    display_text = f"Currently showing Slot: {selected_slot}"
    
    return fig, display_text

# --- Main Execution ---
if __name__ == '__main__':
    if not all_slot_data:
        print("Application cannot start because data is missing.")
    else:
        app.run(debug=True)
