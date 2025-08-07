import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gmean
import time

# PARAMETERS
BUCKET_COUNT = 7
AGENT_COUNT = 50
ROUNDS = 200  # maximum rounds
VALUE_EXPONENT = 1  # exponent for division

# Pareto parameters
alpha = 1.5
x_min = 10.0


def compute_value(df, p):
    return df["original_value"] / (df["subscriber_count"] + 1) ** p


# INITIALIZATION
if "initialized" not in st.session_state:
    # Setup Pareto bucket values
    pareto_draws = (np.random.pareto(alpha, size=BUCKET_COUNT) + 1) * x_min
    buckets = pd.DataFrame(
        {
            "id": range(BUCKET_COUNT),
            "original_value": pareto_draws,
            "subscriber_count": np.zeros(BUCKET_COUNT, dtype=int),
        }
    )
    # Initial weighted assignment
    probs = buckets["original_value"] / buckets["original_value"].sum()
    assignments = []
    for agent in range(AGENT_COUNT):
        choice = np.random.choice(buckets["id"], p=probs)
        assignments.append(choice)
        buckets.at[choice, "subscriber_count"] += 1
    buckets["value"] = compute_value(buckets, VALUE_EXPONENT)

    # State variables
    st.session_state.buckets = buckets
    st.session_state.assignments = assignments
    st.session_state.round = 0
    st.session_state.phase = 0
    st.session_state.current_agent = None
    st.session_state.current_best = None
    st.session_state.geo_means = []
    st.session_state.playing = False
    st.session_state.initialized = True


# CALLBACKS
def reset_sim():
    keys = [
        "initialized",
        "buckets",
        "assignments",
        "round",
        "phase",
        "current_agent",
        "current_best",
        "geo_means",
        "playing",
    ]
    for key in keys:
        st.session_state.pop(key, None)


def toggle_play():
    st.session_state.playing = not st.session_state.playing


# UI
st.title("Geo-Decentralization Buckets")
st.markdown(
    """
- This simulation runs a series of rounds where agents select buckets based on their values, and the system reassigns agents to optimize the distribution of subscribers across buckets.
- Buckets are initialized with values drawn from a Pareto distribution, and agents are assigned to buckets based on these values. 
- The simulation runs for a maximum of 200 rounds, with each round consisting of three phases: agent selection, best bucket identification, and assignment.
            """
)
st.markdown(
    r"""
The value of each bucket is computed as:

$$
\mathrm{value} \;=\;
\frac{\mathrm{original\_value}}
     {\bigl(\mathrm{subscriber\_count} + 1\bigr)^{p}}
$$

Where:

- $\mathrm{original\_value}$ is the initial value of the bucket drawn from a Pareto distribution.
- $\mathrm{subscriber\_count}$ is the number of agents assigned to the bucket.  
- $p$ is the value exponent (currently set to 0.5).  
""",
    unsafe_allow_html=True,
)
st.markdown("---")
st.subheader("Simulation")

# Sidebar Reset button
st.sidebar.button("Reset Simulation", on_click=reset_sim)


# Fetch state
buckets = st.session_state.buckets
assignments = st.session_state.assignments

# Header
st.markdown(
    f"**Round {st.session_state.round} / {ROUNDS} | Phase {st.session_state.phase}**"
)

# Display buckets with highlight
cols = st.columns(BUCKET_COUNT)
for _, row in buckets.iterrows():
    style = "background-color:#F5F5F5; color:#333;"
    if (
        st.session_state.phase == 1
        and row["id"] == assignments[st.session_state.current_agent]
    ):
        style = "background-color:#FFF59D; color:#333;"
    elif st.session_state.phase == 2 and row["id"] == st.session_state.current_best:
        style = "background-color:#A5D6A7; color:#333;"
    elif (
        st.session_state.phase == 3
        and row["id"] == assignments[st.session_state.current_agent]
    ):
        style = "background-color:#90CAF9; color:#333;"
    html = (
        f"<div style='{style} padding:2px; border:1px solid #AAA; border-radius:4px; text-align:center; color:#333;'> "
        f"<strong>Bucket {int(row['id'])}</strong><br>Subs: {int(row['subscriber_count'])}<br>"
        f"Val: {row['value']:.1f}</div>"
    )
    cols[int(row["id"])].markdown(html, unsafe_allow_html=True)

st.markdown("")
# Show phase info
if st.session_state.phase == 1:
    st.info(f"Agent {st.session_state.current_agent} selected.")
elif st.session_state.phase == 2:
    st.info(f"Best bucket: {st.session_state.current_best}")
elif st.session_state.phase == 3:
    st.success(f"Assigned.")
else:
    st.info("Waiting for next step...")


# Step function
def step():
    if st.session_state.phase == 0:
        if st.session_state.round < ROUNDS:
            st.session_state.round += 1
            st.session_state.current_agent = np.random.choice(AGENT_COUNT)
            st.session_state.phase = 1
    elif st.session_state.phase == 1:
        buckets = st.session_state.buckets
        buckets["value"] = compute_value(buckets, VALUE_EXPONENT)
        st.session_state.current_best = int(buckets["value"].idxmax())
        st.session_state.phase = 2
    elif st.session_state.phase == 2:
        buckets = st.session_state.buckets
        assignments = st.session_state.assignments
        agent = st.session_state.current_agent
        old = assignments[agent]
        best = st.session_state.current_best
        buckets.at[old, "subscriber_count"] -= 1
        buckets.at[best, "subscriber_count"] += 1
        assignments[agent] = best
        buckets["value"] = compute_value(buckets, VALUE_EXPONENT)
        gm = gmean(buckets["subscriber_count"] + 1e-6)
        st.session_state.geo_means.append(gm)
        st.session_state.phase = 3
    else:
        st.session_state.phase = 0


col1, col2 = st.columns(2)
with col1:
    play_label = "Play ▶" if not st.session_state.playing else "Pause ⏸"
    st.button(play_label, on_click=toggle_play)
with col2:
    st.button("Step ▶", on_click=step)


# Geo mean chart (shown during play and manual)
st.markdown("---")
st.subheader("Geometric Mean of Subscriber Counts")
st.markdown(
    r"""
The geometric mean of subscriber counts is computed after each assignment step. This metric helps visualize the distribution of subscribers across buckets, indicating how balanced the assignments are over time.
It is defined as:
$$
\mathrm{GeoMean} = \sqrt[n]{x_1 \cdot x_2 \cdot \ldots \cdot x_n}
$$
where $x_i$ are the subscriber counts in each bucket.
""",
    unsafe_allow_html=True,
)
if st.session_state.geo_means:

    st.line_chart(
        pd.DataFrame(st.session_state.geo_means, columns=["Geo Mean"]),
        use_container_width=True,
    )

# Auto-play loop (after rendering everything)
if st.session_state.playing:
    time.sleep(1.0)  # slower playback: 1 second per step
    step()
    st.rerun()
