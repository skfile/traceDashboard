#!/usr/bin/env python3
"""
Enhanced Neural Trajectory Visualization
=======================================

This script creates two interactive Plotly visualizations:

1. PCA-based trajectory animation with crossing detection
2. Neuron-based trajectory animation (no PCA) with user-selectable neurons

Features:
- Interactive epsilon (spatial) and delta (temporal) threshold controls
- Stimulus and direction labeling on hover
- Play/pause/frame controls
- Crossing event visualization
- Neuron selection for non-PCA plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_neural_data():
    """Load retina and V1 neural data with stimulus information."""
    # Use relative path from current directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Load neural traces
    retina_traces = np.load(os.path.join(data_dir, 'retina_tensor_traces.npy'))
    v1_traces = np.load(os.path.join(data_dir, 'V1_tensor_traces.npy'))
    
    # Load cell info
    with open(os.path.join(data_dir, 'retina_cell_info.pkl'), 'rb') as f:
        retina_info = pickle.load(f)
    with open(os.path.join(data_dir, 'V1_cell_info.pkl'), 'rb') as f:
        v1_info = pickle.load(f)
    
    # CORRECTED stimulus names from STIMULUS_MAPPING_DOCUMENTATION.md
    stimulus_names = [
        "Low SF Grating",    # Index 0: gratW12.5
        "High SF Grating",   # Index 1: gratW2
        "Neg 1-dot Flow",    # Index 2: -1dotD2s2bg
        "Neg 3-dot Flow",    # Index 3: -3dotD2s2bg
        "Pos 1-dot Flow",    # Index 4: +1dotD2s2bg
        "Pos 3-dot Flow"     # Index 5: +3dotD2s2bg
    ]
    
    # Direction information (8 directions)
    directions_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    n_directions = len(directions_deg)
    
    print(f"Retina traces shape: {retina_traces.shape}")
    print(f"V1 traces shape: {v1_traces.shape}")
    print(f"Number of stimuli: {len(stimulus_names)}")
    print(f"Number of directions: {n_directions}")
    
    return retina_traces, v1_traces, retina_info, v1_info, stimulus_names, directions_deg

# ============================================================================
# 2. PCA TRAJECTORY EXTRACTION WITH ENHANCED CROSSING DETECTION
# ============================================================================

def extract_pca_trajectories(traces, n_pcs=10):
    """
    Extract PCA-based trajectories with stimulus/direction labeling.
    
    Returns:
    - trajectories: list of (time, n_pcs) arrays
    - pca_model: fitted PCA model
    - labels: list of (stimulus_name, direction_deg) tuples
    """
    n_neurons, n_stim, n_dir, n_time = traces.shape
    
    # Build trials matrix: neurons × (stimulus×direction)
    trials = []
    for s in range(n_stim):
        for d in range(n_dir):
            # Average over time axis
            vec = traces[:, s, d, :].mean(axis=1)
            trials.append(vec)
    trials = np.stack(trials, axis=0)  # shape (n_stim*n_dir, n_neurons)
    
    # Fit PCA
    pca = PCA(n_components=n_pcs)
    pca.fit(trials)
    
    # Extract trajectories in PCA space
    trajectories = []
    labels = []
    
    for s in range(n_stim):
        for d in range(n_dir):
            traj = np.zeros((n_time, n_pcs))
            for t in range(n_time):
                vec = traces[:, s, d, t]  # shape (n_neurons,)
                traj[t, :] = pca.transform(vec.reshape(1, -1))
            trajectories.append(traj)
            labels.append((s, d))  # (stimulus_idx, direction_idx)
    
    return trajectories, pca, labels

def detect_crossings_with_temporal_constraint(trajectories, epsilon=1.0, delta=5):
    """
    Detect crossing events with both spatial (epsilon) and temporal (delta) constraints.
    
    Args:
        trajectories: list of trajectory arrays
        epsilon: spatial distance threshold
        delta: temporal distance threshold (frames)
    
    Returns:
        events: list of crossing events with metadata
    """
    events = []
    N = len(trajectories)
    
    for i in range(N):
        for j in range(i+1, N):
            traj_i = trajectories[i]
            traj_j = trajectories[j]
            
            # Compare all time points
            for ti in range(traj_i.shape[0]):
                for tj in range(traj_j.shape[0]):
                    # Spatial distance check
                    spatial_dist = np.linalg.norm(traj_i[ti, :3] - traj_j[tj, :3])
                    
                    # Temporal distance check
                    temporal_dist = abs(ti - tj)
                    
                    # Both constraints must be satisfied
                    if spatial_dist < epsilon and temporal_dist <= delta:
                        midpoint = (traj_i[ti] + traj_j[tj]) / 2
                        events.append({
                            'coord': midpoint,
                            'traj_ids': (i, j),
                            'times': (ti, tj),
                            'spatial_dist': spatial_dist,
                            'temporal_dist': temporal_dist
                        })
    
    return events

# ============================================================================
# 3. NEURON-BASED TRAJECTORY EXTRACTION (NO PCA)
# ============================================================================

def extract_neuron_trajectories(traces, neuron_indices):
    """
    Extract trajectories for specific neurons without PCA.
    
    Args:
        traces: neural data tensor (neurons, stimuli, directions, time)
        neuron_indices: list of 3 neuron indices to plot
    
    Returns:
        trajectories: list of (time, 3) arrays for selected neurons
        labels: list of (stimulus_name, direction_deg) tuples
    """
    n_neurons, n_stim, n_dir, n_time = traces.shape
    
    trajectories = []
    labels = []
    
    for s in range(n_stim):
        for d in range(n_dir):
            # Extract activity for selected neurons
            traj = traces[neuron_indices, s, d, :].T  # shape (time, 3)
            trajectories.append(traj)
            labels.append((s, d))
    
    return trajectories, labels

# ============================================================================
# 4. INTERACTIVE PLOTLY VISUALIZATIONS
# ============================================================================

def create_pca_trajectory_animation(trajectories, labels, stimulus_names, directions_deg, 
                                   epsilon=1.0, delta=5, title="PCA Trajectories"):
    """
    Create interactive PCA trajectory animation with crossing detection.
    """
    n_frames = trajectories[0].shape[0]
    
    # Detect crossings
    events = detect_crossings_with_temporal_constraint(trajectories, epsilon, delta)
    
    # Create event lookup by frame
    event_frames = {}
    for event in events:
        for traj_id, time_idx in zip(event['traj_ids'], event['times']):
            if time_idx not in event_frames:
                event_frames[time_idx] = []
            event_frames[time_idx].append({
                'coord': event['coord'],
                'traj_id': traj_id
            })
    
    def get_crossings_up_to_frame(frame_idx):
        """Get all unique crossing coordinates up to frame_idx"""
        crossing_coords = []
        for f in range(frame_idx + 1):
            if f in event_frames:
                for event_info in event_frames[f]:
                    coord = event_info['coord'][:3]
                    crossing_coords.append(coord)
        
        # Remove duplicates while preserving order
        unique_crossings = []
        seen_coords = set()
        for coord in crossing_coords:
            coord_key = tuple(coord)
            if coord_key not in seen_coords:
                seen_coords.add(coord_key)
                unique_crossings.append(coord)
        
        return unique_crossings
    
    # Initial frame data
    init_data = []
    
    # Add trajectory lines (just starting points)
    for i, traj in enumerate(trajectories):
        stim_idx, dir_idx = labels[i]
        stim_name = stimulus_names[stim_idx]
        direction = directions_deg[dir_idx]
        
        init_data.append(
            go.Scatter3d(
                x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                mode='lines',
                line=dict(width=3, color=f'rgba({50+i*30%200},{100+i*20%200},{150+i*40%200},0.8)'),
                name=f'{stim_name} {direction}°',
                hovertemplate=f'<b>{stim_name}</b><br>Direction: {direction}°<br>Trajectory: {i}<br>Stimulus: {stim_name}<br>Direction: {direction}°<extra></extra>',
                showlegend=True
            )
        )
    
    # Add start/end points
    for i, traj in enumerate(trajectories):
        stim_idx, dir_idx = labels[i]
        stim_name = stimulus_names[stim_idx]
        direction = directions_deg[dir_idx]
        
        init_data.append(
            go.Scatter3d(
                x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                name=f'Start {i}',
                hovertemplate=f'<b>Start Point</b><br>Stimulus: {stim_name}<br>Direction: {direction}°<br>Trajectory: {i}<extra></extra>',
                showlegend=False
            )
        )
        init_data.append(
            go.Scatter3d(
                x=[traj[-1,0]], y=[traj[-1,1]], z=[traj[-1,2]],
                mode='markers',
                marker=dict(size=8, color='black', symbol='circle'),
                name=f'End {i}',
                hovertemplate=f'<b>End Point</b><br>Stimulus: {stim_name}<br>Direction: {direction}°<br>Trajectory: {i}<extra></extra>',
                showlegend=False
            )
        )
    
    # Add initial crossings
    initial_crossings = get_crossings_up_to_frame(0)
    if initial_crossings:
        crossing_x = [coord[0] for coord in initial_crossings]
        crossing_y = [coord[1] for coord in initial_crossings]
        crossing_z = [coord[2] for coord in initial_crossings]
        init_data.append(
            go.Scatter3d(
                x=crossing_x, y=crossing_y, z=crossing_z,
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name='Crossings',
                hovertemplate='<b>Crossing Event</b><br>ε: {:.2f}, δ: {:.0f}<extra></extra>'.format(epsilon, delta),
                showlegend=True
            )
        )
    else:
        init_data.append(
            go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x'),
                name='Crossings',
                showlegend=True
            )
        )
    
    # Build animation frames
    frames = []
    for t in range(1, n_frames):
        frame_data = []
        
        # Add trajectory lines up to current frame
        for i, traj in enumerate(trajectories):
            stim_idx, dir_idx = labels[i]
            stim_name = stimulus_names[stim_idx]
            direction = directions_deg[dir_idx]
            
            frame_data.append(
                go.Scatter3d(
                    x=traj[:t+1,0], y=traj[:t+1,1], z=traj[:t+1,2],
                    mode='lines',
                    line=dict(width=3, color=f'rgba({50+i*30%200},{100+i*20%200},{150+i*40%200},0.8)'),
                    name=f'{stim_name} {direction}°',
                    hovertemplate=f'<b>{stim_name}</b><br>Direction: {direction}°<br>Frame: {t}<br>Stimulus: {stim_name}<br>Direction: {direction}°<extra></extra>',
                    showlegend=False
                )
            )
        
        # Add start/end points
        for i, traj in enumerate(trajectories):
            stim_idx, dir_idx = labels[i]
            stim_name = stimulus_names[stim_idx]
            direction = directions_deg[dir_idx]
            
            frame_data.append(
                go.Scatter3d(
                    x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                    mode='markers',
                    marker=dict(size=8, color='green', symbol='circle'),
                    hovertemplate=f'<b>Start Point</b><br>Stimulus: {stim_name}<br>Direction: {direction}°<br>Trajectory: {i}<extra></extra>',
                    showlegend=False
                )
            )
            frame_data.append(
                go.Scatter3d(
                    x=[traj[-1,0]], y=[traj[-1,1]], z=[traj[-1,2]],
                    mode='markers',
                    marker=dict(size=8, color='black', symbol='circle'),
                    hovertemplate=f'<b>End Point</b><br>Stimulus: {stim_name}<br>Direction: {direction}°<br>Trajectory: {i}<extra></extra>',
                    showlegend=False
                )
            )
        
        # Add crossings up to current frame
        crossings_to_show = get_crossings_up_to_frame(t)
        if crossings_to_show:
            crossing_x = [coord[0] for coord in crossings_to_show]
            crossing_y = [coord[1] for coord in crossings_to_show]
            crossing_z = [coord[2] for coord in crossings_to_show]
            frame_data.append(
                go.Scatter3d(
                    x=crossing_x, y=crossing_y, z=crossing_z,
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='x'),
                    name='Crossings',
                    hovertemplate='<b>Crossing Event</b><br>ε: {:.2f}, δ: {:.0f}<extra></extra>'.format(epsilon, delta),
                    showlegend=False
                )
            )
        else:
            frame_data.append(
                go.Scatter3d(
                    x=[], y=[], z=[],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='x'),
                    name='Crossings',
                    showlegend=False
                )
            )
        
        frames.append(go.Frame(data=frame_data, name=str(t)))
    
    # Create slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frame: "},
        pad={"t": 50},
        steps=[dict(
            label=str(i),
            method="animate",
            args=[[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
        ) for i in range(1, n_frames)]
    )]
    
    # Create figure
    fig = go.Figure(
        data=init_data,
        frames=frames,
        layout=go.Layout(
            title=f"{title} (ε={epsilon}, δ={delta})",
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            updatemenus=[{
                'type': 'buttons',
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'x': 0.1, 'xanchor': 'right',
                'y': 0, 'yanchor': 'top',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Reset',
                        'method': 'animate',
                        'args': [[str(1)], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=sliders,
            annotations=[
                dict(
                    text=f"Green: Start | Black: End | Red X: Crossings (ε={epsilon}, δ={delta})",
                    showarrow=False, xref="paper", yref="paper",
                    x=0.5, y=-0.1, xanchor='center', yanchor='bottom'
                )
            ]
        )
    )
    
    return fig

def create_neuron_trajectory_animation(trajectories, labels, stimulus_names, directions_deg,
                                      neuron_indices, title="Neuron Trajectories"):
    """
    Create interactive neuron trajectory animation (no PCA, no crossings).
    """
    n_frames = trajectories[0].shape[0]
    
    # Initial frame data
    init_data = []
    
    # Add trajectory lines (just starting points)
    for i, traj in enumerate(trajectories):
        stim_idx, dir_idx = labels[i]
        stim_name = stimulus_names[stim_idx]
        direction = directions_deg[dir_idx]
        
        init_data.append(
            go.Scatter3d(
                x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                mode='lines',
                line=dict(width=3, color=f'rgba({50+i*30%200},{100+i*20%200},{150+i*40%200},0.8)'),
                name=f'{stim_name} {direction}°',
                hovertemplate=f'<b>{stim_name}</b><br>Direction: {direction}°<br>Neurons: {neuron_indices}<br>Stimulus: {stim_name}<br>Direction: {direction}°<extra></extra>',
                showlegend=True
            )
        )
    
    # Add start/end points
    for i, traj in enumerate(trajectories):
        stim_idx, dir_idx = labels[i]
        stim_name = stimulus_names[stim_idx]
        direction = directions_deg[dir_idx]
        
        init_data.append(
            go.Scatter3d(
                x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                name=f'Start {i}',
                hovertemplate=f'<b>Start Point</b><br>Stimulus: {stim_name}<br>Direction: {direction}°<br>Neurons: {neuron_indices}<extra></extra>',
                showlegend=False
            )
        )
        init_data.append(
            go.Scatter3d(
                x=[traj[-1,0]], y=[traj[-1,1]], z=[traj[-1,2]],
                mode='markers',
                marker=dict(size=8, color='black', symbol='circle'),
                name=f'End {i}',
                hovertemplate=f'<b>End Point</b><br>Stimulus: {stim_name}<br>Direction: {direction}°<br>Neurons: {neuron_indices}<extra></extra>',
                showlegend=False
            )
        )
    
    # Build animation frames
    frames = []
    for t in range(1, n_frames):
        frame_data = []
        
        # Add trajectory lines up to current frame
        for i, traj in enumerate(trajectories):
            stim_idx, dir_idx = labels[i]
            stim_name = stimulus_names[stim_idx]
            direction = directions_deg[dir_idx]
            
            frame_data.append(
                go.Scatter3d(
                    x=traj[:t+1,0], y=traj[:t+1,1], z=traj[:t+1,2],
                    mode='lines',
                    line=dict(width=3, color=f'rgba({50+i*30%200},{100+i*20%200},{150+i*40%200},0.8)'),
                    name=f'{stim_name} {direction}°',
                    hovertemplate=f'<b>{stim_name}</b><br>Direction: {direction}°<br>Frame: {t}<br>Neurons: {neuron_indices}<br>Stimulus: {stim_name}<br>Direction: {direction}°<extra></extra>',
                    showlegend=False
                )
            )
        
        # Add start/end points
        for i, traj in enumerate(trajectories):
            stim_idx, dir_idx = labels[i]
            stim_name = stimulus_names[stim_idx]
            direction = directions_deg[dir_idx]
            
            frame_data.append(
                go.Scatter3d(
                    x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                    mode='markers',
                    marker=dict(size=8, color='green', symbol='circle'),
                    hovertemplate=f'<b>Start Point</b><br>Stimulus: {stim_name}<br>Direction: {direction}°<br>Neurons: {neuron_indices}<extra></extra>',
                    showlegend=False
                )
            )
            frame_data.append(
                go.Scatter3d(
                    x=[traj[-1,0]], y=[traj[-1,1]], z=[traj[-1,2]],
                    mode='markers',
                    marker=dict(size=8, color='black', symbol='circle'),
                    hovertemplate=f'<b>End Point</b><br>Stimulus: {stim_name}<br>Direction: {direction}°<br>Neurons: {neuron_indices}<extra></extra>',
                    showlegend=False
                )
            )
        
        frames.append(go.Frame(data=frame_data, name=str(t)))
    
    # Create slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frame: "},
        pad={"t": 50},
        steps=[dict(
            label=str(i),
            method="animate",
            args=[[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
        ) for i in range(1, n_frames)]
    )]
    
    # Create figure
    fig = go.Figure(
        data=init_data,
        frames=frames,
        layout=go.Layout(
            title=f"{title} (Neurons: {neuron_indices})",
            scene=dict(
                xaxis_title=f'Neuron {neuron_indices[0]}',
                yaxis_title=f'Neuron {neuron_indices[1]}',
                zaxis_title=f'Neuron {neuron_indices[2]}',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            updatemenus=[{
                'type': 'buttons',
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'x': 0.1, 'xanchor': 'right',
                'y': 0, 'yanchor': 'top',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Reset',
                        'method': 'animate',
                        'args': [[str(1)], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=sliders,
            annotations=[
                dict(
                    text=f"Green: Start | Black: End | Neurons: {neuron_indices}",
                    showarrow=False, xref="paper", yref="paper",
                    x=0.5, y=-0.1, xanchor='center', yanchor='bottom'
                )
            ]
        )
    )
    
    return fig

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main function to create and save the visualizations."""
    print("Loading neural data...")
    retina_traces, v1_traces, retina_info, v1_info, stimulus_names, directions_deg = load_neural_data()
    
    print("\nExtracting PCA trajectories...")
    # Extract PCA trajectories for both regions
    retina_trajs, pca_retina, retina_labels = extract_pca_trajectories(retina_traces, n_pcs=10)
    v1_trajs, pca_v1, v1_labels = extract_pca_trajectories(v1_traces, n_pcs=10)
    
    print(f"Retina: {len(retina_trajs)} trajectories")
    print(f"V1: {len(v1_trajs)} trajectories")
    
    # Create PCA trajectory animations with crossing detection
    print("\nCreating PCA trajectory animations...")
    
    # Retina PCA with crossings
    fig_retina_pca = create_pca_trajectory_animation(
        retina_trajs, retina_labels, stimulus_names, directions_deg,
        epsilon=1.0, delta=5, title="Retina PCA Trajectories"
    )
    fig_retina_pca.write_html("retina_pca_trajectories_interactive.html")
    print("Saved: retina_pca_trajectories_interactive.html")
    
    # V1 PCA with crossings
    fig_v1_pca = create_pca_trajectory_animation(
        v1_trajs, v1_labels, stimulus_names, directions_deg,
        epsilon=1.0, delta=5, title="V1 PCA Trajectories"
    )
    fig_v1_pca.write_html("v1_pca_trajectories_interactive.html")
    print("Saved: v1_pca_trajectories_interactive.html")
    
    # Create neuron-based trajectory animations (no PCA)
    print("\nCreating neuron trajectory animations...")
    
    # Choose 3 neurons for visualization (you can modify these indices)
    neuron_indices = [0, 1, 2]  # First 3 neurons
    
    # Retina neuron trajectories
    retina_neuron_trajs, retina_neuron_labels = extract_neuron_trajectories(
        retina_traces, neuron_indices
    )
    fig_retina_neurons = create_neuron_trajectory_animation(
        retina_neuron_trajs, retina_neuron_labels, stimulus_names, directions_deg,
        neuron_indices, title="Retina Neuron Trajectories"
    )
    fig_retina_neurons.write_html("retina_neuron_trajectories_interactive.html")
    print("Saved: retina_neuron_trajectories_interactive.html")
    
    # V1 neuron trajectories
    v1_neuron_trajs, v1_neuron_labels = extract_neuron_trajectories(
        v1_traces, neuron_indices
    )
    fig_v1_neurons = create_neuron_trajectory_animation(
        v1_neuron_trajs, v1_neuron_labels, stimulus_names, directions_deg,
        neuron_indices, title="V1 Neuron Trajectories"
    )
    fig_v1_neurons.write_html("v1_neuron_trajectories_interactive.html")
    print("Saved: v1_neuron_trajectories_interactive.html")
    
    print("\nAll visualizations created successfully!")
    print("\nFiles created:")
    print("- retina_pca_trajectories_interactive.html (PCA with crossings)")
    print("- v1_pca_trajectories_interactive.html (PCA with crossings)")
    print("- retina_neuron_trajectories_interactive.html (Neuron-based, no PCA)")
    print("- v1_neuron_trajectories_interactive.html (Neuron-based, no PCA)")
    
    # Print crossing statistics
    print("\nCrossing Statistics:")
    retina_events = detect_crossings_with_temporal_constraint(retina_trajs, epsilon=1.0, delta=5)
    v1_events = detect_crossings_with_temporal_constraint(v1_trajs, epsilon=1.0, delta=5)
    print(f"Retina crossings (ε=1.0, δ=5): {len(retina_events)}")
    print(f"V1 crossings (ε=1.0, δ=5): {len(v1_events)}")

if __name__ == "__main__":
    main() 