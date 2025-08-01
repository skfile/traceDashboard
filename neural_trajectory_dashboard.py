#!/usr/bin/env python3
"""
Neural Trajectory Analysis Dashboard
===================================

A comprehensive interactive dashboard for analyzing neural trajectory data with:

🎯 CORE FEATURES:
- Dual coordinate systems (PCA vs Neuron coordinates)
- Interactive 3D trajectory visualizations with crossing detection
- Selective trace plotting with checkbox controls
- Encoding manifold integration via HTML iframes
- Real-time statistical analysis and PSTH plots

🔧 TECHNICAL IMPLEMENTATION:
- Built with Dash/Plotly for interactive web visualization
- Fixed path resolution for reliable file discovery
- Module-level caching for optimal performance
- Bootstrap styling for professional UI

📊 ANALYSIS CAPABILITIES:
- Trajectory intersection detection with adjustable thresholds
- Stimulus/orientation labeling with corrected naming
- Cross-region comparison (Retina vs V1)
- Real-time parameter adjustment (epsilon/delta)

Author: Neural Encoding Manifolds Project
Version: 2.0 - Stable Release
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

import os
import glob
import re
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Import functions from the local copy
from enhanced_trajectory_visualization import (
    load_neural_data, extract_pca_trajectories, extract_neuron_trajectories,
    detect_crossings_with_temporal_constraint, create_pca_trajectory_animation,
    create_neuron_trajectory_animation
)

# ============================================================================
# PARAMETER PRESETS
# ============================================================================

EPSILON_PRESETS = [0.5, 1.0, 1.5]
DELTA_PRESETS = [3, 5, 7]

# Cache system removed - all plots generated fresh for accuracy and reliability

# ============================================================================
# STIMULUS INFORMATION
# ============================================================================

def get_correct_stimulus_info():
    """Get the correct stimulus information based on the actual data."""
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
    
    return stimulus_names, directions_deg

# ============================================================================
# TRIPLET DISCOVERY AND PARSING
# ============================================================================

def parse_triplet_filename(filename):
    """
    Parse encoding manifold triplet filename to extract metadata.
    
    Args:
        filename (str): HTML filename in format:
                       "{region}_{type}_triplet_{number}_neurons_{n1}_{n2}_{n3}.html"
    
    Returns:
        dict: Parsed metadata containing region, type, number, neurons, filename
        
    Example:
        "retina_close_triplet_1_neurons_219_908_325.html" -> 
        {
            'region': 'retina',
            'type': 'close', 
            'number': 1,
            'neurons': [219, 908, 325],
            'filename': 'retina_close_triplet_1_neurons_219_908_325.html'
        }
    """
    parts = filename.replace('.html', '').split('_')
    
    return {
        'region': parts[0],        # retina or v1
        'type': parts[1],          # close or far
        'number': int(parts[3]),   # 1 or 2
        'neurons': [int(parts[5]), int(parts[6]), int(parts[7])],  # neuron indices
        'filename': filename
    }

def get_available_triplets():
    """
    Discover and parse all available triplet HTML files.
    
    Uses script directory for reliable path resolution regardless of 
    current working directory when Dash callbacks execute.
    
    Returns:
        dict: Mapping of triplet keys to metadata dictionaries
              Key format: "{region}_{type}_{number}"
    """
    import os
    
    # CRITICAL: Use script directory instead of os.getcwd() 
    # to avoid path resolution issues in Dash callbacks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check both the main directory and encodingMans subdirectory
    triplet_patterns = [
        os.path.join(script_dir, "*_triplet_*.html"),          # Main directory
        os.path.join(script_dir, "encodingMans", "*_triplet_*.html")  # Subdirectory
    ]
    
    triplets = {}
    for pattern in triplet_patterns:
        triplet_files = glob.glob(pattern)
        for file in triplet_files:
            filename = Path(file).name
            try:
                metadata = parse_triplet_filename(filename)
                key = f"{metadata['region']}_{metadata['type']}_{metadata['number']}"
                # Just store the filename - Flask route will find it
                triplets[key] = metadata
            except Exception as e:
                print(f"ERROR: Could not parse triplet file {filename}: {e}")
    
    return triplets

# ============================================================================
# TRAJECTORY EXTRACTION FUNCTIONS
# ============================================================================

def extract_trajectories_for_region(traces, coordinate_type, neuron_indices=None, n_pcs=10):
    """Extract trajectories based on coordinate type."""
    if coordinate_type == 'pca':
        trajectories, pca_model, labels = extract_pca_trajectories(traces, n_pcs=n_pcs)
        return trajectories, pca_model, labels
    else:
        # For neuron coordinates, we need the specific neuron indices
        if neuron_indices is None:
            raise ValueError("Neuron indices required for neuron coordinate system")
        
        trajectories, labels = extract_neuron_trajectories(traces, neuron_indices)
        return trajectories, None, labels

def create_3d_trajectory_animation(trajectories, region, coordinate_type, epsilon, delta, neuron_indices=None, 
                                  stimulus_names=None, directions_deg=None, labels=None, show_crossings=True):
    """
    Create interactive 3D trajectory animation with crossing detection.
    
    This is the core visualization function that generates Plotly 3D animations
    with frame-by-frame trajectory progression, crossing event detection,
    and interactive controls (play/pause/slider).
    
    Args:
        trajectories (list): List of trajectory arrays, each shape (time, 3)
        region (str): 'retina' or 'v1' for title/labeling
        coordinate_type (str): 'pca' or 'neuron' for axis labeling
        epsilon (float): Spatial threshold for crossing detection
        delta (int): Temporal threshold for crossing detection
        neuron_indices (list, optional): Neuron indices for neuron coordinates
        stimulus_names (list, optional): Stimulus names for trace labeling
        directions_deg (list, optional): Direction angles for trace labeling
        labels (list, optional): (stim_idx, dir_idx) tuples for each trajectory
        show_crossings (bool): Whether to detect and display crossing events
    
    Returns:
        plotly.graph_objects.Figure: Interactive 3D animation with controls
    """
    
    # Detect crossing events only if show_crossings is True
    events = []
    event_frames = {}
    
    if show_crossings:
        events = detect_crossings_with_temporal_constraint(trajectories, epsilon, delta)
        
        # Create event lookup by frame for efficient access
        for event in events:
            for traj_id, time_idx in zip(event['traj_ids'], event['times']):
                if time_idx not in event_frames:
                    event_frames[time_idx] = []
                event_frames[time_idx].append({
                    'coord': event['coord'],
                    'traj_id': traj_id
                })
    
    # Collect all crossing coordinates that occur up to each frame
    def get_crossings_up_to_frame(frame_idx):
        """Get all unique crossing coordinates that have occurred up to and including frame_idx"""
        if not show_crossings:
            return []
            
        crossing_coords = []
        for f in range(frame_idx + 1):
            if f in event_frames:
                for event_info in event_frames[f]:
                    coord = event_info['coord'][:3]  # Take first 3 dimensions
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
    
    # Handle both list of trajectories and single trajectory array
    if isinstance(trajectories, list):
        n_frames = trajectories[0].shape[0]
    else:
        n_frames = trajectories.shape[0]
    
    # Create trajectory names based on labels
    trajectory_names = []
    if labels and stimulus_names and directions_deg:
        for i, (stim_idx, dir_idx) in enumerate(labels):
            if stim_idx < len(stimulus_names) and dir_idx < len(directions_deg):
                trajectory_names.append(f"{stimulus_names[stim_idx]} - {directions_deg[dir_idx]}°")
            else:
                trajectory_names.append(f"Trajectory {i}")
    else:
        trajectory_names = [f"Trajectory {i}" for i in range(len(trajectories))]
    
    # Initial frame data (frame 0)
    init_data = []
    
    # Add trajectory lines (just starting points)
    for i, traj in enumerate(trajectories):
        init_data.append(
            go.Scatter3d(
                x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                mode='lines',
                line=dict(width=3, color=f'rgba({50+i*30%200},{100+i*20%200},{150+i*40%200},0.8)'),
                name=trajectory_names[i],
                hovertemplate=f'<b>{trajectory_names[i]}</b><br>PC1: %{{x}}<br>PC2: %{{y}}<br>PC3: %{{z}}<extra></extra>',
                showlegend=True
            )
        )
    
    # Add start points (green dots)
    for i, traj in enumerate(trajectories):
        init_data.append(
            go.Scatter3d(
                x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                name=f'Start {i}',
                showlegend=False
            )
        )
    
    # Add end points (black dots)
    for i, traj in enumerate(trajectories):
        init_data.append(
            go.Scatter3d(
                x=[traj[-1,0]], y=[traj[-1,1]], z=[traj[-1,2]],
                mode='markers',
                marker=dict(size=8, color='black', symbol='circle'),
                name=f'End {i}',
                showlegend=False
            )
        )
    
    # Add crossings for initial frame (frame 0)
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
                showlegend=True
            )
        )
    else:
        # Add empty crossing trace for consistency
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
            frame_data.append(
                go.Scatter3d(
                    x=traj[:t+1,0], y=traj[:t+1,1], z=traj[:t+1,2],
                    mode='lines',
                    line=dict(width=3, color=f'rgba({50+i*30%200},{100+i*20%200},{150+i*40%200},0.8)'),
                    name=f'Traj {i}',
                    hovertemplate=f'<b>{trajectory_names[i]}</b><br>PC1: %{{x}}<br>PC2: %{{y}}<br>PC3: %{{z}}<extra></extra>',
                    showlegend=False
                )
            )
        
        # Add start points (green dots)
        for i, traj in enumerate(trajectories):
            frame_data.append(
                go.Scatter3d(
                    x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
                    mode='markers',
                    marker=dict(size=8, color='green', symbol='circle'),
                    name=f'Start {i}',
                    showlegend=False
                )
            )
        
        # Add end points (black dots)
        for i, traj in enumerate(trajectories):
            frame_data.append(
                go.Scatter3d(
                    x=[traj[-1,0]], y=[traj[-1,1]], z=[traj[-1,2]],
                    mode='markers',
                    marker=dict(size=8, color='black', symbol='circle'),
                    name=f'End {i}',
                    showlegend=False
                )
            )
        
        # Add all crossing events that have occurred up to current frame
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
                    showlegend=False
                )
            )
        else:
            # Add empty crossing trace to maintain consistent trace structure
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
    
    # Create slider for frame control
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
    
    # Create figure with enhanced controls
    coord_label = "PCA Coordinates" if coordinate_type == 'pca' else f"Neuron Coordinates {neuron_indices}"
    
    fig = go.Figure(
        data=init_data,
        frames=frames,
        layout=go.Layout(
            title=f"{region.upper()} Neural Trajectories - {coord_label}<br>ε={epsilon}, δ={delta}",
            scene=dict(
                xaxis_title='PC1' if coordinate_type == 'pca' else 'Neuron 1',
                yaxis_title='PC2' if coordinate_type == 'pca' else 'Neuron 2', 
                zaxis_title='PC3' if coordinate_type == 'pca' else 'Neuron 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            updatemenus=[{
                'type': 'buttons',
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 200, 'redraw': True},
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
                    text=f"Green: Start | Black: End | Red X: Crossings (threshold={epsilon})",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.1, xanchor='center', yanchor='bottom'
                )
            ]
        )
    )
    
    return fig

# ============================================================================
# PSTH FUNCTIONS
# ============================================================================

def compute_single_neuron_psth(traces, neuron_idx, stimulus_idx, direction_idx):
    """Compute PSTH for a specific neuron and stimulus/direction combination."""
    psth = traces[neuron_idx, stimulus_idx, direction_idx, :]
    time_points = np.arange(len(psth))
    return time_points, psth

def create_single_neuron_psth_plot(traces, stimulus_names, directions_deg, 
                                 neuron_idx, stimulus_idx, direction_idx):
    """Create a PSTH plot for a specific neuron and stimulus/direction."""
    if neuron_idx is None or stimulus_idx is None or direction_idx is None:
        return go.Figure().add_annotation(
            text="Select a neuron, stimulus, and direction to view PSTH",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
    
    try:
        time_points, psth = compute_single_neuron_psth(traces, neuron_idx, stimulus_idx, direction_idx)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=psth,
            mode='lines+markers',
            line=dict(width=3, color='blue'),
            marker=dict(size=4, color='blue'),
            name=f'Neuron {neuron_idx}'
        ))
        
        # Add stimulus onset marker
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="Stimulus Onset")
        
        # Add mean firing rate line
        mean_rate = np.mean(psth)
        fig.add_hline(y=mean_rate, line_dash="dot", line_color="green",
                     annotation_text=f"Mean: {mean_rate:.2f}")
        
        fig.update_layout(
            title=f"Neuron {neuron_idx} PSTH: {stimulus_names[stimulus_idx]} - {directions_deg[direction_idx]}°",
            xaxis_title="Time (frames)",
            yaxis_title="Firing Rate (Hz)",
            height=400,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating PSTH: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

# Initialize Dash application with Bootstrap styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Configure Flask route for serving HTML manifold files
import flask
import os

# Get script directory for consistent path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))

# Use a more specific route to avoid conflicts with Dash
@app.server.route('/manifolds/<path:filename>')
def serve_manifold(filename):
    """Serve HTML manifold files as static content for iframe display."""
    print(f"🌐 Manifold file request: /manifolds/{filename}")
    
    if not filename.endswith('.html') or 'triplet' not in filename:
        print(f"❌ Invalid file request: {filename}")
        return "Not allowed", 403
    
    try:
        # Check if it's in the main directory first
        main_path = os.path.join(script_dir, filename)
        if os.path.exists(main_path):
            print(f"🎯 Serving from main directory: {main_path}")
            return flask.send_file(main_path)
        
        # Check in encodingMans subdirectory
        sub_path = os.path.join(script_dir, 'encodingMans', filename)
        if os.path.exists(sub_path):
            print(f"🎯 Serving from encodingMans directory: {sub_path}")
            return flask.send_file(sub_path)
        
        print(f"❌ File not found: {filename}")
        return f"File not found: {filename}", 404
        
    except Exception as e:
        print(f"❌ Error serving {filename}: {e}")
        return f"Server error: {e}", 500

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

print("🔄 Loading neural data...")
retina_traces, v1_traces, retina_info, v1_info, _, _ = load_neural_data()
print(f"✅ Loaded retina: {retina_traces.shape}, V1: {v1_traces.shape}")

# Load stimulus information
stimulus_names, directions_deg = get_correct_stimulus_info()
print(f"✅ Loaded {len(stimulus_names)} stimuli, {len(directions_deg)} directions")

# Build triplet cache with fixed path resolution
TRIPLETS_CACHE = get_available_triplets()
print(f"✅ Found {len(TRIPLETS_CACHE)} triplet files:")
for key, metadata in TRIPLETS_CACHE.items():
    print(f"   {key}: {metadata['region']} {metadata['type']} {metadata['number']} - neurons {metadata['neurons']}")

# Get neuron counts for dropdowns
retina_neuron_count = retina_traces.shape[0]
v1_neuron_count = v1_traces.shape[0]

# Pre-compute trajectory data for performance
print("🔄 Pre-computing trajectory data...")
retina_trajs, _, _ = extract_pca_trajectories(retina_traces, n_pcs=10)
v1_trajs, _, _ = extract_pca_trajectories(v1_traces, n_pcs=10)
print("✅ Trajectories pre-computed!")

# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Main Neural Trajectory Analysis Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Parameter Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Parameter Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Epsilon (Spatial Threshold):"),
                            dcc.RadioItems(
                                id='epsilon-radio',
                                options=[{'label': f'ε = {eps}', 'value': eps} for eps in EPSILON_PRESETS],
                                value=0.5,
                                inline=True
                            ),
                        ], width=4),
                        dbc.Col([
                            html.Label("Delta (Temporal Threshold):"),
                            dcc.RadioItems(
                                id='delta-radio',
                                options=[{'label': f'δ = {del_val}', 'value': del_val} for del_val in DELTA_PRESETS],
                                value=5,
                                inline=True
                            ),
                        ], width=4),
                        dbc.Col([
                            html.Label("Show Crossing Events:"),
                            dcc.RadioItems(
                                id='show-crossings-radio',
                                options=[
                                    {'label': 'Yes', 'value': True},
                                    {'label': 'No', 'value': False}
                                ],
                                value=True,
                                inline=True
                            ),
                        ], width=4)
                    ]),
                    html.Div(id="current-params", className="mt-2")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Coordinate System & Region Selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Visualization Settings"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Brain Region:"),
                            dcc.Dropdown(
                                id='region-dropdown',
                                options=[
                                    {'label': 'Retina', 'value': 'retina'},
                                    {'label': 'V1', 'value': 'v1'}
                                ],
                                value='retina',
                                clearable=False
                            ),
                        ], width=4),
                        dbc.Col([
                            html.Label("Coordinate System:"),
                            dcc.Dropdown(
                                id='coordinate-type-dropdown',
                                options=[
                                    {'label': 'PCA Coordinates', 'value': 'pca'},
                                    {'label': 'Neuron Coordinates', 'value': 'neuron'}
                                ],
                                value='pca',
                                clearable=False
                            ),
                        ], width=4),
                        dbc.Col([
                            html.Label("Neuron Triplet (for neuron coordinates):"),
                            dcc.Dropdown(
                                id='triplet-dropdown',
                                options=[],  # Will be populated based on region
                                value=None,
                                clearable=True
                            ),
                        ], width=4)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Trace Selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trace Selection"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Select All", id="select-all-btn", color="success", size="sm", className="me-2"),
                            dbc.Button("Deselect All", id="deselect-all-btn", color="warning", size="sm")
                        ], width=12)
                    ], className="mb-3"),
                    html.Div(id="trace-checkboxes", style={'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Control Buttons
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Create Visualization", 
                id="create-viz-btn", 
                color="primary", 
                size="lg", 
                className="w-100"
            ),
            html.Div(id="loading-output", className="mt-2")
        ])
    ], className="mb-4"),
    
    # Statistics Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Crossing Statistics"),
                dbc.CardBody(id="stats-output")
            ])
        ])
    ], className="mb-4"),
    
    # Visualization Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Neural Trajectory Visualization"),
                dbc.CardBody([
                    dcc.Graph(id="trace-plot", style={'height': '700px'})
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Encoding Manifold Panel (only for neuron coordinates)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Encoding Manifold"),
                dbc.CardBody(id="manifold-container")
            ])
        ])
    ], className="mb-4"),
    
    # PSTH Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Single Neuron PSTH"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Neuron Index:"),
                            dcc.Dropdown(
                                id='psth-neuron-dropdown',
                                options=[{'label': f'Neuron {i}', 'value': i} for i in range(min(retina_neuron_count, v1_neuron_count))],
                                value=0,
                                clearable=False
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("Stimulus:"),
                            dcc.Dropdown(
                                id='psth-stimulus-dropdown',
                                options=[{'label': name, 'value': i} for i, name in enumerate(stimulus_names)],
                                value=0,
                                clearable=False
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("Direction:"),
                            dcc.Dropdown(
                                id='psth-direction-dropdown',
                                options=[{'label': f"{d}°", 'value': i} for i, d in enumerate(directions_deg)],
                                value=0,
                                clearable=False
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Label("Region:"),
                            dcc.Dropdown(
                                id='psth-region-dropdown',
                                options=[
                                    {'label': 'Retina', 'value': 'retina'},
                                    {'label': 'V1', 'value': 'v1'}
                                ],
                                value='retina',
                                clearable=False
                            ),
                        ], width=3)
                    ], className="mb-3"),
                    dcc.Graph(id="psth-plot", style={'height': '400px'})
                ])
            ])
        ])
    ], className="mt-4")
    
], fluid=True)

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output("current-params", "children"),
    [Input("epsilon-radio", "value"),
     Input("delta-radio", "value"),
     Input("show-crossings-radio", "value")]
)
def update_parameters(epsilon, delta, show_crossings):
    """Update current parameter display."""
    crossing_status = "Enabled" if show_crossings else "Disabled"
    return f"Current Parameters: ε = {epsilon}, δ = {delta}, Crossings: {crossing_status}"

@app.callback(
    Output("loading-output", "children"),
    [Input("create-viz-btn", "n_clicks")]
)
def show_loading(n_clicks):
    """Show loading message when button is clicked."""
    if n_clicks is None:
        return ""
    else:
        return dbc.Alert(
            "🔄 Computing visualization... Please wait...",
            color="info",
            dismissable=True
        )

@app.callback(
    Output("triplet-dropdown", "options"),
    [Input("region-dropdown", "value")]
)
def update_triplet_options(region):
    """Update triplet dropdown based on selected region."""
    print(f"DEBUG TRIPLET CALLBACK: Triggered for region: {region}")
    print(f"DEBUG TRIPLET CALLBACK: Cache contains {len(TRIPLETS_CACHE)} triplets: {list(TRIPLETS_CACHE.keys())}")
    
    if region is None:
        print("DEBUG TRIPLET CALLBACK: No region selected, returning empty options")
        return []
    
    # Filter cached triplets for the selected region
    region_triplets = {k: v for k, v in TRIPLETS_CACHE.items() if v['region'] == region}
    print(f"DEBUG TRIPLET CALLBACK: Found {len(region_triplets)} triplets for {region}")
    print(f"DEBUG TRIPLET CALLBACK: Region triplets: {list(region_triplets.keys())}")
    
    options = []
    for key, metadata in region_triplets.items():
        label = f"{metadata['type'].title()} Triplet {metadata['number']} (Neurons {metadata['neurons'][0]}, {metadata['neurons'][1]}, {metadata['neurons'][2]})"
        options.append({'label': label, 'value': key})
        print(f"DEBUG TRIPLET CALLBACK: Added option: {label} -> {key}")
    
    print(f"DEBUG TRIPLET CALLBACK: Final options list: {options}")
    return options

@app.callback(
    Output("trace-checkboxes", "children"),
    [Input("region-dropdown", "value"),
     Input("select-all-btn", "n_clicks"),
     Input("deselect-all-btn", "n_clicks")]
)
def update_trace_checkboxes(region, select_clicks, deselect_clicks):
    """Simplified checkbox state management."""
    if region is None:
        return []
    
    # Determine default state based on which button was clicked
    ctx = callback_context
    default_value = True  # Default to all selected
    
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "select-all-btn":
            default_value = True
        elif button_id == "deselect-all-btn":
            default_value = False
    
    # Create checkboxes with consistent labeling
    checkboxes = []
    for stim_idx in range(len(stimulus_names)):
        for dir_idx in range(len(directions_deg)):
            trace_id = f"trace-{stim_idx}-{dir_idx}"
            trace_label = f"{stimulus_names[stim_idx]} - {directions_deg[dir_idx]}°"
            
            checkboxes.append(
                dbc.Checkbox(
                    id=trace_id,
                    label=trace_label,
                    value=default_value,
                    className="me-3"
                )
            )
    
    return checkboxes

@app.callback(
    [Output("trace-plot", "figure"),
     Output("manifold-container", "children"),
     Output("stats-output", "children")],
    [Input("create-viz-btn", "n_clicks")],
    [State("epsilon-radio", "value"),
     State("delta-radio", "value"),
     State("coordinate-type-dropdown", "value"),
     State("region-dropdown", "value"),
     State("triplet-dropdown", "value"),
     State("trace-checkboxes", "children"),
     State("show-crossings-radio", "value")]
)
def create_visualization(n_clicks, epsilon, delta, coordinate_type, region, triplet_selection, checkbox_children, show_crossings):
    """
    Main visualization callback - generates 3D trajectory plots with analysis.
    
    This is the core callback that orchestrates:
    1. Parameter validation and trace extraction
    2. Trajectory computation (PCA or neuron coordinates)  
    3. Selective trace filtering based on checkbox states
    4. 3D animation generation with crossing detection
    5. Encoding manifold iframe display (neuron coordinates only)
    6. Statistical analysis and summary generation
    
    Returns:
        tuple: (plotly.Figure, html.Div, list) - plot, manifold, statistics
    """
    print(f"🎯 Visualization requested: ε={epsilon}, δ={delta}, {coordinate_type}, {region}")
    
    if n_clicks is None:
        return go.Figure(), "", "Click 'Create Visualization' to generate plots"
    
    try:
        print("🔄 Starting visualization creation...")
        start_time = time.time()
        
        # Get the appropriate traces and extract trajectories
        traces = retina_traces if region == 'retina' else v1_traces
        print(f"DEBUG: Using {region} traces with shape {traces.shape}")
        
        # Get neuron indices if using neuron coordinates
        neuron_indices = None
        if coordinate_type == 'neuron':
            if triplet_selection is None:
                print("DEBUG: No triplet selected for neuron coordinates")
                return go.Figure(), "", "Please select a neuron triplet for neuron coordinates"
            
            # Use cached triplets
            triplet_metadata = TRIPLETS_CACHE[triplet_selection]
            neuron_indices = triplet_metadata['neurons']
            print(f"DEBUG: Using neuron indices {neuron_indices}")
        
        # Extract trajectories (needed for both cached and fresh plots)
        print("DEBUG: Extracting trajectories...")
        trajectories, pca_model, neuron_coords = extract_trajectories_for_region(
            traces, coordinate_type, neuron_indices
        )
        print(f"DEBUG: Extracted {len(trajectories)} trajectories")
        
        # Parse checkbox states to determine which traces to plot
        selected_traces = []
        if checkbox_children:
            for child in checkbox_children:
                if isinstance(child, dbc.Checkbox):
                    if hasattr(child, 'value') and child.value:  # Checkbox is checked
                        # Extract stim_idx and dir_idx from checkbox ID (format: trace-{stim_idx}-{dir_idx})
                        trace_id = child.id
                        if trace_id and trace_id.startswith('trace-'):
                            try:
                                parts = trace_id.split('-')
                                if len(parts) == 3:
                                    stim_idx = int(parts[1])
                                    dir_idx = int(parts[2])
                                    selected_traces.append(f"{stim_idx}_{dir_idx}")
                            except (ValueError, IndexError):
                                print(f"DEBUG: Could not parse checkbox ID: {trace_id}")
        
        print(f"DEBUG: Parsed {len(selected_traces)} selected traces from checkboxes: {selected_traces}")
        
        # If no traces selected, use all traces
        if not selected_traces:
            selected_traces = [f"{stim_idx}_{dir_idx}" 
                             for stim_idx in range(len(stimulus_names))
                             for dir_idx in range(len(directions_deg))]
            print("DEBUG: No traces selected, using all traces")
        
        # Filter trajectories based on selected traces
        filtered_trajectories = []
        filtered_labels = []
        
        for i, (stim_idx, dir_idx) in enumerate([(s, d) for s in range(len(stimulus_names)) for d in range(len(directions_deg))]):
            trace_id = f"{stim_idx}_{dir_idx}"
            if trace_id in selected_traces:
                filtered_trajectories.append(trajectories[i])
                # Always use (stim_idx, dir_idx) for proper labeling
                filtered_labels.append((stim_idx, dir_idx))
        
        print(f"DEBUG: Filtered to {len(filtered_trajectories)} trajectories")
        
        # Update trajectories and labels for plotting
        trajectories = filtered_trajectories
        labels = filtered_labels
        
        # Create 3D trajectory animation
        print("DEBUG: Creating 3D trajectory animation...")
        
        # Generate fresh plot
        print("DEBUG: Generating fresh plot")
        fig = create_3d_trajectory_animation(trajectories, region, coordinate_type, epsilon, delta, 
                                            neuron_indices, stimulus_names, directions_deg, labels, show_crossings)
        
        print("DEBUG: 3D animation created successfully")
        
        # Create manifold display (only for neuron coordinates)
        manifold_html = ""
        print(f"🔍 MANIFOLD CHECK: coordinate_type='{coordinate_type}', triplet_selection='{triplet_selection}'")
        print(f"🔍 CONDITION: (coordinate_type == 'neuron') = {coordinate_type == 'neuron'}")
        print(f"🔍 CONDITION: (triplet_selection) = {bool(triplet_selection)}")
        
        if coordinate_type == 'neuron' and triplet_selection:
            print(f"🧠 ✅ CREATING manifold display for triplet: {triplet_selection}")
            # Use cached triplets
            triplet_metadata = TRIPLETS_CACHE[triplet_selection]
            filename = triplet_metadata['filename']
            print(f"🧠 Manifold filename: {filename}")
            
            # Check if file exists (Flask route will handle the actual serving)
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Check both possible locations
            main_path = os.path.join(script_dir, filename)
            sub_path = os.path.join(script_dir, 'encodingMans', filename)
            
            file_exists = os.path.exists(main_path) or os.path.exists(sub_path)
            print(f"🧠 File check - Main: {os.path.exists(main_path)}, Sub: {os.path.exists(sub_path)}")
            
            if file_exists:
                iframe_src = f"/manifolds/{filename}"
                print(f"🌐 Creating iframe with src: {iframe_src}")
                # Use iframe with src pointing to file (served as static)
                manifold_html = html.Div([
                    html.H5(f"Encoding Manifold: {triplet_metadata['type'].title()} Triplet {triplet_metadata['number']}"),
                    html.P(f"Neurons: {triplet_metadata['neurons']}"),
                    html.Iframe(
                        src=iframe_src,
                        style={'width': '100%', 'height': '600px', 'border': '1px solid #ddd'},
                        id="manifold-iframe",  # Add ID for debugging
                        sandbox="allow-scripts allow-same-origin"  # Allow iframe content to load
                    )
                ])
                print("✅ Manifold iframe created successfully")
            else:
                print(f"❌ Manifold file not found: {filename}")
                manifold_html = html.Div([
                    html.P(f"Manifold file not found: {filename}", style={'color': 'red'})
                ])
        else:
            print(f"ℹ️ ❌ SKIPPING manifold display: coordinate_type={coordinate_type}, triplet_selection={triplet_selection}")
            manifold_html = html.Div([
                html.P("Select 'Neuron Coordinates' and a neuron triplet to view encoding manifold", 
                       style={'color': 'gray', 'font-style': 'italic'})
            ])
        
        # Compute fresh statistics
        stats_text = [
            html.P(f"Region: {region.upper()}"),
            html.P(f"Coordinate System: {'PCA' if coordinate_type == 'pca' else 'Neuron'}"),
            html.P(f"Parameters: ε = {epsilon}, δ = {delta}"),
            html.P(f"Total Trajectories: {len(trajectories)}")
        ]
        
        if show_crossings:
            print("DEBUG: Computing crossing statistics...")
            crossing_events = detect_crossings_with_temporal_constraint(trajectories, epsilon, delta)
            print(f"DEBUG: Found {len(crossing_events)} crossing events")
            stats_text.extend([
                html.P(f"Total Crossings: {len(crossing_events)}"),
                html.P(f"Average Crossings per Trajectory: {len(crossing_events)/len(trajectories):.2f}")
            ])
        else:
            print("DEBUG: Skipping crossing statistics computation")
            stats_text.append(html.P("Crossing Detection: Disabled"))
        
        if coordinate_type == 'neuron' and neuron_indices:
            stats_text.append(html.P(f"Neuron Indices: {neuron_indices}"))
        
        print("DEBUG: Visualization creation completed successfully")
        return fig, manifold_html, stats_text
        
    except Exception as e:
        print(f"DEBUG: Error in visualization creation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_fig = go.Figure().add_annotation(
            text=f"❌ Error creating visualization:<br>{str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="red")
        )
        error_fig.update_layout(
            title="Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return error_fig, "", f"Error: {str(e)}"

@app.callback(
    Output("psth-plot", "figure"),
    [Input("psth-neuron-dropdown", "value"),
     Input("psth-stimulus-dropdown", "value"),
     Input("psth-direction-dropdown", "value"),
     Input("psth-region-dropdown", "value")]
)
def update_psth(neuron_idx, stimulus_idx, direction_idx, region):
    """Update PSTH plot for specific neuron and stimulus/direction."""
    if neuron_idx is None or stimulus_idx is None or direction_idx is None:
        return go.Figure().add_annotation(
            text="Select a neuron, stimulus, and direction to view PSTH",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
    
    # Get the appropriate traces based on region
    traces = retina_traces if region == 'retina' else v1_traces
    
    return create_single_neuron_psth_plot(traces, stimulus_names, directions_deg, 
                                        neuron_idx, stimulus_idx, direction_idx)

# ============================================================================
# RUN DASHBOARD
# ============================================================================

if __name__ == '__main__':
    import os
    
    # Get port from environment variable (for Render) or use default
    port = int(os.environ.get('PORT', 8050))
    
    # Use 0.0.0.0 for production (Render) or 127.0.0.1 for local development
    host = '0.0.0.0' if os.environ.get('RENDER') else '127.0.0.1'
    
    print(f"Starting Neural Trajectory Dashboard...")
    print(f"Open your browser and go to: http://{host}:{port}")
    
    # Disable debug mode in production
    debug = not os.environ.get('RENDER')
    
    app.run(debug=debug, host=host, port=port) 