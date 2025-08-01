# Neural Trajectory Analysis Dashboard

A comprehensive interactive dashboard for analyzing neural trajectory data with dual coordinate systems, crossing detection, and encoding manifold visualization.

## 🎯 Core Features

### **Dual Coordinate Systems**
- **PCA Coordinates**: Dimensionality-reduced trajectories for population-level analysis
- **Neuron Coordinates**: Raw 3-neuron trajectories with encoding manifold integration

### **Interactive 3D Visualizations** 
- Animated trajectory plots with play/pause/slider controls
- Real-time crossing detection with adjustable spatial (ε) and temporal (δ) thresholds
- Hover information showing stimulus/orientation details
- Interactive legends for selective trace display

### **Advanced Analysis Tools**
- Selective trace plotting with checkbox controls and Select All/Deselect All
- Encoding manifold iframe integration for neuron coordinate visualizations  
- Real-time crossing statistics and trajectory metrics
- PSTH (Peristimulus Time Histogram) analysis for individual neurons

### **Technical Excellence**
- **Fixed Path Resolution**: Reliable file discovery regardless of execution context
- **Module-Level Caching**: Optimized performance with smart data management
- **Fresh Plot Generation**: Always accurate, never stale cached results
- **Responsive UI**: Bootstrap-styled interface with professional appearance

### 🎛️ Interactive Controls
- **Parameter Selection**: Epsilon (0.5, 1.0, 1.5) and Delta (3, 5, 7) presets
- **Region Selection**: Retina and V1 datasets
- **Triplet Selection**: Choose from available neuron triplets for neuron coordinates
- **Trace Selection**: Checkboxes for individual stimulus/orientation pairs with Select All/Deselect All
- **Crossing Toggle**: Enable/disable crossing detection and visualization

### 📊 Enhanced Visualization
- **Descriptive Trace Names**: Hover over traces to see stimulus/orientation names (e.g., "Grating W12 - 0°")
- **Interactive Legends**: Click to show/hide individual traces
- **Real-time Statistics**: Crossing counts and averages computed on-demand
- **Loading Feedback**: Clear indicators during computation

## Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Requirements
Ensure the following files are in the `data/` directory:
- `retina_tensor_traces.npy`
- `V1_tensor_traces.npy`
- `retina_cell_info.pkl`
- `V1_cell_info.pkl`

### Encoding Manifold Files
Place HTML encoding manifold files in the root directory with naming pattern:
- `{region}_{distance}_triplet_{instance}_neurons_{neuron1}_{neuron2}_{neuron3}.html`

## Usage

### Running the Dashboard
```bash
python neural_trajectory_dashboard.py
```
Access at: `http://127.0.0.1:8050`

### Performance Note
The dashboard generates plots fresh for each request to ensure accuracy with trace filtering. While this may take 30-200 seconds for complex configurations, it guarantees correct visualization of selected traces.

## File Structure

```
neuron_analysis_results/
├── neural_trajectory_dashboard.py    # Main dashboard application
├── generate_cached_plots.py          # Cache generation script
├── enhanced_trajectory_visualization.py  # Core analysis functions
├── data/                             # Neural data files
├── plot_cache/                       # Cache directory (disabled)
│   └── (cache files not used)
├── requirements.txt                  # Python dependencies
├── Procfile                         # Heroku deployment config
├── render.yaml                      # Render deployment config
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## Performance System

### Current Approach
- **Fresh Generation**: Each plot is generated from scratch for accuracy
- **Trace Filtering**: Only selected stimulus/orientation pairs are plotted
- **Real-time Statistics**: Crossing detection computed for current configuration
- **Optimized Computation**: Efficient algorithms for trajectory analysis

### Performance Characteristics
- **Plot Generation**: 30-200 seconds for complex configurations
- **Memory Usage**: Optimized for web deployment
- **Accuracy**: Guaranteed correct visualization of selected traces
- **Reliability**: No cache-related issues or inconsistencies

## Deployment

### Local Development
```bash
python neural_trajectory_dashboard.py
```

### Render Deployment
The project includes `render.yaml` for easy deployment on Render:
- Automatic Python environment setup
- Port configuration for web services
- Build and start commands

### Heroku Deployment
The project includes `Procfile` for Heroku deployment:
```bash
web: python neural_trajectory_dashboard.py
```

## Technical Details

### Key Functions
- `extract_trajectories_for_region()`: Extract PCA or neuron coordinates
- `detect_crossings_with_temporal_constraint()`: Find trajectory intersections
- `create_3d_trajectory_animation()`: Generate interactive 3D plots
- `parse_triplet_filename()`: Extract neuron indices from HTML filenames
- `update_trace_checkboxes()`: Manage trace selection interface

### Data Flow
1. **Data Loading**: Neural traces loaded from `.npy` files
2. **Trajectory Extraction**: PCA or neuron coordinates computed
3. **Trace Filtering**: Selected stimulus/orientation pairs filtered
4. **Visualization**: 3D animation with optional crossing events
5. **Statistics**: Real-time computation of crossing statistics

### Error Handling
- Clear error messages for missing data
- Robust checkbox parsing and trace filtering
- Safe file path handling for encoding manifolds
- Graceful handling of computation timeouts

## Troubleshooting

### Common Issues
1. **"Module not found"**: Ensure all dependencies installed via `requirements.txt`
2. **"No data files"**: Check `data/` directory contains required `.npy` and `.pkl` files
3. **"No manifold files"**: Verify HTML encoding manifold files are in root directory
4. **Slow plot generation**: Complex configurations may take 30-200 seconds to compute

### Debug Information
The dashboard includes extensive debug logging:
- Parameter values and selections
- Trajectory extraction progress
- Crossing detection results
- File path resolution
- Trace filtering operations

## Performance Tips

1. **Selective Traces**: Use checkboxes to plot only needed traces for faster generation
2. **Parameter Optimization**: Start with smaller epsilon/delta values for quicker computation
3. **Crossing Toggle**: Disable crossing detection when not needed to save computation time
4. **Memory Management**: Close unused browser tabs to free memory

## Future Enhancements

- [ ] Additional coordinate systems
- [ ] More sophisticated crossing detection algorithms
- [ ] Export functionality for plots and statistics
- [ ] Batch processing for multiple configurations
- [ ] Real-time data streaming capabilities 