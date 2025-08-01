
# Neural Encoding Manifolds - Stimulus Mapping Documentation

## Overview

This document provides the **corrected stimulus mapping** for the neural encoding manifolds project. The original documentation was incomplete and misleading - this document provides the accurate mapping based on the `creating-the-tensor` folder analysis.

## ✅ CORRECTED STIMULUS MAPPING

### Complete Stimulus List

| Index | Display Name      | Tensor Name        | Description |
|-------|------------------|-------------------|-------------|
| 0     | Low SF Grating   | gratW12.5         | Low spatial frequency grating stimulus |
| 1     | High SF Grating  | gratW2            | High spatial frequency grating stimulus |
| 2     | Neg 1-dot Flow   | -1dotD2s2bg       | Negative 1-dot optic flow stimulus |
| 3     | Neg 3-dot Flow   | -3dotD2s2bg       | Negative 3-dot optic flow stimulus |
| 4     | **Pos 1-dot Flow** | +1dotD2s2bg    | **Positive 1-dot optic flow stimulus** |
| 5     | **Pos 3-dot Flow** | +3dotD2s2bg    | **Positive 3-dot optic flow stimulus** |

## 🎯 Key Discovery

**POSITIVE FLOW STIMULI ARE PRESENT IN THE NEURAL DATA TENSOR!**

- ✅ Indices 4-5 contain positive flow stimuli
- ✅ All 6 stimuli are included in the tensor
- ✅ The original metadata.json was incomplete/misleading
- ✅ Cell info correctly shows positive flow responses

## Stimulus Descriptions

### Grating Stimuli (Indices 0-1)

**Low SF Grating (Index 0)**
- **Tensor Name**: `gratW12.5`
- **Description**: Low spatial frequency grating stimulus
- **Purpose**: Tests orientation selectivity with low spatial frequency patterns

**High SF Grating (Index 1)**
- **Tensor Name**: `gratW2`
- **Description**: High spatial frequency grating stimulus  
- **Purpose**: Tests orientation selectivity with high spatial frequency patterns

### Negative Flow Stimuli (Indices 2-3)

**Neg 1-dot Flow (Index 2)**
- **Tensor Name**: `-1dotD2s2bg`
- **Description**: Negative 1-dot optic flow stimulus
- **Purpose**: Tests motion-in-depth processing with contraction (negative flow)

**Neg 3-dot Flow (Index 3)**
- **Tensor Name**: `-3dotD2s2bg`
- **Description**: Negative 3-dot optic flow stimulus
- **Purpose**: Tests motion-in-depth processing with contraction using 3-dot patterns

### Positive Flow Stimuli (Indices 4-5) ⭐

**Pos 1-dot Flow (Index 4)**
- **Tensor Name**: `+1dotD2s2bg`
- **Description**: Positive 1-dot optic flow stimulus
- **Purpose**: Tests motion-in-depth processing with expansion (positive flow)

**Pos 3-dot Flow (Index 5)**
- **Tensor Name**: `+3dotD2s2bg`
- **Description**: Positive 3-dot optic flow stimulus
- **Purpose**: Tests motion-in-depth processing with expansion using 3-dot patterns

## Data Structure

### Tensor Dimensions
- **Retina**: `(1146 neurons, 6 stimuli, 8 directions, 135 time bins)`
- **V1**: `(637 neurons, 6 stimuli, 8 directions, 135 time bins)`

### Directions
All stimuli are presented in 8 directions:
- 0° (horizontal right)
- 45° (diagonal up-right)
- 90° (vertical up)
- 135° (diagonal up-left)
- 180° (horizontal left)
- 225° (diagonal down-left)
- 270° (vertical down)
- 315° (diagonal down-right)

## What Was Fixed

### Before (Incorrect)
```python
stimulus_names = [
    "Grating W12", "Grating W1", "Grating W2",
    "Neg 1-dot Flow D1", "Neg 3-dot Flow D1", "Neg 1-dot Flow D2"
]
```

### After (Correct)
```python
stimulus_names = [
    "Low SF Grating", "High SF Grating",
    "Neg 1-dot Flow", "Neg 3-dot Flow", 
    "Pos 1-dot Flow", "Pos 3-dot Flow"
]
```

## Files Updated

The following files have been corrected with the proper stimulus names:

1. `neuron_analysis_results/enhanced_trajectory_visualization.py`
2. `neuron_analysis_results/neural_trajectory_dashboard.py`
3. `enhanced_trajectory_visualization.py`
4. `enhanced_cached_dashboard.py`
5. `static_trajectory_dashboard.py`
6. `demo_dashboard.py`
7. `neuron_analysis_results/generate_cached_plots.py`
8. `static_data/metadata.json`

## Implications for Analysis

### Neural Responses
- **Grating stimuli** (0-1): Test orientation selectivity
- **Negative flow** (2-3): Test contraction motion-in-depth processing
- **Positive flow** (4-5): Test expansion motion-in-depth processing

### Experimental Design
The stimulus set provides a comprehensive test of:
1. **Spatial processing** (gratings)
2. **Motion processing** (flow stimuli)
3. **Depth processing** (expansion vs contraction)

### Analysis Opportunities
- Compare responses between positive and negative flow
- Analyze orientation tuning vs motion tuning
- Study depth processing in visual hierarchy (retina vs V1)

## Conclusion

The neural data tensor contains **all 6 stimulus types**, including the positive flow stimuli that were previously thought to be missing. The corrected mapping reveals a well-designed experimental paradigm that tests multiple aspects of visual processing.

**Key Takeaway**: Always trace back to the original data creation process rather than relying on potentially incomplete documentation! 