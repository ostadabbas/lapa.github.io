
# LAPA (Look Around and Pay Attention)

> **🚀 IMPORTANT NOTE 🚀**  
> **This is a submission repository. For the complete project page with demos and additional materials, please open the `index.html` file in this directory.**
> **Also, we have detailed pipeline figure and performance analysis in `105_supplementary.pdf`**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)

This repository contains a PyTorch implementation of LAPA (Look Around and Pay Attention) - a Transformer-Based Multi-Camera Multi-Point Tracker that leverages volumetric attention and geometric constraints to track points across multiple camera views.

## Overview

LAPA is designed for tracking multiple points across multiple camera views with high accuracy. It consists of several key components:

1. **Multi-view encoding**: Processes each camera view to extract features
2. **Volumetric attention mechanism**: Associates points across different camera views using attention and geometric constraints
3. **3D track reconstruction**: Performs triangulation with attention-weighted correspondences
4. **Reconstruction network**: Improves 3D track quality using learned corrections
5. **Interactive visualization**: Provides both video and interactive HTML visualizations

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Camera View 1  │     │  Camera View 2  │     │  Camera View N  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Volumetric Attention Grid                       │
│                                                                 │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                   │
│  │ Point 1 │     │ Point 2 │     │ Point N │                   │
│  │ Corresp.│     │ Corresp.│     │ Corresp.│                   │
│  └────┬────┘     └────┬────┘     └────┬────┘                   │
└───────┼─────────────┼─────────────┼───────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                3D Track Reconstruction                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Reconstruction Network                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        3D Tracks                                │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (recommended for faster training)
- OpenCV
- NumPy
- Matplotlib
- Plotly (for interactive visualizations)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lapa.git
   cd lapa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download datasets:
   - TAP3D dataset (for multi-camera tracking)
   - PointOdyssey dataset (optional, for additional evaluation)

## Dataset Preparation

### TAP3D Dataset

The TAP3D dataset is organized by categories (boxes, basketball, softball, etc.), each containing multiple view sets. The dataset provides:

- 3D tracks in world coordinates (meters)
- Visibility flags for each point
- JPEG-encoded images
- Camera intrinsics [fx, fy, cx, cy]

**Important**: Always use intrinsics directly from the dataset files (`fx_fy_cx_cy`) instead of the calibration file for accurate results.

When using the 224x224 image resizing, properly scale the intrinsics:
```python
width_scale = 224 / orig_width
height_scale = 224 / orig_height
scaled_fx = fx * width_scale
scaled_fy = fy * height_scale
scaled_cx = cx * width_scale
scaled_cy = cy * height_scale
```

The calibration file maps view names (e.g., "boxes_5") to camera names (e.g., "00_05") for retrieving correct camera parameters.

## Training

The training process consists of two main stages:

1. **Attention Model Training**: Trains the volumetric attention mechanism
2. **Reconstruction Model Training**: Trains the reconstruction network to improve 3D track quality

### Training the Attention Model

```bash
python train_with_attention.py \
  --data_dir ./data/tap3d_boxes \
  --calibration_file ./data/tap3d_boxes/calibration_161029_sports1.json \
  --epochs 10 \
  --num_frames 60 \
  --view_set boxes \
  --output_dir ./outputs/attention_training/boxes/boxes
```

Parameters:
- `--data_dir`: Directory containing TAP3D data for a specific category
- `--calibration_file`: Path to the calibration file
- `--epochs`: Number of epochs for training
- `--num_frames`: Number of frames to process per epoch
- `--view_set`: View set to use (e.g., boxes, boxes_alt)
- `--output_dir`: Directory to save training outputs

### Training the Reconstruction Model

```bash
python train_reconstruction.py \
  --data_dir ./data/tap3d_boxes \
  --calibration_file ./data/tap3d_boxes/calibration_161029_sports1.json \
  --epochs 10 \
  --num_frames 60 \
  --view_set boxes \
  --output_dir ./outputs/reconstruction_training/boxes/boxes \
  --attention_checkpoint ./outputs/attention_training/boxes/boxes/attention_model_epoch_10.pth
```

Additional parameters:
- `--attention_checkpoint`: Path to pre-trained attention model (optional)

### Training All Sequences

To train models for all categories and view sets:

```bash
python run_all_sequences.py \
  --data_root ./data \
  --output_root ./outputs \
  --calib_file ./data/tap3d_boxes/calibration_161029_sports1.json \
  --epochs 10 \
  --num_frames 60 \
  --train_attention
```

Parameters:
- `--data_root`: Root directory containing all TAP3D data categories
- `--output_root`: Root directory to save all results
- `--train_attention`: Flag to train attention models before reconstruction
- `--categories`: Specific categories to process (default: all)
- `--view_sets`: Specific view sets to process (default: all for selected categories)

## Loss Components

The training uses three main loss components with different weights:

1. **Reconstruction Loss** (weight 1.0): Dominant component (~128) that measures how well the model reconstructs 2D tracks
2. **Temporal Loss** (weight 0.5): Much smaller (~1.2) that enforces temporal consistency between frames
3. **Identity Loss** (weight 1.0): Similar magnitude to temporal loss (~1.7) that maintains point identity across views

The identity loss is calculated in 2D space using points in each camera view. It is normalized by dividing the pixel coordinates by the image dimensions (width and height) to make the loss scale-invariant, which helps balance it with other loss components.

## Visualization

### Video Visualization

Create dynamic video visualizations showing 2D tracks, 3D reconstructed tracks, and combined visualizations:

```bash
python create_video_visualization.py \
  --categories boxes \
  --view_sets boxes \
  --num_video_frames 60 \
  --track_history 30 \
  --fps 10 \
  --skip_training
```

Parameters:
- `--num_video_frames`: Number of frames to include in video
- `--track_history`: Number of frames to show in track history
- `--fps`: Frames per second for output video
- `--skip_training`: Skip training and only create videos
- `--video_format`: Video format (mp4 or avi)
- `--codec`: Video codec to use (mp4v for mp4, XVID for avi)

### Interactive 3D Visualization

Create interactive HTML visualizations that can be viewed in a web browser:

```bash
python interactive_3d_visualization.py \
  --categories boxes \
  --view_sets boxes \
  --num_video_frames 20 \
  --skip_training
```

The interactive visualization allows:
- Rotating, zooming, and panning the 3D scene
- Toggling visibility of different elements
- Hovering over points to see additional information
- Navigating through frames with a slider

### Combined Visualization

Generate both video and interactive HTML visualizations:

```bash
python create_all_visualizations.py \
  --categories boxes \
  --view_sets boxes \
  --num_video_frames 60 \
  --track_history 30 \
  --fps 10 \
  --skip_training
```

Additional parameters:
- `--skip_video`: Skip video visualization creation
- `--skip_interactive`: Skip interactive HTML visualization creation

## Evaluation

Evaluate the tracking performance:

```bash
python evaluate_tracking.py \
  --data_dir ./data/tap3d_boxes \
  --calibration_file ./data/tap3d_boxes/calibration_161029_sports1.json \
  --view_set boxes \
  --reconstruction_checkpoint ./outputs/reconstruction_training/boxes/boxes/reconstruction_model_epoch_10.pth \
  --attention_checkpoint ./outputs/attention_training/boxes/boxes/attention_model_epoch_10.pth \
  --output_dir ./outputs/evaluation/boxes/boxes
```

The evaluation metrics include:
- Mean Reconstruction Error (pixels)
- Mean 3D Track Error (meters)
- Temporal Consistency Score
- Identity Preservation Score

## Project Structure

```
lapa_code/
├── data/                         # Dataset directory
│   ├── tap3d_boxes/              # TAP3D boxes category
│   ├── tap3d_basketball/         # TAP3D basketball category
│   └── ...
├── lapa/                         # Core LAPA implementation
│   ├── data/                     # Data loading modules
│   ├── models/                   # Model implementations
│   ├── utils/                    # Utility functions
│   ├── visualization/            # Visualization tools
│   └── lapa_pipeline.py          # Main pipeline implementation
├── outputs/                      # Training and visualization outputs
├── debug_code/                   # Debugging utilities
├── train_with_attention.py       # Attention model training script
├── train_reconstruction.py       # Reconstruction model training script
├── run_all_sequences.py          # Script to train all sequences
├── create_video_visualization.py # Video visualization script
├── interactive_3d_visualization.py # Interactive HTML visualization script
├── create_all_visualizations.py  # Combined visualization script
├── evaluate_tracking.py          # Evaluation script
└── requirements.txt              # Project dependencies
```

## Acknowledgments

- TAP3D dataset for multi-camera point tracking
- PointOdyssey dataset for additional evaluation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

