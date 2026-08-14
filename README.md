# 3D Volumetric Brain MRI Segmentation and Multi-Planar Reconstruction

An end-to-end volumetric medical image processing and deep learning pipeline built using MONAI (Medical Open Network for AI) and PyTorch. The system processes 3D NIfTI neuroimaging scans (`.nii.gz`), executes standardized spatial and intensity transforms, runs volumetric 3D UNet segmentation inference, and performs multi-planar orthogonal reconstruction across Axial, Coronal, and Sagittal anatomical planes.

---

## Table of Contents

- [Overview](#overview)
- [Multi-Planar Orthogonal Visualization](#multi-planar-orthogonal-visualization)
- [Methodology and Preprocessing Pipeline](#methodology-and-preprocessing-pipeline)
- [3D Architecture Details](#3d-architecture-details)
- [Technical Stack](#technical-stack)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [License](#license)

---

## Overview

Clinical neuroimaging workflows rely heavily on volumetric 3D scans such as Magnetic Resonance Imaging (MRI) and Computed Tomography (CT). Unlike standard 2D computer vision tasks, 3D medical image analysis requires handling continuous spatial voxel spacing, anatomical coordinate reorientation (RAS/LPS standards), and high-dimensional volumetric tensors.

This repository implements a modular MONAI and PyTorch pipeline designed to:
1. Standardize 3D compressed NIfTI neuroimaging volumes.
2. Apply affine reorientation and isotropic voxel resampling.
3. Perform 3D convolutional segmentation inference using a volumetric UNet architecture.
4. Extract synchronized orthogonal cross-sections (Axial, Coronal, Sagittal) with segmented lesion overlays.

---

## Multi-Planar Orthogonal Visualization

The pipeline outputs synchronized cross-sectional reconstructions across all three primary spatial axes, mapping localized pathological regions directly onto structural brain scans:

![3D Multi-Planar Reconstruction](sample_3d_mpr_visualization.png)

---

## Methodology and Preprocessing Pipeline

### Dictionary-Based Data Transforms (MONAI)
* **Spatial Reorientation:** Aligns voxel arrays to standardized `RAS` (Right-Anterior-Superior) anatomical coordinate space via `Orientationd`.
* **Isotropic Resampling:** Resamples heterogeneous voxel dimensions to uniform spatial resolution (1.5 x 1.5 x 1.5 mm) via bilinear interpolation using `Spacingd`.
* **Intensity Normalization:** Scales voxel intensities based on 1st-to-99th percentile distributions via `ScaleIntensityRangePercentilesd` to eliminate scanner contrast variability.

---

## 3D Architecture Details

### Volumetric UNet Model
* **Convolutions:** 3D spatial convolutional kernels (3 x 3 x 3).
* **Residual Units:** Two residual units per resolution stage to preserve fine spatial context across volumetric feature hierarchies.
* **Channel Progression:** Multi-scale feature channels (16 -> 32 -> 64 -> 128).
* **Striding:** 3D spatial downsampling with stride (2, 2, 2).

---

## Technical Stack

* **Deep Learning Framework:** MONAI, PyTorch, Torchvision
* **Medical Data Standards:** SimpleITK, Nibabel
* **Numerical Computation:** NumPy, SciPy
* **Visualization & Rendering:** Matplotlib Multi-Planar Slicing Engine

---

## Repository Structure

```text

Installation and Setup
1. Clone the Repository
Bash
git clone [https://github.com/cyberpunk92/3d-brain-mri-monai.git](https://github.com/cyberpunk92/3d-brain-mri-monai.git)
cd 3d-brain-mri-monai
2. Create and Activate a Virtual Environment
On Windows (PowerShell):

PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
On Linux/macOS:

Bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
Usage
Run the primary pipeline script to execute preprocessing, model inference, and multi-planar visual artifact generation:

Bash
python main.py
Upon execution, the script produces:

brain_mri_volume.nii.gz: Compressed volumetric NIfTI 3D scan.

sample_3d_mpr_visualization.png: Synchronized 3-axis orthogonal reconstruction figure.

License
This project is open-source and available under the MIT License.


---

### Step 3: Save and Check
1. Click **"Commit changes..."** $\rightarrow$ **"Commit changes"**.
2. Click the repository name **`3d-brain-mri-monai`** in the top left breadcrumbs or go to:  
   👉 **[https://github.com/cyberpunk92/3d-brain-mri-monai](https://github.com/cyberpunk92/3d-brain-mri-monai)**
3. Scroll down on the main repo page: it will render the formatted Markdown layout with headers, lists, code boxes, and your 3-axis brain image.
3d-brain-mri-monai/
├── main.py                          # Full 3D generation, transform, and inference pipeline
├── requirements.txt                 # Project dependencies
├── sample_3d_mpr_visualization.png  # Multi-planar visualization output
├── .gitignore                       # Ignored cache and environment directories
└── README.md                        # Project documentation
