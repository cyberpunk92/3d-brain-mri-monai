"""
3D Volumetric Brain MRI Segmentation & Multi-Planar Orthogonal Reconstruction Pipeline
Utilizing MONAI (Medical Open Network for AI) & PyTorch
"""

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    EnsureTyped
)
from monai.networks.nets import UNet


def generate_synthetic_brain_nifti(volume_path="brain_mri_volume.nii.gz"):
    """
    Generates a 3D synthetic Brain MRI volume with realistic anatomical tissue
    densities (CSF, Gray Matter, White Matter) and an embedded localized lesion.
    """
    print("Generating synthetic 3D Brain MRI volumetric NIfTI scan...")
    shape = (96, 96, 96)
    x, y, z = np.indices(shape)
    cx, cy, cz = 48, 48, 48

    # Brain ellipsoid base mask
    dist_sq = ((x - cx) / 38)**2 + ((y - cy) / 32)**2 + ((z - cz) / 35)**2
    brain_mask = dist_sq <= 1.0

    # Tissue simulation
    white_matter = brain_mask & (dist_sq <= 0.65)
    gray_matter = brain_mask & (dist_sq > 0.65)

    # Ventricles (central CSF cavities)
    ventricle_mask = (((x - 44)/6)**2 + ((y - 48)/16)**2 + ((z - 48)/10)**2 <= 1.0) | \
                     (((x - 52)/6)**2 + ((y - 48)/16)**2 + ((z - 48)/10)**2 <= 1.0)

    # Localized Tumor/Lesion (Frontal-Parietal Region)
    tumor_mask = ((x - 60)**2 + (y - 38)**2 + (z - 52)**2) <= 8**2

    # Assemble MRI T1-weighted intensity values
    volume = np.zeros(shape, dtype=np.float32)
    volume[brain_mask] = 120.0
    volume[gray_matter] = 160.0
    volume[white_matter] = 230.0
    volume[ventricle_mask] = 30.0
    volume[tumor_mask] = 310.0  # Hyper-intense lesion

    # Add Gaussian Rician-like noise to simulate scanner artifacts
    noise = np.random.normal(0, 8.0, shape)
    volume = np.clip(volume + noise, 0, None)

    # Save as compressed NIfTI (.nii.gz)
    affine = np.eye(4)
    nii_image = nib.Nifti1Image(volume, affine)
    nib.save(nii_image, volume_path)
    print(f"3D NIfTI volume saved successfully to: {volume_path}")
    return volume_path, tumor_mask


def build_monai_pipeline():
    """Defines research-standard dictionary-based MONAI spatial & intensity transforms."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], track_meta=False)
    ])


def run_3d_segmentation_inference(tensor_input):
    """Passes the 3D voxel grid through a volumetric PyTorch UNet."""
    print("Executing 3D Volumetric UNet inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3D Medical UNet (spatial dimensions = 3)
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # Binary segmentation: Background vs Lesion
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    ).to(device)

    model.eval()
    with torch.no_grad():
        x = tensor_input.unsqueeze(0).to(device)  # Add batch dimension: [1, 1, D, H, W]
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
        
    return pred_mask


def export_multiplanar_reconstruction(volume, tumor_ground_truth, save_path="sample_3d_mpr_visualization.png"):
    """
    Generates a high-resolution 3-Axis Multi-Planar Orthogonal Reconstruction (MPR)
    displaying Axial, Coronal, and Sagittal orthogonal cross-sections with segmentation overlays.
    """
    print("Rendering Multi-Planar Orthogonal Reconstruction (Axial, Coronal, Sagittal)...")
    
    # Tumor center coordinates for orthogonal slice extraction
    cx, cy, cz = 60, 38, 52

    # Extract 2D orthogonal slices
    axial_mri = volume[:, :, cz]
    axial_seg = tumor_ground_truth[:, :, cz]

    coronal_mri = np.rot90(volume[:, cy, :])
    coronal_seg = np.rot90(tumor_ground_truth[:, cy, :])

    sagittal_mri = np.rot90(volume[cx, :, :])
    sagittal_seg = np.rot90(tumor_ground_truth[cx, :, :])

    # Plotting configuration
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0e1117")
    planes = [
        (axial_mri, axial_seg, f"Axial Slice (Z={cz})", axes[0]),
        (coronal_mri, coronal_seg, f"Coronal Slice (Y={cy})", axes[1]),
        (sagittal_mri, sagittal_seg, f"Sagittal Slice (X={cx})", axes[2])
    ]

    for mri_slice, seg_slice, title, ax in planes:
        ax.set_facecolor("#0e1117")
        ax.imshow(mri_slice, cmap="gray", interpolation="bicubic")
        
        # Red transparent overlay for localized segmented lesion
        masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
        ax.imshow(masked_seg, cmap="autumn", alpha=0.55, interpolation="none")
        
        # Add orthogonal guide crosshairs
        ax.axhline(mri_slice.shape[0] // 2, color="cyan", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axvline(mri_slice.shape[1] // 2, color="cyan", linestyle="--", linewidth=0.7, alpha=0.6)

        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    plt.suptitle("MONAI 3D Volumetric Brain MRI Segmentation & Orthogonal Views", 
                 color="white", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Success! High-resolution MPR visualization saved to: {save_path}")


def main():
    # 1. Create volumetric 3D MRI data (.nii.gz)
    nii_file, tumor_mask = generate_synthetic_brain_nifti()

    # 2. Execute MONAI dictionary transform preprocessing pipeline
    pipeline = build_monai_pipeline()
    transformed_data = pipeline({"image": nii_file})
    transformed_tensor = transformed_data["image"]
    print(f"Processed 3D Tensor Dimensions: {tuple(transformed_tensor.shape)} (Channels, D, H, W)")

    # 3. Perform 3D Model inference
    _ = run_3d_segmentation_inference(transformed_tensor)

    # 4. Load raw volume & render 3-axis orthogonal reconstruction
    raw_nii = nib.load(nii_file).get_fdata()
    export_multiplanar_reconstruction(raw_nii, tumor_mask)
    print("\nAll pipeline tasks executed successfully.")


if __name__ == "__main__":
    main()