import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional


def plot_case_overview(img: np.ndarray, mask: Optional[np.ndarray] = None, 
                      title: str = "Case Overview", save_path: Optional[str] = None):
    """Plot middle slices of image and mask"""
    fig, axes = plt.subplots(1, 3 if mask is not None else 1, figsize=(15, 5))
    if mask is None:
        axes =[axes]
    
    # Plot middle slice of image
    mid_slice = img.shape[2] // 2
    axes[0].imshow(img[:, :, mid_slice], cmap='gray')
    axes[0].set_title(f'Image (slice {mid_slice})')
    axes[0].axis('off')
    
    if mask is not None:
        # Plot mask
        axes[1].imshow(mask[:, :, mid_slice], cmap='tab10')
        axes[1].set_title(f'Mask (slice {mid_slice})')
        axes[1].axis('off')
        # Plot overlay
        axes[2].imshow(img[:, :, mid_slice], cmap='gray', alpha=0.7)
        mask_slice = mask[:, :, mid_slice]
        mask_slice = np.ma.masked_where(mask_slice == 0, mask_slice)
        axes[2].imshow(mask_slice, cmap='tab10', alpha=0.8)
        axes[2].set_title(f'Overlay (slice {mid_slice})')
        axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_dataset_distribution(stats: dict, save_path: Optional[str] = None):
    """Plot dataset distribution by subtype"""
    splits = list(stats.keys())
    subtypes =["subtype0", "subtype1", "subtype2"]
    
    fig, axes = plt.subplots(1, len(splits), figsize=(12, 5))
    if len(splits) == 1:
        axes =[axes]
    
    for i, split in enumerate(splits):
        counts = [stats[split][subtype] for subtype in subtypes]
        axes[i].bar(subtypes, counts, color=['red', 'green', 'blue'])
        axes[i].set_title(f'{split.capitalize()} Set')
        axes[i].set_ylabel('Number of Cases')
        #Add count labels on bars
        for j, count in enumerate(counts):
            axes[i].text(j, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_intensity_distribution(img: np.ndarray, mask: Optional[np.ndarray] = None, 
                              title: str = "Intensity Distribution", save_path: Optional[str] = None):
    """Plot intensity distribution of image and mask regions"""
    plt.figure(figsize=(12, 4))
    # Overall image histogram
    plt.subplot(1, 2, 1)
    plt.hist(img.flatten(), bins=100, alpha=0.7, label='All voxels')
    if mask is not None:
        pancreas_voxels = img[mask > 0]
        if len(pancreas_voxels) > 0:
            plt.hist(pancreas_voxels, bins=50, alpha=0.7, label='Pancreas region')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Image Intensity Distribution')
    plt.legend()
    
    # Mask label distribution
    if mask is not None:
        plt.subplot(1, 2, 2)
        unique, counts = np.unique(mask, return_counts=True)
        plt.bar(unique, counts, color=['black', 'red', 'green'])
        plt.xlabel('Label')
        plt.ylabel('Voxel Count')
        plt.title('Mask Label Distribution')
        plt.xticks(unique, ['Background', 'Pancreas', 'Lesion'])
    
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show() 