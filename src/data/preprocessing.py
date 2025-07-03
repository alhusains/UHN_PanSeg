import numpy as np
from typing import Tuple, Optional


def normalize_image(img: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize image using specified method"""
    if method == "zscore":
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            return (img - mean) / std
        return img - mean
    elif method == "minmax":
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def crop_to_roi(img: np.ndarray, mask: Optional[np.ndarray] = None, 
                margin: int = 10) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Crop image to region of interest with margin"""
    if mask is not None:
        # Find bounding box from mask
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            min_x, max_x = coords[0].min(), coords[0].max()
            min_y, max_y = coords[1].min(), coords[1].max()
            min_z, max_z = coords[2].min(), coords[2].max()
            # Add margin
            min_x = max(0, min_x - margin)
            max_x = min(img.shape[0], max_x + margin)
            min_y = max(0, min_y - margin)
            max_y = min(img.shape[1], max_y + margin)
            min_z = max(0, min_z - margin)
            max_z = min(img.shape[2], max_z + margin)
            
            img_cropped = img[min_x:max_x, min_y:max_y, min_z:max_z]
            mask_cropped = mask[min_x:max_x, min_y:max_y, min_z:max_z]
            return img_cropped, mask_cropped
    
    return img, mask


def resize_volume(img: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Simple resize using nearest neighbor interpolation"""
    from scipy.ndimage import zoom
    
    zoom_factors = [t / s for t, s in zip(target_shape, img.shape)]
    return zoom(img, zoom_factors, order=0)


def prepare_segmentation_target(mask: np.ndarray) -> np.ndarray:
    """Prepare mask for segmentation training (ensure 3 classes: 0, 1, 2)"""
    # Ensure mask has values 0, 1, 2
    mask_clean = np.zeros_like(mask, dtype=np.uint8)
    mask_clean[mask == 1] = 1  # Normal pancreas
    mask_clean[mask == 2] = 2  # Lesion
    return mask_clean 