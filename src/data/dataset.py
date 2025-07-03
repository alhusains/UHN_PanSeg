import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PancreasDataset:
    """Dataset loader for UHN pancreas CT scans"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.train_path = self.data_root / "train"
        self.val_path = self.data_root / "validation"
        self.test_path = self.data_root / "test"
        
    def get_file_list(self, split: str = "train") -> Dict[str, List[str]]:
        """Get list of files organized by subtype, only if both image and mask exist"""
        if split == "train":
            base_path = self.train_path
        elif split == "validation":
            base_path = self.val_path
        else:
            raise ValueError(f"Unknown split: {split}")
            
        files = {}
        for subtype in ["subtype0", "subtype1", "subtype2"]:
            subtype_path = base_path / subtype
            if subtype_path.exists():
                mask_files = []
                for f in os.listdir(subtype_path):
                    if f.endswith('.nii.gz') and not f.endswith('_0000.nii.gz'):
                        case_id = f.replace('.nii.gz', '')
                        img_file = subtype_path / f"{case_id}_0000.nii.gz"
                        mask_file = subtype_path / f
                        if img_file.exists() and mask_file.exists():
                            mask_files.append(f)
                files[subtype] = mask_files
        return files
    
    def load_case(self, split: str, subtype: str, case_id: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load image and mask for a specific case"""
        if split in ["train", "validation"]:
            base_path = self.data_root / split / subtype
            img_path = base_path / f"{case_id}_0000.nii.gz"
            mask_path = base_path / f"{case_id}.nii.gz"
            
            img = nib.load(img_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            return img, mask
        else:
            # Test set only has images
            img_path = self.data_root / "test" / f"{case_id}_0000.nii.gz"
            img = nib.load(img_path).get_fdata()
            return img, None
    
    def get_dataset_stats(self) -> Dict:
        """Get basic dataset statistics"""
        stats = {}
        
        for split in ["train", "validation"]:
            files = self.get_file_list(split)
            split_stats = {}
            
            for subtype, file_list in files.items():
                split_stats[subtype] = len(file_list)
                
                # Sample a few cases for size analysis
                if file_list:
                    sample_case = file_list[0].replace('.nii.gz', '')
                    img, mask = self.load_case(split, subtype, sample_case)
                    split_stats[f"{subtype}_shape"] = img.shape
                    if mask is not None:
                        split_stats[f"{subtype}_unique_labels"] = np.unique(mask)
                        
            stats[split] = split_stats
            
        return stats


def create_subtype_mapping() -> Dict[str, int]:
    """Create mapping from subtype folder names to class indices"""
    return {"subtype0": 0, "subtype1": 1, "subtype2": 2} 