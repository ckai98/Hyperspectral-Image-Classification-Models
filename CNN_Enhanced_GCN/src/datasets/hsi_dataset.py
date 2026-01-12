import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from .superpixel_utils import perform_slic_segmentation, build_association_matrix, build_superpixel_adjacency

class HSIDataset(Dataset):
    def __init__(self, data_path, gt_path, patch_size=9, n_components=30, n_superpixels=1000, is_train=True):
        """
        Args:
            data_path: Path to .mat file containing HSI data (key 'indian_pines_corrected' etc)
            gt_path: Path to .mat file containing GT labels
            patch_size: Size of the spatial patch (odd number)
            n_components: PCA components to keep. If 0, use original channels.
            n_superpixels: Number of superpixels for SLIC
            is_train: Boolean, currently not used for splitting (splitting handled externally or via indices)
        """
        self.patch_size = patch_size
        self.pad_width = patch_size // 2
        
        # Load Data
        self.data, self.gt = self._load_data(data_path, gt_path)
        
        # Preprocessing: Normalize -> PCA
        self.data = self._preprocess(self.data, n_components)
        
        # Superpixel Segmentation
        # Use first 3 PCs for SLIC as is common practice, or all if n_components is small
        print(f"Executing SLIC with {n_superpixels} segments...")
        slic_input = self.data if self.data.shape[2] <= 3 else self.data[:, :, :3]
        self.segments, self.n_segments = perform_slic_segmentation(slic_input, n_segments=n_superpixels)
        
        # Build Graphs
        print("Building Association Matrix Q...")
        self.Q = build_association_matrix(self.segments)
        
        print("Building Adjacency Matrix A...")
        self.A = build_superpixel_adjacency(self.segments)
        
        # Prepare Indices (Exclude background 0)
        # Assuming 0 is background in GT
        self.indices = []
        H, W = self.gt.shape
        for r in range(H):
            for c in range(W):
                if self.gt[r, c] != 0:
                    self.indices.append((r, c))
                    
        # Pad Image for Patch Extraction
        self.data_padded = np.pad(self.data, ((self.pad_width, self.pad_width), 
                                              (self.pad_width, self.pad_width), 
                                              (0, 0)), mode='reflect')
        
        print(f"Dataset Initialized. Samples: {len(self.indices)}. Segments: {self.n_segments}")

    def _load_data(self, data_path, gt_path):
        # Generic loader assuming standard keys usually found in public datasets
        data_mat = sio.loadmat(data_path)
        gt_mat = sio.loadmat(gt_path)
        
        # Find key for data (usually ends with _corrected or just the name)
        data_key = [k for k in data_mat.keys() if not k.startswith('__')][0]
        data = data_mat[data_key]
        
        # Find key for gt
        gt_key = [k for k in gt_mat.keys() if not k.startswith('__')][0]
        gt = gt_mat[gt_key]
        
        return data, gt

    def _preprocess(self, data, n_components):
        H, W, C = data.shape
        orig_data = data.reshape(-1, C)
        
        # Normalize first
        scaler = StandardScaler()
        orig_data = scaler.fit_transform(orig_data)
        
        if n_components is not None and n_components > 0 and n_components < C:
            pca = PCA(n_components=n_components)
            data_pca = pca.fit_transform(orig_data)
            data_pca = data_pca.reshape(H, W, n_components)
            print(f"PCA performed: {C} -> {n_components}")
            return data_pca
        else:
            return orig_data.reshape(H, W, C)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            patch: (C, H, W) tensor
            label: int
            pixel_idx: int (flattened index matching Q rows)
        """
        r, c = self.indices[idx]
        label = self.gt[r, c] - 1 # 0-indexed labels
        
        # Extract Patch
        # Pad width is already added to data_padded
        # coord (r, c) in original maps to (r+pad, c+pad) in padded
        # slice: r+pad - pad : r+pad + pad + 1  => r : r + 2*pad + 1
        r_pad, c_pad = r + self.pad_width, c + self.pad_width
        patch = self.data_padded[r_pad-self.pad_width : r_pad+self.pad_width+1,
                                 c_pad-self.pad_width : c_pad+self.pad_width+1]
        
        # To Tensor (C, H, W)
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        
        # Pixel Index for Adjacency/Association
        H, W = self.gt.shape
        pixel_idx = r * W + c
        
        return patch, label, pixel_idx
