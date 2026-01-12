import numpy as np
import scipy.sparse as sp
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import torch

def perform_slic_segmentation(hsi_img, n_segments=1000, compactness=10.0, sigma=0, seed=None):
    """
    Perform SLIC segmentation on HSI image.
    
    Args:
        hsi_img (np.ndarray): Input HSI image (H, W, C).
        n_segments (int): Approximate number of superpixels.
        compactness (float): Balances color proximity and space proximity.
        sigma (float): Width of Gaussian smoothing kernel.
    
    Returns:
        segments (np.ndarray): Superpixel label map (H, W), values in [0, n_actual-1].
        n_actual (int): Actual number of segments produced.
    """
    # SLIC in skimage expects (H, W, C) or (H, W)
    # For HSI with many channels, it computes distance in C-dim space + XY space.
    # Usually PCA is performed before SLIC to reduce dimensionality and noise, 
    # but the paper says "original HSI" or "first three PCs".
    # We will use the raw image or simple mean if C is too large, but typically valid to use all.
    
    # Using PCA for SLIC is a common trick to speed up and denote structure better.
    # We will assume input is normalized feature.
    
    segments = slic(hsi_img, n_segments=n_segments, compactness=compactness, sigma=sigma, 
                    start_label=0, enforce_connectivity=True)
    
    # Ensure labels are consecutive [0, N-1]
    n_actual = segments.max() + 1
    return segments, n_actual

def build_association_matrix(segments):
    """
    Build Pixel-to-Superpixel association matrix Q.
    
    Args:
        segments (np.ndarray): Superpixel label map (H, W).
    
    Returns:
        Q (sp.csr_matrix): Shape (N_pixels, N_segments). Q[i, j] = 1 if pixel i belongs to seg j.
    """
    H, W = segments.shape
    N_pixels = H * W
    N_segments = segments.max() + 1
    
    # Flatten segments to get segment assignment for each pixel
    segment_ids = segments.reshape(-1)
    pixel_ids = np.arange(N_pixels)
    
    # Create sparse matrix
    # Data is all 1s
    data = np.ones(N_pixels)
    Q = sp.coo_matrix((data, (pixel_ids, segment_ids)), shape=(N_pixels, N_segments))
    
    return Q.tocsr() # Column-oriented might be better if we access by segment, but CSR is standard.

def build_superpixel_adjacency(segments):
    """
    Build Superpixel Adjacency Matrix A.
    Two superpixels are connected if they share a boundary in the spatial map.
    
    Args:
        segments (np.ndarray): Superpixel label map (H, W).
        
    Returns:
        A (sp.csr_matrix): Adjacency matrix (N_segments, N_segments). Binary.
    """
    H, W = segments.shape
    N_segments = segments.max() + 1
    
    # Use a set of edges to avoid duplicates
    edges = set()
    
    # Check horizontal neighbors
    # segments[:, :-1] vs segments[:, 1:]
    # If different, they are neighbors
    mask_h = segments[:, :-1] != segments[:, 1:]
    src_h = segments[:, :-1][mask_h]
    dst_h = segments[:, 1:][mask_h]
    for s, d in zip(src_h, dst_h):
        if s != d: # Should be true by mask, but just in case
            edges.add(tuple(sorted((s, d))))

    # Check vertical neighbors
    mask_v = segments[:-1, :] != segments[1:, :]
    src_v = segments[:-1, :][mask_v]
    dst_v = segments[1:, :][mask_v]
    for s, d in zip(src_v, dst_v):
        if s != d:
            edges.add(tuple(sorted((s, d))))
            
    # Build matrix
    row_ind = []
    col_ind = []
    for s, d in edges:
        row_ind.extend([s, d])
        col_ind.extend([d, s])
        
    data = np.ones(len(row_ind))
    A = sp.coo_matrix((data, (row_ind, col_ind)), shape=(N_segments, N_segments))
    
    # Add self-loops? usually GCN adds I. We return pure adjacency A here.
    return A.tocsr()

def compute_superpixel_features(Q, pixel_features):
    """
    Aggregate pixel features to superpixels (Average Pooling).
    H_super = D_super^{-1} * Q^T * H_pixel
    
    Args:
        Q (sp.csr_matrix): Association matrix (N_pix, N_seg).
        pixel_features (np.ndarray or torch.Tensor): (N_pix, C).
        
    Returns:
        superpixel_features (np.ndarray): (N_seg, C).
    """
    # Count pixels per superpixel
    segment_counts = np.array(Q.sum(axis=0)).flatten() # (N_seg,)
    segment_counts[segment_counts == 0] = 1 # Avoid div by zero
    
    # Q.T * X
    if isinstance(pixel_features, torch.Tensor):
        pixel_features = pixel_features.cpu().numpy()
        
    pooled = Q.T.dot(pixel_features)
    
    # Average
    averaged = pooled / segment_counts[:, None]
    
    return averaged
