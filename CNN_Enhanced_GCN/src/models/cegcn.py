import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution

class CEGCN(nn.Module):
    def __init__(self, height, width, bands, num_classes, hidden_dim=128):
        super(CEGCN, self).__init__()
        
        # 1. CNN Branch (Pixel-level)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Assuming patch size doesn't change through layers (padding=1, stride=1)
        self.cnn_flat_dim = 64 * height * width
        self.cnn_projector = nn.Linear(self.cnn_flat_dim, hidden_dim)
        
        # 2. GCN Branch (Superpixel-level)
        # Input: Node features (can be spectral or aggregated CNN features)
        # We assume input dim = bands (spectral) or hidden_dim (if pre-projected)
        # Let's align it to 'bands' for maximum flexibility if we feed spectral means
        self.gcn1 = GraphConvolution(bands, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        
        # 3. Fusion & Classifier
        # Concatenate: 128 (CNN) + 128 (GCN) = 256
        self.fusion_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x_patch, pixel_superpixel_ids, adj_matrix, node_features):
        """
        Args:
            x_patch: (B, C, H, W) - Pixel patches
            pixel_superpixel_ids: (B,) - The Superpixel ID each pixel belongs to
            adj_matrix: (N_nodes, N_nodes) - Graph Adjacency
            node_features: (N_nodes, C) - Input features for the graph (e.g. Mean Spectral)
            
        Returns:
            logits: (B, num_classes)
        """
        # --- CNN Path ---
        cnn_feat = self.cnn_encoder(x_patch)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        cnn_feat = self.cnn_projector(cnn_feat) # (B, 128)
        
        # --- GCN Path ---
        # Perform graph convolution on the WHOLE graph
        # In Transductive learning, the graph is small (~1000 nodes), so this is cheap.
        gcn_h1 = F.relu(self.gcn1(node_features, adj_matrix))
        gcn_h2 = self.gcn2(gcn_h1, adj_matrix) # (N_segments, 128)
        
        # --- Fusion (Decoder) ---
        # Gather GCN features for the current batch's pixels
        # pixel_superpixel_ids shape (B,), values in [0, N_segments-1]
        context_feat = gcn_h2[pixel_superpixel_ids] # (B, 128)
        
        # Concatenate
        combined = torch.cat([cnn_feat, context_feat], dim=1) # (B, 256)
        
        # --- Classification ---
        logits = self.classifier(combined)
        
        return logits
