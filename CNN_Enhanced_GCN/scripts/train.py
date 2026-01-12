import sys
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.hsi_dataset import HSIDataset
from src.models.cegcn import CEGCN
from src.utils import sparse_mx_to_torch_sparse_tensor, normalize_adj

def train(config_path):
    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set Seeds
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load Dataset
    print("Loading Dataset...")
    dataset = HSIDataset(
        data_path=config['dataset']['data_path'],
        gt_path=config['dataset']['gt_path'],
        patch_size=config['dataset']['patch_size'],
        n_components=config['dataset']['n_components'],
        n_superpixels=config['dataset']['n_superpixels']
    )
    
    # Prepare Graph Structures
    print("Preparing Graph Structures...")
    # Adjacency
    adj = dataset.A
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0])) # Add self-loop and normalize
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
    
    # Node Features (Mean Spectral)
    # Use unpadded data PCA'd. shape (H, W, C).
    # Segments shape (H, W).
    # We need to compute mean feature for each segment 0..M-1
    H, W, C = dataset.data.shape
    flat_data = dataset.data.reshape(-1, C) # (N, C)
    flat_segments = dataset.segments.reshape(-1) # (N,)
    
    n_segments = dataset.n_segments
    node_features = np.zeros((n_segments, C))
    cnts = np.zeros(n_segments)
    
    # Slow Loop? Can be optimized
    # For now, quick loop
    for i in range(H*W):
        seg_id = flat_segments[i]
        node_features[seg_id] += flat_data[i]
        cnts[seg_id] += 1
        
    cnts[cnts==0] = 1
    node_features = node_features / cnts[:, None]
    node_features_tensor = torch.from_numpy(node_features).float().to(device)
    
    # Pixel to Superpixel Mapping
    # Create a tensor for fast lookup
    # Need to map dataset.indices (which are valid pixels) to superpixels
    # But dataset returns 'pixel_idx' = r*W + c.
    # So we can just use flat_segments[pixel_idx]
    pixel_to_superpixel = torch.from_numpy(flat_segments).long().to(device)
    
    # Split Data (Per Class Sampling)
    print("Splitting Data...")
    gt = dataset.gt
    indices = dataset.indices
    n_samples_per_class = config['training']['samples_per_class']
    
    train_indices = []
    test_indices = []
    
    classes = np.unique(gt)
    classes = classes[classes != 0] # Remove background
    
    for c in classes:
        # Find all indices belonging to class c
        c_indices = [i for i, (r, col) in enumerate(indices) if gt[r, col] == c]
        if len(c_indices) > n_samples_per_class:
            c_train = np.random.choice(c_indices, n_samples_per_class, replace=False)
            c_test = list(set(c_indices) - set(c_train))
        else:
            c_train = c_indices # Take all if not enough
            c_test = []
            
        train_indices.extend(c_train)
        test_indices.extend(c_test)
        
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    train_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], 
                              sampler=SubsetRandomSampler(train_indices))
    test_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], 
                             sampler=SubsetRandomSampler(test_indices))
    
    # Init Model
    model = CEGCN(
        height=config['dataset']['patch_size'], 
        width=config['dataset']['patch_size'], 
        bands=config['dataset']['n_components'], 
        num_classes=len(classes),
        hidden_dim=config['model']['hidden_dim']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['training']['learning_rate'], 
                           weight_decay=config['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print("Starting Training...")
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (patches, labels, pixel_idxs) in enumerate(train_loader):
            patches, labels = patches.to(device), labels.to(device)
            pixel_idxs = pixel_idxs.to(device)
            
            # Lookup superpixel IDs
            sp_ids = pixel_to_superpixel[pixel_idxs]
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(patches, sp_ids, adj_tensor, node_features_tensor)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%")
            
    # Evaluation
    print("Evaluating...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for patches, labels, pixel_idxs in test_loader:
            patches, labels = patches.to(device), labels.to(device)
            pixel_idxs = pixel_idxs.to(device)
            sp_ids = pixel_to_superpixel[pixel_idxs]
            
            outputs = model(patches, sp_ids, adj_tensor, node_features_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # Metrics
    oa = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    conf_mat = confusion_matrix(all_targets, all_preds)
    
    # Per Class Acc
    per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    aa = np.mean(per_class_acc)
    
    print("\nResults:")
    print(f"OA: {oa*100:.2f}%")
    print(f"AA: {aa*100:.2f}%")
    print(f"Kappa: {kappa:.4f}")
    
    # Save Results locally if needed (omitted for brevity)

if __name__ == "__main__":
    import scipy.sparse as sp # Need for train func
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cegcn.yaml', help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
