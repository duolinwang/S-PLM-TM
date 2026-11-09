import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class ImprovedGVPLayer(nn.Module):
    """
    Enhanced GVP-inspired layer with skip connections and normalization
    """
    def __init__(self, node_dim, edge_dim=2, dropout=0.1,heads = 4):
        super(ImprovedGVPLayer, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.heads = heads
        
        # GCN for sequence-adjacent connections
        self.seq_conv = GCNConv(node_dim, node_dim)
        
        # GAT for embedding-similarity connections
        self.sim_conv = GATConv(node_dim, node_dim, heads=self.heads, concat=False)
        
        # MLP to combine both types of messages
        self.combine = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim)
        )
        
        # Layer normalization for stabilizing training
        self.layer_norm1 = nn.LayerNorm(node_dim)
        self.layer_norm2 = nn.LayerNorm(node_dim)
        self.layer_norm_final = nn.LayerNorm(node_dim)
    
    def forward(self, x, seq_edge_index, sim_edge_index, edge_attr=None):
        # Process sequence-adjacent connections with first skip connection
        seq_out = self.seq_conv(x, seq_edge_index)
        seq_out = F.relu(seq_out)
        seq_out = self.layer_norm1(seq_out + x)  # Skip connection + normalization
        
        # Process embedding-similarity connections with second skip connection
        sim_out = self.sim_conv(seq_out, sim_edge_index)  # Building on the seq_out
        sim_out = F.relu(sim_out)
        sim_out = self.layer_norm2(sim_out + seq_out)  # Skip connection + normalization
        
        # Combine the messages
        combined = torch.cat([seq_out, sim_out], dim=-1)
        combined_out = self.combine(combined)
        
        # Final skip connection to preserve original features
        out = self.layer_norm_final(x + combined_out)  # Additional skip from input
        
        return out

class ProteinGNN(nn.Module):
    """
    Improved GNN model for protein embeddings from ESM2
    """
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=1280, n_layers=2, 
                 k_neighbors=5, seq_neighborhood=2, dropout=0.1,heads=4):
        super(ProteinGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k_neighbors = k_neighbors
        self.seq_neighborhood = seq_neighborhood
        self.dropout = dropout
        self.heads = heads
        
        # Input projection and normalization
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # GVP layers with interspersed skip connections
        self.gvp_layers = nn.ModuleList([
            ImprovedGVPLayer(hidden_dim, dropout=dropout,heads=self.heads) for _ in range(n_layers)
        ])
        
        # Global feature extraction from original ESM embeddings
        self.global_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Node importance prediction for attention pooling
        self.node_importance = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 4),  # Include global features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def _build_edges(self, embeddings, mask):
        """
        Build both sequence-adjacent and embedding-similarity edges with proper indexing
        
        Args:
            embeddings: Tensor of shape (batch_size, max_len, hidden_dim)
            mask: Boolean mask of shape (batch_size, max_len) where True indicates valid positions
                
        Returns:
            tuple of sequence edges, similarity edges, and batch indices
        """
        batch_size, max_len, _ = embeddings.shape
        device = embeddings.device
        
        # Lists to store edges and batch indices
        seq_edge_list = []
        sim_edge_list = []
        batch_idx = []
        
        # Map from original position (batch_idx, seq_pos) to new global node index
        node_mapping = {}
        global_node_idx = 0
        
        # First pass: create node mapping
        for b in range(batch_size):
            # Get valid positions in this sequence
            valid_pos = torch.where(mask[b])[0]
            seq_len = len(valid_pos)
            
            if seq_len == 0:
                continue
                
            # Map each valid position to a global node index
            for pos in valid_pos:
                node_mapping[(b, pos.item())] = global_node_idx
                global_node_idx += 1
                
            # Add batch indices for each node
            batch_idx.extend([b] * seq_len)
        
        # Total number of nodes in the graph
        total_nodes = global_node_idx
        
        # Second pass: create edges using the node mapping
        for b in range(batch_size):
            valid_pos = torch.where(mask[b])[0]
            seq_len = len(valid_pos)
            
            if seq_len == 0:
                continue
            
            # Create sequence-adjacent edges
            for i in range(seq_len):
                curr_pos = valid_pos[i].item()
                curr_idx = node_mapping[(b, curr_pos)]
                
                # Connect to neighbors within sequence neighborhood
                for j in range(max(0, i-self.seq_neighborhood), min(seq_len, i+self.seq_neighborhood+1)):
                    if i == j:
                        continue  # Skip self-connections
                    
                    neighbor_pos = valid_pos[j].item()
                    neighbor_idx = node_mapping[(b, neighbor_pos)]
                    
                    # Add bidirectional edges
                    seq_edge_list.append([curr_idx, neighbor_idx])
            
            # Create embedding-similarity edges
            if seq_len > 1:  # Only if we have at least 2 nodes
                # Extract valid embeddings for this sequence
                valid_embeddings = embeddings[b, mask[b]]
                
                # Calculate pairwise cosine similarity
                norm_embeds = F.normalize(valid_embeddings, p=2, dim=1)
                sim_matrix = torch.mm(norm_embeds, norm_embeds.transpose(0, 1))
                
                # Zero out self-connections
                sim_matrix.fill_diagonal_(-float('inf'))
                
                # Get top-k neighbors for each node
                k = min(self.k_neighbors, seq_len-1)
                if k > 0:
                    _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
                    
                    for i, neighbors in enumerate(topk_indices):
                        src_pos = valid_pos[i].item()
                        src_idx = node_mapping[(b, src_pos)]
                        
                        for j in neighbors:
                            tgt_pos = valid_pos[j.item()].item()
                            tgt_idx = node_mapping[(b, tgt_pos)]
                            sim_edge_list.append([src_idx, tgt_idx])
        
        # Convert to tensors
        if len(seq_edge_list) > 0:
            seq_edges = torch.tensor(seq_edge_list, dtype=torch.long, device=device).t()
        else:
            seq_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
            
        if len(sim_edge_list) > 0:
            sim_edges = torch.tensor(sim_edge_list, dtype=torch.long, device=device).t()
        else:
            sim_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
        
        return seq_edges, sim_edges, batch_idx, total_nodes
    
    def forward(self, embeddings, mask=None):
        """
        Forward pass
        
        Args:
            embeddings: ESM2 embeddings of shape (batch_size, max_len, input_dim)
            mask: Boolean mask of shape (batch_size, max_len) where True indicates valid positions
                  If None, assume all positions are valid
        
        Returns:
            Protein-level embeddings of shape (batch_size, output_dim)
        """
        batch_size, max_len, _ = embeddings.shape
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=embeddings.device)
        
        # Pre-normalize the input ESM2 embeddings - important for stable training
        normalized_embeddings = F.normalize(embeddings, p=2, dim=2)
        
        # Project input if needed
        x = self.input_proj(normalized_embeddings)
        x = self.input_norm(x)  # Apply layer norm to stabilize
        
        # Build edges
        seq_edges, sim_edges, batch_idx, total_nodes = self._build_edges(x, mask)
        
        # Extract global features from the original embeddings (will be used in attention)
        global_features = []
        for b in range(batch_size):
            valid_indices = mask[b]
            if torch.any(valid_indices):
                # Get the mean of normalized embeddings for global features
                global_features.append(self.global_feature(
                    normalized_embeddings[b, valid_indices].mean(dim=0)
                ))
        
        if not global_features:  # Handle empty case
            return torch.zeros(batch_size, self.output_dim, device=embeddings.device)
            
        global_features = torch.stack(global_features)
        
        # Gather only the node features we need
        node_features = []
        original_features = []  # Store original features for residual connections
        for b in range(batch_size):
            valid_pos = torch.where(mask[b])[0]
            if len(valid_pos) > 0:
                node_features.append(x[b, valid_pos])
                original_features.append(normalized_embeddings[b, valid_pos])
        
        if not node_features:  # Handle empty case
            return torch.zeros(batch_size, self.output_dim, device=embeddings.device)
        
        node_features = torch.cat(node_features, dim=0)
        original_features = torch.cat(original_features, dim=0)
        
        # Keep the input features for the final skip connection
        input_features = node_features
        
        # Apply GVP layers
        layer_outputs = [node_features]  # Store outputs for dense skip connections
        
        for i, layer in enumerate(self.gvp_layers):
            # Add dense skip connections - concatenate all previous outputs
            if i > 0:
                dense_input = sum(layer_outputs) / len(layer_outputs)
                node_features = layer(dense_input, seq_edges, sim_edges)
            else:
                node_features = layer(node_features, seq_edges, sim_edges)
                
            layer_outputs.append(node_features)
        
        # Final skip connection to the input
        node_features = node_features + input_features
        
        # Calculate attention weights for pooling, incorporating global features
        node_importance = []
        for b in range(batch_size):
            b_mask = (batch_idx == b)
            if not torch.any(b_mask):
                continue
                
            b_features = node_features[b_mask]
            
            # Expand global features to match the number of nodes
            b_global = global_features[b].unsqueeze(0).expand(b_features.size(0), -1)
            
            # Concatenate with global features for context-aware importance
            combined_features = torch.cat([b_features, b_global], dim=1)
            importance = self.node_importance(combined_features)
            
            node_importance.append(importance)
        
        if not node_importance:  # Handle empty case
            return torch.zeros(batch_size, self.output_dim, device=embeddings.device)
            
        node_importance = torch.cat(node_importance, dim=0).squeeze(-1)
        
        # Apply softmax per protein to get attention weights
        attention_weights = []
        start_idx = 0
        for b in range(batch_size):
            b_mask = (batch_idx == b)
            if not torch.any(b_mask):
                continue
                
            b_count = torch.sum(b_mask).item()
            b_importance = node_importance[start_idx:start_idx + b_count]
            b_weights = F.softmax(b_importance, dim=0).unsqueeze(1)
            attention_weights.append(b_weights)
            start_idx += b_count
        
        if not attention_weights:  # Handle empty case
            return torch.zeros(batch_size, self.output_dim, device=embeddings.device)
            
        attention_weights = torch.cat(attention_weights, dim=0)
        
        # Weighted pooling per graph/protein
        protein_embeddings = torch.zeros(batch_size, self.hidden_dim, device=embeddings.device)
        
        start_idx = 0
        for b in range(batch_size):
            b_mask = (batch_idx == b)
            if not torch.any(b_mask):
                continue
                
            b_count = torch.sum(b_mask).item()
            b_features = node_features[start_idx:start_idx + b_count]
            b_weights = attention_weights[start_idx:start_idx + b_count]
            
            # Attention-weighted sum
            protein_embeddings[b] = torch.sum(b_features * b_weights, dim=0)
            start_idx += b_count
        
        # Apply output projection and normalization
        protein_embeddings = self.output_proj(protein_embeddings)
        
        # Final normalization to maintain consistent embedding space
        protein_embeddings = F.normalize(protein_embeddings, p=2, dim=1)
        
        return protein_embeddings


# Example usage
def get_protein_embeddings(esm2_embeddings, attention_mask=None, model=None, training=True):
    """
    Get protein-level embeddings from ESM2 amino acid embeddings
    
    Args:
        esm2_embeddings: Tensor of shape (batch_size, max_len, 1280)
        attention_mask: Boolean mask of shape (batch_size, max_len) where True indicates valid positions
                      If None, assume all positions are valid
        model: An existing ProteinGNN model, or None to create a new one
        training: Whether we're in training mode
    
    Returns:
        Protein-level embeddings of shape (batch_size, 1280)
    """
    if model is None:
        model = ProteinGNN(
            input_dim=1280,          # ESM2 embedding dimension
            hidden_dim=1280,         # Keep same dimension internally
            output_dim=1280,         # Output dimension (same as ESM2)
            n_layers=2,              # Reduced number of layers to avoid over-smoothing
            k_neighbors=5,           # Number of similarity neighbors
            seq_neighborhood=2,      # Number of sequence neighbors
            dropout=0.1,              # Dropout for regularization
            heads = 4,
        )
    
    # Move to same device as input
    model = model.to(esm2_embeddings.device)
    
    # Set training mode
    if training:
        model.train()
        protein_embeddings = model(esm2_embeddings, attention_mask)
    else:
        model.eval()
        with torch.no_grad():
            protein_embeddings = model(esm2_embeddings, attention_mask)
    
    return protein_embeddings

"""
B, L, D = 40, 12, 1280
esm2_embeddings = torch.randn(B, L, D)
#mask = torch.constant([0,1,1,1,1,1,0,1,0,1,0,1])
protein_embeddings = get_protein_embeddings(esm2_embeddings, attention_mask=None,training=False)

"""