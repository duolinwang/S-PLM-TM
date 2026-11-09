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
            nn.Linear(node_dim, node_dim),
            #nn.Tanh()  #newly added Tanh to keep values bounded but allow negative values #remove to keep gnnv2
        )
        
        # Layer normalization for stabilizing training 
        #self.layer_norm1 = nn.LayerNorm(node_dim) removed 
        #self.layer_norm2 = nn.LayerNorm(node_dim) removed 
        #Preserves Natural Variance: LayerNorm forces features to have zero mean and unit variance, which can make embeddings artificially similar
        #Maintains Feature Magnitudes: Different proteins might naturally have different "scales" of features that LayerNorm erases
        #Less Aggressive Normalization: You'd still keep the final L2 normalization for consistent embedding space, but remove the intermediate ones that might be causing over-smoothing
        #self.layer_norm_final = nn.LayerNorm(node_dim) removed
    
    def forward(self, x, seq_edge_index, sim_edge_index, edge_attr=None):
        # Process sequence-adjacent connections with first skip connection
        seq_out = self.seq_conv(x, seq_edge_index)
        seq_out = F.relu(seq_out)
        seq_out = seq_out + x  #add on 6/20/2025
        #seq_out = self.layer_norm1(seq_out + x)  # Skip connection + normalization
        
        # Process embedding-similarity connections with second skip connection
        sim_out = self.sim_conv(seq_out, sim_edge_index)  # Building on the seq_out
        sim_out = F.relu(sim_out)
        #sim_out = self.layer_norm2(sim_out + seq_out)  # Skip connection + normalization
        sim_out = sim_out + seq_out #keep the skip connection remove normalization
        # Combine the messages
        combined = torch.cat([seq_out, sim_out], dim=-1)
        combined_out = self.combine(combined)
        
        # Final skip connection to preserve original features
        #out = self.layer_norm_final(x + combined_out)  # Additional skip from input
        out = x + combined_out  # Additional skip from input
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
        Build sequence-adjacent and embedding-similarity edges.

        Vectorized rewrite: replaces O(B*L^2) Python loops + dict with
        O(B) Python iterations whose inner work is pure PyTorch, eliminating
        the CPU bottleneck from the original implementation.

        Args:
            embeddings: (batch_size, max_len, hidden_dim) — already projected/normed
            mask:       (batch_size, max_len) bool, True = valid residue

        Returns:
            seq_edges:   (2, E_seq)  long tensor
            sim_edges:   (2, E_sim)  long tensor
            batch_idx:   (total_nodes,) long tensor
            total_nodes: int
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # --- Node counts and global offsets (fully vectorized) ---
        valid_counts = mask.sum(dim=1)                          # (B,)
        offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            offsets[1:] = valid_counts[:-1].cumsum(0)
        total_nodes = valid_counts.sum().item()

        # batch_idx: which batch each global node belongs to
        batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=device), valid_counts
        )  # (total_nodes,)

        seq_edge_parts = []
        sim_edge_parts = []

        for b in range(batch_size):
            n = valid_counts[b].item()
            if n == 0:
                continue
            offset = offsets[b]                                 # scalar tensor

            # Global indices for this batch's nodes: offset, offset+1, ..., offset+n-1
            global_idx = torch.arange(n, device=device) + offset  # (n,)

            # ---- Sequence edges (vectorized with broadcasting) ----
            # diff[i,j] = |i - j|; keep entries where 0 < diff <= seq_neighborhood
            local = torch.arange(n, device=device)
            diff = (local.unsqueeze(0) - local.unsqueeze(1)).abs()  # (n, n)
            seq_mask = (diff > 0) & (diff <= self.seq_neighborhood)
            src_local, dst_local = seq_mask.nonzero(as_tuple=True)
            if src_local.numel() > 0:
                seq_edge_parts.append(
                    torch.stack([global_idx[src_local], global_idx[dst_local]])
                )  # (2, E_seq_b)

            # ---- Similarity edges (topk on cosine sim matrix) ----
            if n > 1:
                valid_emb = embeddings[b, mask[b]]              # (n, D)
                norm_emb = F.normalize(valid_emb, p=2, dim=1)
                sim_matrix = torch.mm(norm_emb, norm_emb.t())  # (n, n)
                sim_matrix.fill_diagonal_(-float('inf'))
                k = min(self.k_neighbors, n - 1)
                if k > 0:
                    _, topk_idx = torch.topk(sim_matrix, k=k, dim=1)  # (n, k)
                    # Vectorized: build src/dst without any Python loops
                    src_local = torch.arange(n, device=device).unsqueeze(1).expand(n, k).reshape(-1)
                    dst_local = topk_idx.reshape(-1)
                    sim_edge_parts.append(
                        torch.stack([global_idx[src_local], global_idx[dst_local]])
                    )  # (2, E_sim_b)

        seq_edges = (
            torch.cat(seq_edge_parts, dim=1)
            if seq_edge_parts
            else torch.zeros(2, 0, dtype=torch.long, device=device)
        )
        sim_edges = (
            torch.cat(sim_edge_parts, dim=1)
            if sim_edge_parts
            else torch.zeros(2, 0, dtype=torch.long, device=device)
        )

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
        
        # Final normalization to maintain consistent embedding space  #can remove the F.normalize if still too similar
        #protein_embeddings = F.normalize(protein_embeddings, p=2, dim=1)
        #if remove F.normalize not stable, try following
        """
        # Step 1: Get original magnitudes (how "strong" each embedding is)
        original_norms = torch.norm(protein_embeddings, p=2, dim=1, keepdim=True)
        # Step 2: Get normalized directions (which direction, but length=1)
        directions = F.normalize(protein_embeddings, p=2, dim=1)
        # Step 3: Combine - keep direction but modify magnitude
        protein_embeddings = directions * torch.tanh(original_norms) #add constrain with tanh on magnitudes.
        """
        
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