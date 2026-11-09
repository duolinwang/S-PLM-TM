import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch

class GVPLayer(nn.Module):
    """
    Simplified GVP-GNN layer for sequence-only data
    """
    def __init__(self, node_dim, edge_dim=2):
        super(GVPLayer, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # GCN for sequence-adjacent connections
        self.seq_conv = GCNConv(node_dim, node_dim)
        
        # GAT for embedding-similarity connections
        self.sim_conv = GATConv(node_dim, node_dim, heads=4, concat=False)
        
        # MLP to combine both types of messages
        self.combine = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
    
    def forward(self, x, seq_edge_index, sim_edge_index, edge_attr=None):
        # Process sequence-adjacent connections
        seq_out = self.seq_conv(x, seq_edge_index)
        
        # Process embedding-similarity connections
        sim_out = self.sim_conv(x, sim_edge_index)
        
        # Combine the messages
        combined = torch.cat([seq_out, sim_out], dim=-1)
        out = self.combine(combined)
        
        # Residual connection
        return x + out

class ProteinGNN(nn.Module):
    """
    GVP-GNN-inspired model for protein embeddings from ESM2
    """
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=1280, n_layers=3, k_neighbors=10,seq_neighborhood=10):
        super(ProteinGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k_neighbors = k_neighbors
        self.seq_neighborhood = seq_neighborhood  # New parameter 
        # Project ESM2 embeddings if dimensions don't match
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # GVP layers
        self.gvp_layers = nn.ModuleList([
            GVPLayer(hidden_dim) for _ in range(n_layers)
        ])
        
        # Node importance prediction for attention pooling
        self.node_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
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
        
        # Project input if needed
        x = self.input_proj(embeddings)
        
        # Build edges
        seq_edges, sim_edges, batch_idx, total_nodes = self._build_edges(x, mask)
        
        # Gather only the node features we need
        node_features = []
        for b in range(batch_size):
            valid_pos = torch.where(mask[b])[0]
            if len(valid_pos) > 0:
                node_features.append(x[b, valid_pos])
        
        if not node_features:  # Handle empty case
            return torch.zeros(batch_size, self.output_dim, device=embeddings.device)
        
        node_features = torch.cat(node_features, dim=0)
        
        # Apply GVP layers
        for layer in self.gvp_layers:
            node_features = layer(node_features, seq_edges, sim_edges)
        
        # Calculate attention weights for pooling
        node_importance = self.node_importance(node_features).squeeze(-1)
        node_importance = torch.softmax(node_importance, dim=0)
        
        # Weighted pooling per graph/protein
        protein_embeddings = torch.zeros(batch_size, self.output_dim, device=embeddings.device)
        
        for b in range(batch_size):
            b_mask = (batch_idx == b)
            if not torch.any(b_mask):
                continue
                
            b_features = node_features[b_mask]
            b_weights = node_importance[b_mask].unsqueeze(1)
            
            # Attention-weighted sum
            protein_embeddings[b] = torch.sum(b_features * b_weights, dim=0)
        
        return protein_embeddings

# Example usage
def get_protein_embeddings(esm2_embeddings, attention_mask=None):
    """
    Get protein-level embeddings from ESM2 amino acid embeddings
    
    Args:
        esm2_embeddings: Tensor of shape (batch_size, max_len, 1280)
        attention_mask: Boolean mask of shape (batch_size, max_len) where True indicates valid positions
                       If None, assume all positions are valid
    
    Returns:
        Protein-level embeddings of shape (batch_size, 1280)
    """
    model = ProteinGNN(
        input_dim=1280,          # ESM2 embedding dimension
        hidden_dim=1280,         # Keep same dimension internally
        output_dim=1280,         # Output dimension (same as ESM2)
        n_layers=3,              # Number of GVP layers
        k_neighbors=5,           # Number of similarity neighbors
        seq_neighborhood = 2    # Number of sequence neighbors
    )
    
    # Move to same device as input
    model = model.to(esm2_embeddings.device)
    
    # Forward pass
    with torch.no_grad():
        protein_embeddings = model(esm2_embeddings, attention_mask)
    
    return protein_embeddings


#B, L, D = 40, 12, 1280
#esm2_embeddings = torch.randn(B, L, D)
#mask = torch.constant([0,1,1,1,1,1,0,1,0,1,0,1])
#protein_embeddings = get_protein_embeddings(esm2_embeddings, attention_mask=None)
