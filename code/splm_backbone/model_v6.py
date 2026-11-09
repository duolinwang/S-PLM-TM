import torch.nn as nn
import torch
import torch.nn.functional as F


from peft import PeftModel, LoraConfig, get_peft_model
import gvp.models

# 序列编码器：本目录用 ProtT5 / ProstT5（只用 encoder）。
# 起别名 ESM2 是为了不改按名字实例化的调用点 ——
# 实现已经完全换成 T5 了。
from protT5_encoder import ProtT5, MoBYMLP
ESM2 = ProtT5

from torch_geometric.nn import radius, global_mean_pool, global_max_pool
from utils.SequenceSimGNN_pooling import ProteinGNN
from utils.SequenceSimGNN_pooling_V2 import ProteinGNN as ProteinGNN_V2
from utils.SequenceSimGNN_pooling_V3 import ProteinGNN as ProteinGNN_V3
from utils.robust_norm import RobustNormConfig,EnhancedRobustNorm
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

import math

def patch_num_32(n):
    remainder = n % 32
    if remainder == 0:
        return n // 32
    else:
        return (n + (32 - remainder)) // 32


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_negative_mean_logtis(logits, mode, minibatch_size):
    N = minibatch_size
    if mode == "struct_struct":
        a11 = logits[0:N, 1:N]
        return a11.mean().item()

    if mode == "struct_seq":
        a12 = logits[0:N, N:2 * N - 1]
        return a12.mean().item()

    if mode == "seq_struct":
        a21 = logits[N:2 * N, 1:N]
        return a21.mean().item()
    if mode == "seq_seq":
        a22 = logits[N:2 * N, N:2 * N - 1]
        return a22.mean().item()


class GVPEncoder(nn.Module):
    """
    修改版的GVPEncoder，支持Enhanced Robust Normalization
    """
    def __init__(self, configs=None, residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2,
                 seqlen=512,
                 # Robust norm相关参数
                 use_robust_norm=True,
                 robust_norm_type='instance',  # 'instance' or 'batch'
                 percentile_range=(25, 75),
                 warmup_steps=1000,
                 warmup_strategy='none'):  # 'none', 'l2', 'gradual'
        super(GVPEncoder, self).__init__()
        
        # 保持原有的GVP初始化逻辑
        if hasattr(configs.model.struct_encoder,"version") and configs.model.struct_encoder.version == 'v2':
              node_in_dim = [12+11, 3]
        else:
            node_in_dim = [6, 3]
        
        if configs.model.struct_encoder.use_foldseek:
            node_in_dim[0] += 10
        
        if configs.model.struct_encoder.use_foldseek_vector:
            node_in_dim[1] += 6
        
        node_in_dim = tuple(node_in_dim)
        if configs.model.struct_encoder.use_rotary_embeddings:
            if configs.model.struct_encoder.rotary_mode==3:
                edge_in_dim = (configs.model.struct_encoder.num_rbf+8,1)
            else: 
                edge_in_dim = (configs.model.struct_encoder.num_rbf+2,1)
        else:
              edge_in_dim = (configs.model.struct_encoder.num_rbf+configs.model.struct_encoder.num_positional_embeddings, 1)
        
        if hasattr(configs.model.struct_encoder,"version") and configs.model.struct_encoder.version == 'v2':
            edge_in_dim = (edge_in_dim[0]+7,1)
        
        node_h_dim = configs.model.struct_encoder.node_h_dim
        # node_h_dim=(100, 16) #default
        #node_h_dim = (100, 32)  # seems best?
        edge_h_dim=configs.model.struct_encoder.edge_h_dim
        #edge_h_dim = (32, 1) #default
        gvp_num_layers = configs.model.struct_encoder.gvp_num_layers
        #gvp_num_layers = 3
        self.use_seq = configs.model.struct_encoder.use_seq.enable
        if self.use_seq:
            self.seq_embed_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
            import gvp.models
            self.backbone = gvp.models.structure_encoder(node_in_dim, node_h_dim,
                                                         edge_in_dim, edge_h_dim, seq_in=True,
                                                         seq_embed_mode=self.seq_embed_mode,
                                                         seq_embed_dim=configs.model.struct_encoder.use_seq.seq_embed_dim,
                                                         num_layers=gvp_num_layers)
        else:
            import gvp.models
            self.backbone = gvp.models.structure_encoder(node_in_dim, node_h_dim,
                                                         edge_in_dim, edge_h_dim, seq_in=False,
                                                         num_layers=gvp_num_layers)

        dim_mlp = node_h_dim[0]
        
        # ===== Enhanced Robust Normalization =====
        self.use_robust_norm = use_robust_norm
        self.robust_norm_type = robust_norm_type
        
        if self.use_robust_norm:
            self.robust_norm_residue = EnhancedRobustNorm(
                norm_type=robust_norm_type,
                percentile_range=percentile_range,
                warmup_steps=warmup_steps,
                warmup_strategy=warmup_strategy
            )
        
        # 原有的projectors
        self.projectors_protein = MoBYMLP(in_dim=dim_mlp, inner_dim=protein_inner_dim, out_dim=protein_out_dim,
                                          num_layers=protein_num_projector)

        self.projectors_residue = MoBYMLP(in_dim=dim_mlp, inner_dim=residue_inner_dim, out_dim=residue_out_dim,
                                          num_layers=residue_num_projector)
        
        # 按 config 冻结结构塔与蛋白级 projector
        if hasattr(configs.model.struct_encoder, "fine_tuning") and not configs.model.struct_encoder.fine_tuning.enable:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        if hasattr(configs.model.struct_encoder, "fine_tuning_projct") and not configs.model.struct_encoder.fine_tuning_projct.enable:
            for name, param in self.projectors_protein.named_parameters():
                param.requires_grad = False

            for name, param in self.projectors_residue.named_parameters():
                param.requires_grad = False

        # ===== Perceiver Pooling (replaces global_mean_pool when enabled) =====
        # Backward-compatible: old configs have no perceiver_pool → use_perceiver=False
        _struct_cfg = getattr(configs.model, 'struct_encoder', None)
        perceiver_cfg = getattr(_struct_cfg, 'perceiver_pool', None)
        self.use_perceiver = getattr(perceiver_cfg, 'enable', False)
        if self.use_perceiver:
            self.perceiver_pool = PerceiverPooling(
                d_model=dim_mlp,
                n_latents=getattr(perceiver_cfg, 'n_latents', 32),
                n_heads=getattr(perceiver_cfg, 'n_heads', 4),
                n_self_attn_layers=getattr(perceiver_cfg, 'n_self_attn_layers', 1),
                use_dist_pe=getattr(perceiver_cfg, 'use_dist_pe', True),
            )

        # ===== Structural Aux Losses: contact & distance prediction =====
        # Backward-compatible: old configs have no struct_aux → all weights=0 → disabled
        aux_cfg = getattr(_struct_cfg, 'struct_aux', None)
        self.aux_contact_weight   = getattr(aux_cfg, 'contact_weight',   0.0)
        self.aux_dist_weight      = getattr(aux_cfg, 'dist_weight',      0.0)
        self.aux_n_pairs          = getattr(aux_cfg, 'n_pairs',          256)
        self.aux_contact_threshold= getattr(aux_cfg, 'contact_threshold', 8.0)
        self.aux_min_seq_sep      = getattr(aux_cfg, 'min_seq_sep',       12)
        self.has_struct_aux       = (self.aux_contact_weight > 0 or self.aux_dist_weight > 0)

        if self.aux_contact_weight > 0:
            self.contact_head = nn.Sequential(
                nn.Linear(dim_mlp * 2, dim_mlp), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(dim_mlp, 1)
            )
        if self.aux_dist_weight > 0:
            n_dist_bins = getattr(aux_cfg, 'n_dist_bins', 64)
            # AlphaFold2-style distogram: 2.3125~21.6875 Å, uniform ~0.307 Å/bin
            # Distances beyond 21.6875 Å fall into the last bin (catch-all)
            self.register_buffer('dist_boundaries',
                                 torch.linspace(2.3125, 21.6875, n_dist_bins - 1))
            self.dist_head = nn.Sequential(
                nn.Linear(dim_mlp * 2, dim_mlp), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(dim_mlp, n_dist_bins)
            )

    def forward(self, graph, esm2_representation=None, return_embedding=False):
        """
        Args:
            graph: torch_geometric batch data
        """
        #print("graph.node shapes")
        #print(graph.node_s.shape) #torch.Size([20650, 6])         torch.Size([16865, 23])
        #print(graph.node_v.shape) #torch.Size([20650, 9, 3])      torch.Size([16865, 9, 3])
        #print(graph.edge_s.shape) #torch.Size([619500, 18])     torch.Size([443846, 25])
        #print(graph.edge_v.shape) #torch.Size([619500, 1, 3])    torch.Size([443846, 1, 3])
        nodes = (graph.node_s, graph.node_v)
        edges = (graph.edge_s, graph.edge_v)
        
        # Backbone forward
        if self.use_seq and self.seq_embed_mode != "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges, seq=graph.seq)
        elif self.use_seq and self.seq_embed_mode == "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges, seq=esm2_representation)
        else:
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges, seq=None)


        # ===== Apply Enhanced Robust Normalization =====
        if self.use_robust_norm:
            if self.robust_norm_type == 'instance':
                # Instance normalization: per-protein statistics
                #with torch.cuda.amp.autocast():
                residue_feature_embedding_normed = self.robust_norm_residue(
                        residue_feature_embedding, batch_indices=graph.batch)
            else:  # batch normalization
                # Batch normalization: across all residues/proteins in batch
                #with torch.cuda.amp.autocast():
                residue_feature_embedding_normed = self.robust_norm_residue(residue_feature_embedding)
        else:
            # No robust norm, keep original behavior
            residue_feature_embedding_normed = residue_feature_embedding
        
        # Global pooling for protein-level features
        # use_perceiver: Perceiver cross-attention pooling (structure-aware)
        # else: plain global_mean_pool (original behaviour, backward-compatible)
        with torch.cuda.amp.autocast():
            if self.use_perceiver:
                graph_feature_embedding_normed = self.perceiver_pool(
                    residue_feature_embedding_normed, graph.batch, graph.x)
            else:
                graph_feature_embedding_normed = global_mean_pool(
                    residue_feature_embedding_normed, graph.batch)
        
        # Apply projectors
        with torch.cuda.amp.autocast():
           graph_feature = self.projectors_protein(graph_feature_embedding_normed)
           residue_feature = self.projectors_residue(residue_feature_embedding_normed)
        
        if return_embedding:
            return graph_feature, residue_feature, graph_feature_embedding_normed, residue_feature_embedding_normed
        else:
            return graph_feature, residue_feature

    def compute_struct_aux_loss(self, graph):
        """
        Compute contact- and distance-prediction auxiliary losses on struct graph.
        Called OUTSIDE torch.no_grad() so gradients flow through backbone.
        Only runs when has_struct_aux=True (aux weights > 0 in config).
        """
        if not self.has_struct_aux:
            return {}

        nodes = (graph.node_s, graph.node_v)
        edges = (graph.edge_s, graph.edge_v)

        if self.use_seq and self.seq_embed_mode != "ESM2":
            residue_emb = self.backbone(nodes, graph.edge_index, edges, seq=graph.seq)
        else:
            residue_emb = self.backbone(nodes, graph.edge_index, edges, seq=None)

        if self.use_robust_norm:
            if self.robust_norm_type == 'instance':
                residue_emb = self.robust_norm_residue(
                    residue_emb, batch_indices=graph.batch)
            else:
                residue_emb = self.robust_norm_residue(residue_emb)

        coords = graph.x  # [N, 3] Cα coordinates
        gi, gj, dists, labels = self._sample_long_range_pairs(coords, graph.batch)
        if gi is None:
            return {}

        pair_emb = torch.cat([residue_emb[gi], residue_emb[gj]], dim=-1)  # [K, 2d]

        aux_losses = {}
        with torch.cuda.amp.autocast():
            if hasattr(self, 'contact_head'):
                logits = self.contact_head(pair_emb).squeeze(-1)
                aux_losses['contact'] = F.binary_cross_entropy_with_logits(
                    logits, labels)
            if hasattr(self, 'dist_head'):
                logits = self.dist_head(pair_emb)
                bins = torch.bucketize(dists.detach(), self.dist_boundaries)
                aux_losses['dist'] = F.cross_entropy(logits, bins)
        return aux_losses

    def _sample_long_range_pairs(self, coords, batch):
        """
        Sample long-range residue pairs (seqsep >= aux_min_seq_sep) per protein.
        Uses rejection sampling for simplicity.
        Returns (gi, gj, dists, contact_labels) or (None,)*4 if no valid pairs.
        """
        n_target = self.aux_n_pairs
        min_sep  = self.aux_min_seq_sep
        thresh   = self.aux_contact_threshold

        all_i, all_j = [], []
        for pid in torch.unique(batch):
            idx = torch.where(batch == pid)[0]
            L = len(idx)
            if L <= min_sep + 1:
                continue
            k = min(n_target, L * (L - 1) // 2)
            oversample = min(k * 6, L * L)
            si = torch.randint(0, L, (oversample,), device=batch.device)
            sj = torch.randint(0, L, (oversample,), device=batch.device)
            valid = (si - sj).abs() >= min_sep
            si, sj = si[valid][:k], sj[valid][:k]
            if len(si) == 0:
                continue
            all_i.append(idx[si])
            all_j.append(idx[sj])

        if not all_i:
            return None, None, None, None

        gi = torch.cat(all_i)
        gj = torch.cat(all_j)
        dists  = (coords[gi] - coords[gj]).norm(dim=-1)
        labels = (dists < thresh).float()
        return gi, gj, dists, labels


class MaskedLMDataCollator:
    """Data collator for masked language modeling.
    
    The idea is based on the implementation of DataCollatorForLanguageModeling at
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L751C7-L782
    """
    
    def __init__(self, batch_converter, mlm_probability=0.15):
        """_summary_

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.15.
        """
        self.mlm_probability = mlm_probability
        self.special_token_indices = [batch_converter.alphabet.cls_idx, 
                                batch_converter.alphabet.padding_idx, 
                                batch_converter.alphabet.eos_idx,  
                                batch_converter.alphabet.unk_idx, 
                                batch_converter.alphabet.mask_idx]
        self.vocab_size = batch_converter.alphabet.all_toks.__len__()
        self.mask_idx = batch_converter.alphabet.mask_idx
        

        
    def get_special_tokens_mask(self, tokens):
        return [1 if token in self.special_token_indices else 0 for token in tokens]
    
    def mask_tokens(self, batch_tokens):
        """make a masked input and label from batch_tokens.

        Args:
            batch_tokens (tensor): tensor of batch tokens
            batch_converter (tensor): batch converter from ESM-2.

        Returns:
            inputs: inputs with masked
            labels: labels for masked tokens
        """
        ## mask tokens
        inputs = batch_tokens.clone().to(batch_tokens.device)
        labels = batch_tokens.clone().to(batch_tokens.device)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [self.get_special_tokens_mask(val) for val in labels ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx
        
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        ## Generate valid random tokens (excluding special tokens)
        valid_tokens = torch.tensor([i for i in range(self.vocab_size) if i not in self.special_token_indices])
        random_indices = torch.randint(0, len(valid_tokens), labels.shape)
        random_words = valid_tokens[random_indices].to(batch_tokens.device) # Select only from valid tokens

        #random_words = torch.randint(self.vocab_size,
        #                             labels.shape, dtype=torch.long).to(batch_tokens.device)
        
        #print(indices_random)
        #print(random_words)
        #print(inputs)
        
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class SimCLR(nn.Module):
    def __init__(self, model_seq, model_struct, configs):
        super(SimCLR, self).__init__()
        self.model_seq = model_seq
        self.model_struct = model_struct
        self.temperature = configs.train_settings.temperature
        self.n_views = configs.train_settings.n_views

    def forward(self, graph=None, batch_tokens=None, mode=False, return_embedding=False,return_logits=False):
        if mode == 'sequence':
            return self.model_seq(batch_tokens, return_embedding=return_embedding,return_logits=return_logits)
        elif mode == 'structure':
            return self.model_struct(graph, return_embedding=return_embedding)
        else:
            features_seq, residue_seq = self.model_seq(batch_tokens)
            features_struct, residue_struct = self.model_struct(graph)
            return features_struct, residue_struct, features_seq, residue_seq

    def forward_structure(self, graph):
        features_struct, residue_struct = self.model_struct(graph)
        return features_struct, residue_struct

    def forward_sequence(self, batch_tokens):
        features_seq, residue_seq = self.model_seq(batch_tokens)
        return features_seq, residue_seq


def info_nce_loss(features_struct, features_seq, n_views, temperature):  # loss function
    batch_size = len(features_struct)

    labels = torch.cat([torch.arange(batch_size, device=features_struct.device) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features_struct.device)

    features_struct = F.normalize(features_struct, dim=1)
    features_seq = F.normalize(features_seq, dim=1)
    features = torch.cat([features_struct, features_seq], dim=0)

    similarity_matrix = torch.matmul(features, features.T)

    # Discard the main diagonal from both labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features_struct.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # Select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features_struct.device)

    logits = logits / temperature
    return logits, labels


def print_trainable_parameters(model, logging):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_models(configs, robust_norm_config=None, logging=None, accelerator=None):
    """
    创建带有robust normalization的GVP模型
    
    Args:
        configs: 原有配置
        robust_norm_config: RobustNormConfig对象或dict
        logging: 日志对象
        accelerator: 加速器对象
    """
    if robust_norm_config is None:
        # 默认使用推荐配置
        robust_norm_config = RobustNormConfig.get_predefined_configs()['strategy_a_instance']
    
    if isinstance(robust_norm_config, RobustNormConfig):
        robust_norm_dict = robust_norm_config.to_dict()
    else:
        robust_norm_dict = robust_norm_config
    
    model_struct = GVPEncoder(configs=configs, residue_inner_dim=configs.model.struct_encoder.residue_inner_dim,
                              residue_out_dim=configs.model.residue_out_dim,
                              protein_out_dim=configs.model.protein_out_dim,
                              protein_inner_dim=configs.model.struct_encoder.protein_inner_dim,
                              residue_num_projector=configs.model.residue_num_projector,
                              protein_num_projector=configs.model.protein_num_projector,
                              seqlen=configs.model.seq_encoder.max_length,
                              **robust_norm_dict
                              )
    
    # 序列侧：ProtT5。模型名由它自己从 configs.model.seq_encoder 读，
    # 不再作为第一个位置参数传入（ESM2 那版是那样传的）。
    model_seq = ESM2(residue_inner_dim=configs.model.seq_encoder.residue_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_num_projector=configs.model.protein_num_projector,
                     configs=configs, logging=logging)
    
    
    if hasattr(configs.train_settings,"train_lm_head_only"):
            if configs.train_settings.train_lm_head_only is True:
                for name, param in model_struct.named_parameters():
                    param.requires_grad = False
                for name, param in model_seq.named_parameters():
                    param.requires_grad = False
                # ProtT5 只加载 encoder、没有 lm_head，这条路在本目录不适用。
                raise NotImplementedError(
                    "train_lm_head_only 需要 LM head，而本目录只加载 T5 encoder。"
                    "请在 config 里去掉该开关。")
    
    if logging is not None and accelerator is not None and accelerator.is_main_process:
       print_trainable_parameters(model_seq, logging)
       print_trainable_parameters(model_struct, logging)
    
    simclr = SimCLR(model_seq, model_struct, configs=configs)
    return simclr  # model_seq, model_struct #, new_model

def info_nce_weights(features_struct, features_seq, device, n_views=2, mode="positive"):  # loss function
    """
    mode = "positive" only apply plddt weights on positive pairs
    mode = "all" apply plddt weights on all pairs
    """
    # print("batch_size="+str(self.args.batch_size))
    features_struct = features_struct.to(device)
    features_seq = features_seq.to(device)
    batch_size = len(features_struct)
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)],
                       dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = torch.cat([features_struct, features_seq], dim=0)
    if len(features.shape) < 2:
        features = features.unsqueeze(1)

    avg_weights_matrix = (features + features.T) / 2.0

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    avg_weights_matrix = avg_weights_matrix[~mask].view(avg_weights_matrix.shape[0], -1)
    # apply plddt weight on positives
    positives = avg_weights_matrix[labels.bool()].view(labels.shape[0], -1)
    # apply plddt weight on negatives
    # negatives = avg_weights_matrix[~labels.bool()].view(avg_weights_matrix.shape[0], -1)
    # only apply plddt weight on positives
    if mode == "positive":
        negative_weights = torch.ones(avg_weights_matrix.shape).to(device)
        negatives = negative_weights[~labels.bool()].view(negative_weights.shape[0], -1)
    elif mode == "all":
        negatives = avg_weights_matrix[~labels.bool()].view(avg_weights_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    return logits


if __name__ == '__main__':
    print('test')
