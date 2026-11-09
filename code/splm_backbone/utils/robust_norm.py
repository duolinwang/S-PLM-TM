import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class EnhancedRobustNorm(nn.Module):
    """
    Enhanced Robust Normalization with support for different warmup strategies
    """
    def __init__(self, 
                 norm_type='instance',  # 'instance' or 'batch'
                 percentile_range=(25, 75), 
                 eps=1e-6, 
                 momentum=0.1,
                 warmup_steps=1000,
                 warmup_strategy='none'):  # 'none', 'l2', 'gradual'
        super().__init__()
        self.norm_type = norm_type
        self.percentile_range = percentile_range
        self.eps = eps
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.warmup_strategy = warmup_strategy
        # 共享的step_count - 两种模式都可以使用
        self.register_buffer('step_count', torch.tensor(0))
        
        # Running statistics for inference (only for batch mode)
        if norm_type == 'batch':
            self.register_buffer('running_median', None)
            self.register_buffer('running_iqr', None)
        
    def forward(self, x, batch_indices=None):
        """
        Args:
            x: [num_residues, embed_dim] or [batch_size, embed_dim]
            batch_indices: [num_residues] indicating which protein each residue belongs to (for instance mode)
        """
        
        if self.norm_type == 'instance':
            if self.training:
                return self._instance_normalize_train(x, batch_indices)
            else:
                return self._apply_instance_normalization(x, batch_indices)
        elif self.norm_type == 'batch':
            return self._batch_normalize(x)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
        
    def _instance_normalize_train(self, x, batch_indices=None):
        """
        训练时的instance normalization with warmup support
        """
        self.step_count += 1
        # Warmup handling
        if self.step_count < self.warmup_steps or self.warmup_steps == -1: #if warmup_steps = -1 means no robust norm for training
            return self._handle_warmup_instance(x, self.step_count, batch_indices)
        
        # 正常的instance normalization
        return self._apply_instance_normalization(x, batch_indices)
    
    def _handle_warmup_instance(self, x, step, batch_indices):
        """
        Instance mode的warmup处理
        """
        if self.warmup_strategy == 'none':
            # Strategy A: 保持原始特征
            return x
        elif self.warmup_strategy == 'l2':
            # Strategy B: 使用L2 normalization
            return F.normalize(x, p=2, dim=-1)
        elif self.warmup_strategy == 'gradual':
            # Strategy C: 渐进式从原始特征过渡到robust norm
            alpha = step / self.warmup_steps
            x_normalized = self._apply_instance_normalization(x, batch_indices)
            return alpha * x_normalized + (1 - alpha) * x
        else:
            raise ValueError(f"Unknown warmup_strategy: {self.warmup_strategy}")
    
    def _apply_instance_normalization(self, x, batch_indices):
        """
        应用instance normalization的核心逻辑
        """
        if batch_indices is None:
            # 如果没有batch_indices，假设所有residues属于同一个蛋白质
            return self._normalize_single_protein(x)
        
        # 对每个蛋白质分别进行normalization
        normalized_features = []
        unique_indices = torch.unique(batch_indices)
        
        for protein_idx in unique_indices:
            protein_mask = (batch_indices == protein_idx)
            protein_features = x[protein_mask]
            normalized_protein = self._normalize_single_protein(protein_features)
            normalized_features.append(normalized_protein)
        
        # 重新组装，保持原始顺序
        result = torch.zeros_like(x)
        for i, protein_idx in enumerate(unique_indices):
            protein_mask = (batch_indices == protein_idx)
            result[protein_mask] = normalized_features[i]
            
        return result
    
    def _normalize_single_protein(self, protein_features):
        """
        protein_features [num_residues, embed_dim]
        对单个蛋白质进行robust normalization
        """
        # 检查特征数量，如果太少则fallback到原始特征
        if protein_features.size(0) < 8:  # 残基太少
            return protein_features  # 保持原始特征
        
        # ===== 修复：确保数据类型正确 =====
        # 保存原始dtype
        original_dtype = protein_features.dtype
        
        # 转换为float32进行计算
        if protein_features.dtype not in [torch.float32, torch.float64]:
            protein_features = protein_features.float()
        
        # 检查并清理输入
        if torch.isnan(protein_features).any() or torch.isinf(protein_features).any():
            #print("WARNING: Input contains NaN/inf, replacing with zeros")
            protein_features = torch.where(
                torch.isnan(protein_features) | torch.isinf(protein_features),
                torch.zeros_like(protein_features),
                protein_features
            )
        
        # 计算分位数
        q1 = torch.quantile(protein_features, self.percentile_range[0]/100, dim=0, keepdim=True)
        q3 = torch.quantile(protein_features, self.percentile_range[1]/100, dim=0, keepdim=True)
        median = torch.median(protein_features, dim=0, keepdim=True).values
        
        iqr = q3 - q1
        # Debug信息
        small_iqr_count = (iqr < self.eps).sum()
        #if small_iqr_count > 0:
        #    print(f"DEBUG: step_count {self.step_count} {small_iqr_count} dimensions have IQR < eps({self.eps})")
        #    print(f"DEBUG: step_count {self.step_count} min IQR = {iqr.min().item():.8f}")
        
        iqr = torch.clamp(iqr, min=self.eps)
        # Robust scaling
        x_robust = (protein_features - median) / iqr
        # 更精确的outlier处理：只clamp真正的outliers
        #abs_values = torch.abs(x_robust)
        #outlier_threshold = torch.quantile(abs_values, 0.99)  # 99%分位数
        #outlier_threshold = torch.clamp(outlier_threshold, min=10.0, max=50.0)
        #x_robust = torch.clamp(x_robust, min=-outlier_threshold, max=outlier_threshold)
        x_robust = torch.clamp(x_robust, min=-10.0, max=10.0)
        # 调试：打印实际范围
        #print(f"DEBUG step_count {self.step_count} x_robust range: [{x_robust.min().item():.4f}, {x_robust.max().item():.4f}]")
        #print(f"DEBUG step_count {self.step_count} x_robust std: {x_robust.std().item():.4f}")
        
        # 检查异常值
        #if torch.abs(x_robust).max() > 100:  # 如果有异常大的值
        #    print("Warning: Very large values detected in x_robust")
        #    # 这时再决定合适的clamp范围
        
        # L2 normalization
        normalized = F.normalize(x_robust, p=2, dim=-1)
        # 最终检查 - 确保输出绝对干净
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            print("WARNING: Final result contains NaN/inf, returning zero tensor")
            normalized = torch.zeros_like(normalized)
        
        # ===== 转换回原始dtype =====
        if original_dtype != normalized.dtype:
            normalized = normalized.to(original_dtype)
        
        return normalized
    
    def _batch_normalize(self, x):
        """
        Batch-level normalization
        """
        if self.training:
            return self._batch_normalize_train(x)
        else:
            return self._batch_normalize_inference(x)
    
    def _batch_normalize_train(self, x):
        """
        训练时的batch normalization with different warmup strategies
        """
        self.step_count += 1
        
        # Warm-up handling
        if self.step_count < self.warmup_steps or self.warmup_steps == -1:
            return self._handle_warmup_batch(x, self.step_count)
        
        # Check batch size
        if x.size(0) < 16:  # 批次太小，使用running stats
            return self._batch_normalize_inference(x)
        
        # 正常的robust normalization
        return self._apply_robust_normalization(x)
    
    def _handle_warmup_batch(self, x, step):
        """
        处理不同的warmup策略
        """
        if self.warmup_strategy == 'none':
            # Strategy A: 保持原始特征
            return x
        elif self.warmup_strategy == 'l2':
            # Strategy B: 使用L2 normalization
            return F.normalize(x, p=2, dim=-1)
        elif self.warmup_strategy == 'gradual':
            # Strategy C: 逐渐从原始特征过渡到robust norm
            alpha = step / self.warmup_steps
            if x.size(0) >= 16:  # 有足够样本计算robust norm
                x_robust = self._apply_robust_normalization(x, update_stats=False)
                return alpha * x_robust + (1 - alpha) * x
            else:
                # 样本太少，在原始特征和L2 norm之间插值
                x_l2 = F.normalize(x, p=2, dim=-1)
                return alpha * x_l2 + (1 - alpha) * x
        else:
            raise ValueError(f"Unknown warmup_strategy: {self.warmup_strategy}")
    
    def _apply_robust_normalization(self, x, update_stats=True):
            """
            应用robust normalization (只用于batch mode)
            """
            # ===== 确保数据类型正确 =====
            original_dtype = x.dtype
            if x.dtype not in [torch.float32, torch.float64]:
                x = x.float()
            
            # 计算batch统计量
            q1 = torch.quantile(x, self.percentile_range[0]/100, dim=0, keepdim=True)
            q3 = torch.quantile(x, self.percentile_range[1]/100, dim=0, keepdim=True)
            median = torch.median(x, dim=0, keepdim=True).values
            
            iqr = torch.clamp(q3 - q1, min=self.eps)
            
            # Update running statistics
            if update_stats:
                with torch.no_grad():
                    if self.running_median is None:
                        self.running_median = median.detach().clone()
                        self.running_iqr = iqr.detach().clone()
                    else:
                        self.running_median = (self.momentum * self.running_median + 
                                             (1 - self.momentum) * median.detach())
                        self.running_iqr = (self.momentum * self.running_iqr + 
                                          (1 - self.momentum) * iqr.detach())
            
            # Normalize
            x_robust = (x - median) / iqr
            x_robust = torch.clamp(x_robust, min=-10.0, max=10.0)
            normalized = F.normalize(x_robust, p=2, dim=-1)
            # 最终检查 - 确保输出绝对干净
            if torch.isnan(normalized).any() or torch.isinf(normalized).any():
               print("WARNING: Final result contains NaN/inf, returning zero tensor")
            
            # 转换回原始dtype
            if original_dtype != normalized.dtype:
                normalized = normalized.to(original_dtype)
            
            return normalized
    
    def _batch_normalize_inference(self, x):
        """
        推理时的batch normalization
        """
        if self.running_median is not None:
            # ===== 确保数据类型正确 =====
            original_dtype = x.dtype
            if x.dtype not in [torch.float32, torch.float64]:
                x = x.float()
            
            x_robust = (x - self.running_median) / self.running_iqr
            normalized = F.normalize(x_robust, p=2, dim=-1)
            
            # 转换回原始dtype
            if original_dtype != normalized.dtype:
                normalized = normalized.to(original_dtype)
            
            return normalized
        else:
            # Fallback to original features
            return x

# ===== 配置和工厂函数 =====

class RobustNormConfig:
    """
    Robust Normalization配置类
    """
    def __init__(self):
        # 基础配置
        self.use_robust_norm = True
        self.robust_norm_type = 'instance'  # 'instance' or 'batch'
        self.percentile_range = (25, 75)
        self.warmup_steps = 1000
        self.warmup_strategy = 'none'  # 'none', 'l2', 'gradual'
    
    @classmethod
    def get_predefined_configs(cls):
        """
        返回预定义的配置
        """
        configs = {}
        
        # Strategy A: 无warmup，保持原始特征 (推荐)
        configs['strategy_a_instance'] = cls()
        configs['strategy_a_instance'].robust_norm_type = 'instance'
        configs['strategy_a_instance'].warmup_strategy = 'none'
        configs['strategy_a_instance'].warmup_steps = 0  # 直接开始robust norm
        
        configs['strategy_a_batch'] = cls()
        configs['strategy_a_batch'].robust_norm_type = 'batch'
        configs['strategy_a_batch'].warmup_strategy = 'none'
        configs['strategy_a_batch'].warmup_steps = 0
        
        # Strategy B: Warmup使用L2 norm
        configs['strategy_b_instance'] = cls()
        configs['strategy_b_instance'].robust_norm_type = 'instance'
        configs['strategy_b_instance'].warmup_strategy = 'l2'
        configs['strategy_b_instance'].warmup_steps = 1000
        
        configs['strategy_b_batch'] = cls()
        configs['strategy_b_batch'].robust_norm_type = 'batch'
        configs['strategy_b_batch'].warmup_strategy = 'l2'
        configs['strategy_b_batch'].warmup_steps = 1000
        
        # Strategy C: 渐进式过渡
        configs['strategy_c_instance'] = cls()
        configs['strategy_c_instance'].robust_norm_type = 'instance'
        configs['strategy_c_instance'].warmup_strategy = 'gradual'
        configs['strategy_c_instance'].warmup_steps = 2000
        
        configs['strategy_c_batch'] = cls()
        configs['strategy_c_batch'].robust_norm_type = 'batch'
        configs['strategy_c_batch'].warmup_strategy = 'gradual'
        configs['strategy_c_batch'].warmup_steps = 2000
        
        # 不同激进程度的配置
        configs['conservative_instance'] = cls()
        configs['conservative_instance'].robust_norm_type = 'instance'
        configs['conservative_instance'].percentile_range = (25, 75)
        configs['conservative_instance'].warmup_strategy = 'none'
        configs['conservative_instance'].warmup_steps = 0
        
        configs['aggressive_instance'] = cls()
        configs['aggressive_instance'].robust_norm_type = 'instance'
        configs['aggressive_instance'].percentile_range = (10, 90)
        configs['aggressive_instance'].warmup_strategy = 'none'
        configs['aggressive_instance'].warmup_steps = 0
        
        configs['very_aggressive_instance'] = cls()
        configs['very_aggressive_instance'].robust_norm_type = 'instance'
        configs['very_aggressive_instance'].percentile_range = (5, 95)
        configs['very_aggressive_instance'].warmup_strategy = 'none'
        configs['very_aggressive_instance'].warmup_steps = 0
        
        # Baseline: 完全禁用
        configs['disabled'] = cls()
        configs['disabled'].use_robust_norm = False
        
        return configs
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'use_robust_norm': self.use_robust_norm,
            'robust_norm_type': self.robust_norm_type,
            'percentile_range': self.percentile_range,
            'warmup_steps': self.warmup_steps,
            'warmup_strategy': self.warmup_strategy
        }
