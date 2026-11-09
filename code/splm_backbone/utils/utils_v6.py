# -*- coding: utf-8 -*-
"""配置解析 —— 推理侧需要的唯一一个 utils_v6 函数。

2026-10-26：release 的 code/ 只保留推理。这个文件原本有 923 行、28 个顶层符号，
其中 27 个是训练用的（优化器、调度器、checkpoint 存取、wandb / tensorboard 初始化
…），并且因此在顶部 import 了 timm / wandb / tensorboard / cosine_annealing_warmup
—— 推理包要装这四个依赖的唯一原因。

推理路径只用 `load_configs`（utils/generate_seq_embedding.py 里调用），而它完全
自包含：不碰同文件任何其他符号，模块级只需要 Box 与 ast。所以这里只留它。

⚠️ 下面的 load_configs 是**逐字原文**，没有任何改动 —— 发布 checkpoint 的解析
   行为必须与训练时完全一致。
⚠️ 训练侧的完整版在 training/splm_backbone/utils/utils_v6.py，它不复制这个函数，
   而是从本文件动态加载后重新导出。**要改 load_configs 只改这里一处。**
"""
import ast

from box import Box


def load_configs(config, args=None):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    #tree_config.optimizer.decay.min_lr_struct = float(tree_config.optimizer.decay.min_lr_struct)
    #tree_config.optimizer.decay.min_lr_seq = float(tree_config.optimizer.decay.min_lr_seq)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    tree_config.train_settings.temperature = float(tree_config.train_settings.temperature)
    tree_config.optimizer.beta_1 = float(tree_config.optimizer.beta_1)
    tree_config.optimizer.beta_2 = float(tree_config.optimizer.beta_2)
    # 序列侧（ProtT5）不使用 LoRA，seq_encoder 下没有 lora 段 —— 这行删除。
    # 结构侧的 lora 段仍在，下一行保留。
    tree_config.model.struct_encoder.lora.dropout = float(tree_config.model.struct_encoder.lora.dropout)

    # overwrite parameters if set through commandline
    if args is not None:
        if args.result_path:
            tree_config.result_path = args.result_path

        if args.resume_path:
            tree_config.resume.resume_path = args.resume_path

        if args.num_end_adapter_layers:
            tree_config.encoder.adapter_h.num_end_adapter_layers = int(args.num_end_adapter_layers)

        if args.module_type:
            tree_config.encoder.adapter_h.module_type = args.module_type
        
        if args.restart_optimizer:
           tree_config.resume.restart_optimizer = args.restart_optimizer

    
    #set configs value to default if doesn't have the attr
    if not hasattr(tree_config.model.struct_encoder, "use_seq"):
        tree_config.model.struct_encoder.use_seq = None
        tree_config.model.struct_encoder.use_seq.enable = False
        tree_config.model.struct_encoder.use_seq.seq_embed_mode = "embedding"
        tree_config.model.struct_encoder.use_seq.seq_embed_dim = 20
    
    if not hasattr(tree_config.model.struct_encoder,"top_k"):
       tree_config.model.struct_encoder.top_k = 30 #default
    
    if not hasattr(tree_config.model.struct_encoder,"gvp_num_layers"):
       tree_config.model.struct_encoder.gvp_num_layers = 3 #default
    
    if not hasattr(tree_config.model.struct_encoder,"use_rotary_embeddings"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_rotary_embeddings=False
    
    if not hasattr(tree_config.model.struct_encoder,"rotary_mode"):
        tree_config.model.struct_encoder.rotary_mode=1
    
    if not hasattr(tree_config.model.struct_encoder,"use_foldseek"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek=False
    
    if not hasattr(tree_config.model.struct_encoder,"use_foldseek_vector"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek_vector=False
    
    if not hasattr(tree_config.model.struct_encoder,"num_rbf"):
       tree_config.model.struct_encoder.num_rbf = 16 #default
    
    if not hasattr(tree_config.model.struct_encoder,"num_positional_embeddings"):
       tree_config.model.struct_encoder.num_positional_embeddings = 16 #default
    
    if not hasattr(tree_config.model.struct_encoder,"node_h_dim"):
       tree_config.model.struct_encoder.node_h_dim = (100,32) #default
    else:
       tree_config.model.struct_encoder.node_h_dim = ast.literal_eval(tree_config.model.struct_encoder.node_h_dim)
    
    if not hasattr(tree_config.model.struct_encoder,"edge_h_dim"):
       tree_config.model.struct_encoder.edge_h_dim = (32,1) #default
    else:
       tree_config.model.struct_encoder.edge_h_dim = ast.literal_eval(tree_config.model.struct_encoder.edge_h_dim)
    
    
    
    return tree_config
