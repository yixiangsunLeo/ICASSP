import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import dhg
from dhg.nn import HGNNPConv

# ===================================================================
# 5. 模型定义 (Model Definition - EEG_AdaMSHyper)
# ===================================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha
    def forward(self, x: torch.Tensor, alpha: float = None) -> torch.Tensor:
        current_alpha = alpha if alpha is not None else self.alpha
        return GradientReversalFunction.apply(x, current_alpha)


# ==============================================================================
# 2. 时序对齐组件 (新增)
# ==============================================================================

class LearnableTemporalShift(nn.Module):
    """
    可学习的时序平移模块 (Learnable Temporal Shift).
    """
    def __init__(self, num_channels, sampling_rate, max_shift_sec=0.2):
        super().__init__()
        self.num_channels = num_channels
        # 将最大时间偏移量转换为采样点数 (例如 0.2s * 64Hz = 12.8 points)
        self.max_shift_steps = int(max_shift_sec * sampling_rate)
        
        # 定义可学习参数，初始化为 0 (即初始不偏移)
        # Shape: [1, Channels, 1] 以便广播到 Batch 和 Time 维度
        self.shift_param = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: 输入 EEG 数据 [Batch, Channel, Time]
        Returns:
            x_shifted: 对齐后的 EEG 数据
        """
        B, C, T = x.shape
        
        # 1. 计算偏移步数 
        # Tanh 将参数限制在 [-1, 1]，然后乘以 max_shift_steps
        shift_steps = self.tanh(self.shift_param) * self.max_shift_steps
        
        # 2. 构建基础时间网格 [0, 1, 2, ..., T-1]
        base_grid = torch.arange(T, device=x.device, dtype=torch.float32).view(1, 1, T)
        
        # 3. 生成偏移后的采样位置: [1, C, T]
        # 每个通道都有自己独立的偏移量，广播应用到所有时间点
        new_grid = base_grid + shift_steps
        
        # 4. 可微分线性插值采样 (Differentiable Linear Interpolation)
        # 找到左右相邻的整数索引
        idx_floor = torch.floor(new_grid).long()
        idx_ceil = idx_floor + 1
        alpha = new_grid - torch.floor(new_grid) # 插值权重 (小数部分)
        
        # 边界处理: 防止索引越界 (使用 Clamp 实现 Replicate Padding 效果)
        idx_floor = torch.clamp(idx_floor, 0, T-1).expand(B, -1, -1)
        idx_ceil = torch.clamp(idx_ceil, 0, T-1).expand(B, -1, -1)
        alpha = alpha.expand(B, -1, -1)
        
        # 根据索引取值 (Gather)
        val_floor = torch.gather(x, 2, idx_floor)
        val_ceil = torch.gather(x, 2, idx_ceil)
        
        # 加权融合
        x_shifted = (1 - alpha) * val_floor + alpha * val_ceil
        return x_shifted

# --- 模块1: 时间特征提取器 ---

class TemporalStem(nn.Module):
    """
    时间特征提取的主干网络，改编自您之前的研究成果。
    作用: 对单通道时间序列进行多尺度卷积，并进行降采样以形成分块(Patch)。
    """
    def __init__(self, out_planes: int, kernel_size: int, patch_size: int, radix: int = 2):
        super(TemporalStem, self).__init__()
        # 核心参数设置
        self.radix = radix
        self.in_planes = 1 # 固定为1，以适配单通道处理
        self.out_planes = out_planes
        self.mid_planes = self.out_planes * self.radix

        # 1. 初始逐点卷积，用于升维
        self.sconv = nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.mid_planes)

        # 2. 多分支时间卷积，并行捕捉不同时间尺度的特征
        self.tconv = nn.ModuleList([
            nn.Sequential(
                # 使用深度可分离卷积，在保持通道独立性的同时提高效率
                nn.Conv1d(self.out_planes, self.out_planes, ks, 1, padding=ks // 2, bias=False, groups=self.out_planes),
                nn.BatchNorm1d(self.out_planes)
            ) for ks in [kernel_size // (2**i) for i in range(self.radix)]
        ])

        # 3. 特征融合与降采样（分块）
        self.interFre = lambda x: F.gelu(sum(x)) # 融合多分支特征
        self.downSampling = nn.AvgPool1d(patch_size, stride=patch_size) # 均值池化实现分块
        self.dp = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [批次大小, 1, 时间点数]
        out = self.bn1(self.sconv(x))
        branches = torch.split(out, self.out_planes, dim=1) if self.radix > 1 else [out]
        out_branches = [conv(branch) for conv, branch in zip(self.tconv, branches)]
        out_fused = self.interFre(out_branches)
        out_patched = self.downSampling(out_fused)
        return self.dp(out_patched)

class TemporalFeatureExtractor(nn.Module):
    """
    时间特征提取器模块。
    作用: 使用多尺度主干网络对每个EEG通道独立提取分块特征，并融合时间位置编码。
    """
    def __init__(self, num_channels, d_model, patch_size, time_sample_num):
        super().__init__()
        # 计算时间维度上分块的数量
        num_patches = time_sample_num // patch_size
        self.lts = LearnableTemporalShift(num_channels, sampling_rate=128, max_shift_sec=0.1)
        # 为每个EEG通道创建一个独立的、强大的特征提取器实例
        self.channel_extractors = nn.ModuleList([
            TemporalStem(out_planes=d_model, kernel_size=63, patch_size=patch_size, radix=2) 
            for _ in range(num_channels)
        ])
        
        # 创建可学习的时间位置编码，维度与分块后的特征相匹配
        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, num_channels, num_patches, d_model))

    def forward(self, x):
        """
        输入 x: [批次大小, 通道数, 时间点数] (B, C, T)
        输出:   [批次大小, 通道数, 特征维度] (B, C, D) (作为后续GNN的初始节点特征)
        """
        x = self.lts(x) 
        # 1. 对每个通道独立应用主干网络，提取时间分块特征
        outputs = [extractor(x[:, i, :].unsqueeze(1)) 
                   for i, extractor in enumerate(self.channel_extractors)]
        
        # 2. 将所有通道的结果堆叠起来
        # List of [B, D, P] -> [B, C, D, P] (P是分块数)
        patched_features = torch.stack(outputs, dim=1)
        
        # 3. 转换维度以适配位置编码的广播机制
        # [B, C, D, P] -> [B, C, P, D]
        patched_features = patched_features.permute(0, 1, 3, 2)
        
        # 4. 添加时间位置编码
        features_with_pos = patched_features + self.pos_embedding_temporal
        
        # 5. 通过在时间分块维度上进行平均池化，得到最终的、代表每个通道的单一特征向量
        # [B, C, P, D] -> [B, C, D]
        final_features = features_with_pos.mean(dim=2)
        
        return final_features
    

# --- 模块2: 多尺度超图构建器 ---
class MultiScale_EEG_Hypergraph(nn.Module):
    """
    多尺度自适应超图构建器。
    作用: 为每个预定义的空间尺度，独立地学习一个超图的“结构蓝图”。
          它不处理节点特征，只负责学习节点间的连接关系。
    """
    def __init__(self, configs):
        super().__init__()
        self.scale_node_nums = configs.scale_node_nums
        self.dim = configs.d_model
        self.hyper_num_list = configs.hyper_num
        self.alpha = 3.0
        self.k = configs.k
        # 为每个尺度的节点和超边创建独立的、可学习的嵌入向量
        self.embedhy = nn.ModuleList()
        self.embednod = nn.ModuleList()
        for i in range(len(self.scale_node_nums)):
            num_nodes_at_scale, num_hyperedges_at_scale = self.scale_node_nums[i], self.hyper_num_list[i]
            self.embedhy.append(nn.Embedding(num_hyperedges_at_scale, self.dim))
            self.embednod.append(nn.Embedding(num_nodes_at_scale, self.dim))

    def forward(self, x):
        """
        输入 x: 任意张量 (仅用于获取device信息)
        输出:   一个列表，包含每个尺度的超图结构 (hyperedge_index格式的Tensor)
        """
        hyperedge_all = []
        # 遍历每个尺度 (例如 64节点, 16节点, 4节点)
        for i in range(len(self.scale_node_nums)):
            num_nodes, num_hyperedges = self.scale_node_nums[i], self.hyper_num_list[i]
            hypidxc, nodeidx = torch.arange(num_hyperedges).to(x.device), torch.arange(num_nodes).to(x.device)
            # 获取当前尺度对应的节点和超边嵌入
            hyperedge_embeds, node_embeds = self.embedhy[i](hypidxc), self.embednod[i](nodeidx)
            # 计算节点与超边的相似度矩阵
            sim_matrix = torch.mm(node_embeds, hyperedge_embeds.t())
            # 稀疏化处理，学习连接关系
            adj = F.softmax(F.relu(self.alpha * sim_matrix), dim=1)
            mask = torch.zeros(num_nodes, num_hyperedges, device=x.device)
            k = min(adj.size(1), self.k)
            if k > 0:
                _, topk_indices = adj.topk(k, 1)
                mask.scatter_(1, topk_indices, 1)
            adj = adj * mask
            adj = torch.where(adj > 0.5, 1, 0).float()
            # 移除没有连接任何节点的空超边
            adj = adj[:, (adj != 0).any(dim=0)]
            if adj.shape[1] > 0:
                node_indices, hyperedge_indices = torch.nonzero(adj, as_tuple=True)
                # 【修正点】使用 torch.stack 确保输出是 Tensor
                hypergraph = torch.stack([node_indices, hyperedge_indices])
                hyperedge_all.append(hypergraph)
            else:
                # 【修正点】确保返回空的 Tensor
                hyperedge_all.append(torch.empty((2, 0), dtype=torch.long, device=x.device))
        return hyperedge_all
    
# --- 模块3: 基于解剖学的空间池化模块 ---
class AnatomicalSpatialPooling(nn.Module):
    """
    基于预定义的解剖脑区进行空间池化。
    作用: 将一个精细尺度上的节点特征，根据预设的映射关系，聚合生成
          下一个更粗糙尺度上的“超级节点”特征。
    """
    def __init__(self, pre_defined_mapping):
        super().__init__()
        # 计算旧节点和新节点的数量
        if not pre_defined_mapping: num_old_nodes = 0
        else: num_old_nodes = max(max(indices) for indices in pre_defined_mapping if indices) + 1
        num_new_nodes = len(pre_defined_mapping)
        # 构建一个权重矩阵用于高效的矩阵乘法池化 (平均池化)
        weight_matrix = torch.zeros(num_old_nodes, num_new_nodes)
        for new_idx, old_indices in enumerate(pre_defined_mapping):
            if old_indices:
                for old_idx in old_indices:
                    weight_matrix[old_idx, new_idx] = 1.0 / len(old_indices)
        # 注册为 buffer，它会被模型保存和移动，但不会被视为可训练参数
        self.register_buffer('pooling_weight', weight_matrix)

    def forward(self, x):
        """
        输入 x: [批次大小, 旧节点数, 特征维度] (B, N_old, D)
        输出:   [批次大小, 新节点数, 特征维度] (B, N_new, D)
        """
        # 高效的矩阵乘法池化: (B, D, N_old) @ (N_old, N_new) -> (B, D, N_new)
        x_permuted = x.permute(0, 2, 1)
        pooled = torch.matmul(x_permuted, self.pooling_weight.to(x.device))
        return pooled.permute(0, 2, 1)
    
# --- 模块4: DHG超图卷积层 (优化版) ---
class HypergraphConvDHG(nn.Module):
    """
    一个复合的超图卷积层（经过批处理优化）。
    作用: 1. 对输入特征进行线性变换。
          2. 以批处理方式计算定制的NHC约束损失。
          3. 使用拼接后的“超级大图”高效执行超图卷积，避免for循环。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.conv = HGNNPConv(out_channels, out_channels)
            
    def _create_batched_hypergraph(self, hg_single, batch_size, num_nodes, device):
        """
        将单个超图结构复制B次，拼接成一个大的批处理超图。
        """
        e_list_batched = []
        # 获取单一样本的超边列表
        single_e_list = hg_single.e[0]
        
        # 遍历批次中的每个样本，为其创建带偏移的超边
        for i in range(batch_size):
            offset = i * num_nodes
            for edge in single_e_list:
                # 为每个节点索引添加偏移量
                offset_edge = [node + offset for node in edge]
                e_list_batched.append(offset_edge)
        
        # 使用新的节点总数和拼接后的超边列表创建批处理超图
        num_total_nodes = batch_size * num_nodes
        hg_batched = dhg.Hypergraph(num_total_nodes, e_list_batched, device=device)
        return hg_batched

    def forward(self, x, hg):
        """
        输入 x: [批次大小, 节点数, 输入特征维度] (B, N, C_in)
        输入 hg: 一个单一样本的DHG超图对象 (结构在批次内共享)
        输出:   (更新后的特征[B, N, C_out], 该批次的平均约束损失)
        """
        batch_size, num_nodes, _ = x.shape
        
        # 1. 初始特征变换 (在整个批次上高效完成)
        x_transformed_batched = self.lin(x) # -> [B, N, C_out]
        
        # 2. 将特征张量reshape以适应批处理
        # [B, N, C_out] -> [B * N, C_out]
        x_reshaped = x_transformed_batched.reshape(batch_size * num_nodes, self.out_channels)
        
        # 3. 创建批处理的“超级超图”
        hg_batched = self._create_batched_hypergraph(hg, batch_size, num_nodes, x.device)
        
        # 4. 以批处理方式计算 NHC 损失 (无循环)
        constrain_losstotal = torch.tensor(0.0, device=x.device)
        if hg_batched.num_e > 0:
            hyperedge_features = hg_batched.v2e(x_reshaped, aggr="sum")
            loss_hyper = 0
            if hg_batched.num_e > 1:
                # 计算超边间的相似度与距离
                sim = F.cosine_similarity(hyperedge_features.unsqueeze(1), hyperedge_features.unsqueeze(0), dim=-1)
                dist = torch.cdist(hyperedge_features, hyperedge_features, p=2)
                # 论文中的损失项
                loss_item = sim * dist + (1 - sim) * torch.clamp(4.2 - dist, min=0.0)
                loss_hyper = torch.mean(torch.abs(loss_item))
            
            # 节点重构损失
            node_features_from_hyperedges = hg_batched.e2v(hyperedge_features, aggr="mean")
            loss_node = torch.mean(torch.abs(x_reshaped - node_features_from_hyperedges))
            
            constrain_losstotal = loss_node + loss_hyper
        
        # 5. 在拼接后的大图上执行一次超图卷积 (无循环)
        out_reshaped = self.conv(x_reshaped, hg_batched)
        
        # 6. 将输出结果reshape回原来的批次维度
        # [B * N, C_out] -> [B, N, C_out]
        out_batched = out_reshaped.reshape(batch_size, num_nodes, self.out_channels)
        
        return out_batched, constrain_losstotal
    
# --- 模块5: 完整的主模型 ---
class EEG_AdaMSHyper(nn.Module):
    """
    用于EEG分类的自适应多尺度超图神经网络主模型。
    作用: 调度所有子模块，实现从原始EEG信号到最终分类结果的完整端到端流程。
    """
    def __init__(self, configs, pre_defined_mappings, num_domain_classes):
        super().__init__()
        self.scale_node_nums = configs.scale_node_nums
        
        # 实例化所有子模块
        # =======================【核心修改点】=======================
        # 使用新的 TemporalFeatureExtractor，并传入所需的额外参数
        self.feature_extractor = TemporalFeatureExtractor(
            num_channels=configs.num_channels, 
            d_model=configs.d_model,
            patch_size=configs.patch_size,
            time_sample_num=configs.time_sample_num
        )
        # ==========================================================
        
        self.hypergraph_builder = MultiScale_EEG_Hypergraph(configs)
        self.conv_layers = nn.ModuleList()
        self.spatial_pooling_layers = nn.ModuleList()
        
        # 为每个尺度创建一个卷积层
        for _ in range(len(self.scale_node_nums)):
            self.conv_layers.append(HypergraphConvDHG(configs.d_model, configs.d_model))
        # 根据提供的映射关系列表，创建相应数量的池化层 (n-1个)
        for mapping in pre_defined_mappings:
            self.spatial_pooling_layers.append(AnatomicalSpatialPooling(mapping))
        
        # 最终的分类器，融合所有尺度的信息
        fused_dim_for_classifier = configs.d_model * len(self.scale_node_nums)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim_for_classifier, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, configs.num_classes)
        )
        
        # DANN GRL 和领域分类器
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(fused_dim_for_classifier, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_domain_classes)
        )

    def forward(self, x, alpha=1.0):
        # 1. 提取初始节点特征
        current_features = self.feature_extractor(x)

        # 2. 一次性生成所有尺度的超图结构
        hypergraph_structures = self.hypergraph_builder(x)

        # 3. 层次化空间交互循环
        total_constraint_loss = 0
        final_scale_outputs = []

        for i in range(len(self.scale_node_nums)):
            # 获取当前尺度的结构
            hg_structure_tensor = hypergraph_structures[i]
            num_nodes_at_scale = self.scale_node_nums[i]
            
            # 将Tensor格式的超图结构转换为DHG库所需的e_list格式
            e_list = []
            if hg_structure_tensor.numel() > 0:
                num_hyperedges = int(hg_structure_tensor[1].max()) + 1
                e_list = [[] for _ in range(num_hyperedges)]
                for node_idx, he_idx in hg_structure_tensor.t().cpu().numpy():
                    e_list[he_idx].append(node_idx)
            
            # 使用转换后的 e_list 构建 DHG 对象
            hg = dhg.Hypergraph(num_nodes_at_scale, e_list, device=x.device)
            
            # 执行超图卷积
            updated_features, constraint_loss = self.conv_layers[i](current_features, hg)
            total_constraint_loss += constraint_loss
            
            # 保存当前尺度的图级表示 (通过平均池化) 用于最终融合
            graph_level_feature = updated_features.mean(dim=1)
            final_scale_outputs.append(graph_level_feature)
            
            # 如果不是最后一层，进行空间池化为下一层准备特征
            if i < len(self.scale_node_nums) - 1:
                current_features = self.spatial_pooling_layers[i](updated_features)

        # 4. 尺度间融合与分类
        fused_features = torch.cat(final_scale_outputs, dim=1)
        
        # -> 任务分类路径
        label_logits = self.classifier(fused_features)

        # -> 领域对抗路径
        reversed_features = self.grl(fused_features, alpha)
        domain_logits = self.domain_classifier(reversed_features)

        return label_logits, domain_logits, total_constraint_loss