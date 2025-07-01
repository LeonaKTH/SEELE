import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.event_proxy import prepare_event_proxy_tensor, prepare_event_proxy_tensor_v2

class RoleContrastLoss(nn.Module):
    """
    角色对比损失（L_rc）的实现
    
    该损失函数鼓励同一事件类型中不同事件实例的相同角色有相似的表示，
    同时不同角色有不同的表示。
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        """
        初始化角色对比损失
        
        参数:
            temperature: 温度参数，控制softmax的平滑度
            margin: 边界参数，控制不同角色之间的最小距离
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(self, event_proxy: torch.Tensor) -> torch.Tensor:
        """
        计算角色对比损失
        
        参数:
            event_proxy: 形状为(N, R, D)的张量，其中N是事件数，R是角色数，D是嵌入维度
            
        返回:
            loss: 角色对比损失值
        """
        # 如果没有事件或只有一个事件，返回零损失
        if event_proxy.size(0) <= 1:
            return torch.tensor(0.0, device=event_proxy.device)
        
        N, R, D = event_proxy.size()
        
        # 计算所有事件的所有角色之间的相似度矩阵
        # 将event_proxy重塑为(N*R, D)以便计算所有角色对之间的相似度
        flat_proxy = event_proxy.view(-1, D)  # (N*R, D)
        
        # 使用分块计算相似度矩阵以减少内存占用
        chunk_size = 128  # 可根据实际情况调整
        NR = flat_proxy.size(0)
        
        sim_matrix = torch.zeros(NR, NR, device=event_proxy.device)
        for i in range(0, NR, chunk_size):
            end_i = min(i + chunk_size, NR)
            chunk_i = flat_proxy[i:end_i]
            
            # 计算当前块与所有其他块的相似度
            for j in range(0, NR, chunk_size):
                end_j = min(j + chunk_size, NR)
                chunk_j = flat_proxy[j:end_j]
                sim_matrix[i:end_i, j:end_j] = torch.matmul(chunk_i, chunk_j.transpose(0, 1))
        
        sim_matrix = sim_matrix / self.temperature
        
        # 创建掩码，标识哪些角色对应于相同角色（正样本）
        # 对于索引(i,j)，如果i和j对应于不同事件的相同角色，则为1，否则为0
        role_mask = torch.zeros_like(sim_matrix)
        for r in range(R):
            # 获取所有事件的第r个角色
            idx_start = r
            idx_step = R
            indices = torch.arange(idx_start, N*R, idx_step, device=event_proxy.device)
            
            # 设置相同角色之间的掩码为1（不包括自身）
            for i, idx1 in enumerate(indices):
                for j, idx2 in enumerate(indices):
                    if i != j:  # 排除自身
                        role_mask[idx1, idx2] = 1.0
        
        # 创建有效角色掩码，排除零向量（未填充的角色）
        valid_mask = (event_proxy.norm(dim=-1) > 0).view(-1, 1)  # (N*R, 1)
        valid_mask = valid_mask & valid_mask.transpose(0, 1)  # (N*R, N*R)
        
        # 应用有效角色掩码
        role_mask = role_mask * valid_mask
        
        # 计算对比损失
        # 对于每个角色，将其与所有其他相同角色的相似度最大化，与不同角色的相似度最小化
        exp_sim = torch.exp(sim_matrix)
        
        # 对角线元素设为0（排除自身）
        exp_sim = exp_sim * (1 - torch.eye(N*R, device=exp_sim.device))
        
        # 找出难例（相似度高的负样本）
        with torch.no_grad():
            hard_negative_mask = torch.zeros_like(exp_sim)
            for i in range(exp_sim.size(0)):
                # 获取非同角色的样本
                neg_indices = torch.where((1 - role_mask[i]) * valid_mask[i] > 0)[0]
                if len(neg_indices) > 0:
                    # 选择相似度最高的k个负样本
                    k = min(5, len(neg_indices))  # 可以根据需要调整k值
                    hard_indices = torch.topk(exp_sim[i, neg_indices], k=k)[1]
                    hard_negative_mask[i, neg_indices[hard_indices]] = 1.0
        
        # 计算正样本的损失
        pos_sim = torch.sum(exp_sim * role_mask, dim=1)  # (N*R,)
        
        # 加权计算负样本损失，给难例更高的权重
        hard_neg_weight = 2.0  # 难例权重
        weighted_neg_sim = torch.sum(exp_sim * ((1 - role_mask) * valid_mask * 
                                              (1.0 + (hard_neg_weight - 1.0) * hard_negative_mask)), dim=1)
        
        # 有效角色的数量
        valid_count = valid_mask.sum(dim=1)  # (N*R,)
        valid_role_count = (role_mask * valid_mask).sum(dim=1)  # (N*R,)
        
        # 避免除以零
        valid_role_count = torch.clamp(valid_role_count, min=1.0)
        valid_count = torch.clamp(valid_count - valid_role_count, min=1.0)
        
        # 计算归一化的损失
        loss_per_role = -torch.log(pos_sim / valid_role_count) + torch.log(weighted_neg_sim / valid_count)
        
        # 只考虑有效角色的损失
        valid_roles = (valid_mask.sum(dim=1) > 0).float()  # (N*R,)
        loss = (loss_per_role * valid_roles).sum() / (valid_roles.sum() + 1e-10)
        
        return loss


def add_role_contrast_loss_to_model(model_outputs, batch, args, current_epoch=0, max_epochs=100):
    """
    在模型训练中添加角色对比损失
    
    参数:
        model_outputs: 模型的输出
        batch: 输入批次数据
        args: 模型参数
        current_epoch: 当前训练的轮次
        max_epochs: 总训练轮次
        
    返回:
        total_loss: 添加了角色对比损失的总损失
    """
    # 获取原始损失
    original_loss = model_outputs[0]
    
    # 如果没有标签或未启用角色对比损失，直接返回原始损失
    if batch.get('labels') is None or not hasattr(args, 'use_role_contrast') or not args.use_role_contrast:
        return original_loss
    
    # 获取模型输出
    aht_output = model_outputs[2][0]  # (argu_output, head_output, tail_output)
    last_hidden_state = model_outputs[2][1]  # 最后的隐藏状态
    
    # 准备event_proxy张量
    event_proxy = prepare_event_proxy_tensor_v2(
        batch_outputs=aht_output,
        last_hidden_state=last_hidden_state,
        offset_mappings=batch['offset_mapping'],
        args=args,
        threshold=0.0  # 可以根据需要调整阈值
    )
    
    # 动态调整权重（例如，随着训练进行逐渐增加权重）
    progress = current_epoch / max_epochs
    dynamic_weight = args.role_contrast_weight * min(1.0, progress * 2)  # 在前半程训练中逐渐增加权重
    
    # 计算角色对比损失
    role_contrast_loss_fn = RoleContrastLoss(
        temperature=args.role_contrast_temperature, 
        margin=args.role_contrast_margin
    )
    role_contrast_loss = role_contrast_loss_fn(event_proxy)
    
    # 更新总损失
    total_loss = original_loss + dynamic_weight * role_contrast_loss
    
    # 记录损失值（可以在训练循环中记录）
    loss_info = {
        'original_loss': original_loss.item(),
        'role_contrast_loss': role_contrast_loss.item(),
        'dynamic_weight': dynamic_weight,
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_info


class RoleContrastLossOriginalEPAL(nn.Module):
    """
    简化版角色对比损失，更贴近EPAL论文原始公式的实现
    
    基于EPAL论文Eq. 10的公式：
    - f(e_i, e_j) = Σ exp(sim(A_{i,u}, A_{j,u}) / τ) (正例对)
    - g(e_i, e_j) = Σ exp(sim(A_{i,u}, A_{j,u}) / τ) (负例对)
    - loss_pair = -log( f(e_i, e_j) / (f(e_i, e_j) + g(e_i, e_j)) )
    - L_rc = mean(loss_pair)
    
    该实现不包含难例挖掘和动态权重，专用于消融实验。
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        初始化简化版角色对比损失
        
        参数:
            temperature: 温度参数τ，控制softmax的平滑度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, event_proxy: torch.Tensor, entity_label: torch.Tensor) -> torch.Tensor:
        """
        计算角色对比损失
        
        参数:
            event_proxy: 形状为(batch_size, event_type_num, max_role_num, hidden_size)的张量
            entity_label: 形状为(batch_size, event_type_num, max_role_num)的张量，
                         标识每个角色位置对应的实体ID，相同ID表示同一实体
            
        返回:
            loss: 角色对比损失值
        """
        batch_size, event_type_num, max_role_num, hidden_size = event_proxy.shape
        
        # 如果批次大小小于2，无法计算对比损失
        if batch_size < 2:
            return torch.tensor(0.0, device=event_proxy.device)
        
        # 预计算掩码：all_same_mask[i, j, k] = True 当且仅当事件i和事件j的第k个角色是同一个实体
        # entity_label: (batch_size, event_type_num, max_role_num)
        # 扩展维度以便广播比较
        entity_i = entity_label.unsqueeze(1)  # (batch_size, 1, event_type_num, max_role_num)
        entity_j = entity_label.unsqueeze(0)  # (1, batch_size, event_type_num, max_role_num)
        
        # 计算同实体掩码，只有当两个位置都有有效实体（非零）且实体ID相同时才为True
        valid_i = (entity_i != 0)  # 有效实体掩码
        valid_j = (entity_j != 0)
        all_same_mask = (entity_i == entity_j) & valid_i & valid_j  # (batch_size, batch_size, event_type_num, max_role_num)
        
        # 排除自身比较（i == j的情况）
        self_mask = torch.eye(batch_size, device=event_proxy.device, dtype=torch.bool)
        self_mask = self_mask.unsqueeze(-1).unsqueeze(-1)  # (batch_size, batch_size, 1, 1)
        all_same_mask = all_same_mask & (~self_mask)
        
        total_loss = 0.0
        valid_pairs = 0
        
        # 遍历所有事件对(i, j)
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                
                # 获取事件对(i, j)的掩码
                same_mask_ij = all_same_mask[i, j]  # (event_type_num, max_role_num)
                
                # 检查是否有有效的角色对
                if not same_mask_ij.any():
                    continue
                
                # 计算正例对的相似度
                f_ij = 0.0
                for e in range(event_type_num):
                    for r in range(max_role_num):
                        if same_mask_ij[e, r]:  # 如果是正例对
                            # 获取角色嵌入
                            emb_i = event_proxy[i, e, r]  # (hidden_size,)
                            emb_j = event_proxy[j, e, r]  # (hidden_size,)
                            
                            # 计算余弦相似度
                            sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0), dim=1)
                            f_ij += torch.exp(sim / self.temperature)
                
                # 计算负例对的相似度
                g_ij = 0.0
                # 获取所有有效的角色位置（非零嵌入）
                valid_mask_i = (event_proxy[i].norm(dim=-1) > 1e-6)  # (event_type_num, max_role_num)
                valid_mask_j = (event_proxy[j].norm(dim=-1) > 1e-6)  # (event_type_num, max_role_num)
                
                for e in range(event_type_num):
                    for r in range(max_role_num):
                        if valid_mask_i[e, r] and valid_mask_j[e, r] and not same_mask_ij[e, r]:
                            # 如果是负例对（都有效但不是同一实体）
                            emb_i = event_proxy[i, e, r]
                            emb_j = event_proxy[j, e, r]
                            
                            # 计算余弦相似度
                            sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0), dim=1)
                            g_ij += torch.exp(sim / self.temperature)
                
                # 计算该事件对的损失
                if f_ij > 0 and (f_ij + g_ij) > 0:
                    loss_pair = -torch.log(f_ij / (f_ij + g_ij))
                    total_loss += loss_pair
                    valid_pairs += 1
        
        # 返回平均损失
        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=event_proxy.device)


# 在训练脚本中的使用示例
"""
# 在train.py的训练循环中添加以下代码

# 导入角色对比损失
from utils.role_contrast_loss import add_role_contrast_loss_to_model

# 模型前向传播
outputs = model(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    type_inputs_ids=batch['type_input_ids'],
    type_attention_mask=batch['type_attention_mask'],
    role_index_labels=batch['role_index_labels'],
    labels=batch['labels'],
    current_epoch_id=current_epoch
)

# 添加角色对比损失
loss = add_role_contrast_loss_to_model(
    model_outputs=outputs,
    batch=batch,
    args=args,
    role_contrast_weight=args.role_contrast_weight  # 从参数中获取权重
)

# 反向传播和优化器步骤
loss.backward()
optimizer.step()
"""