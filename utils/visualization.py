import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_role_embeddings(event_proxy, event_types, role_names, output_dir):
    """
    可视化角色嵌入
    
    参数:
        event_proxy: 形状为(N, R, D)的张量，其中N是事件数，R是角色数，D是嵌入维度
        event_types: 长度为N的列表，包含每个事件的类型
        role_names: 长度为N*R的列表，包含每个角色的名称
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 将角色嵌入展平为2D张量
    N, R, D = event_proxy.size()
    flat_embeddings = event_proxy.reshape(-1, D).cpu().detach().numpy()
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(flat_embeddings)
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(set(role_names))))
    role_to_color = {role: colors[i] for i, role in enumerate(set(role_names))}
    
    for i, (emb, event_type, role) in enumerate(zip(embeddings_2d, event_types, role_names)):
        plt.scatter(emb[0], emb[1], color=role_to_color[role], label=role if role not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(emb[0], emb[1], f"{event_type}-{role}", fontsize=8)
    
    plt.legend()
    plt.title("Role Embeddings Visualization")
    plt.savefig(os.path.join(output_dir, "role_embeddings.png"))
    plt.close()

def analyze_role_contrast_loss(event_proxy, args):
    """
    分析角色对比损失
    
    参数:
        event_proxy: 形状为(N, R, D)的张量，其中N是事件数，R是角色数，D是嵌入维度
        args: 模型参数
        
    返回:
        stats: 包含相似度统计信息的字典
    """
    N, R, D = event_proxy.size()
    flat_proxy = event_proxy.view(-1, D)
    
    # 计算所有角色对之间的余弦相似度
    cos_sim = torch.nn.CosineSimilarity(dim=-1)
    sim_matrix = torch.zeros(N*R, N*R, device=event_proxy.device)
    
    for i in range(N*R):
        for j in range(N*R):
            if i != j:  # 排除自身
                sim_matrix[i, j] = cos_sim(flat_proxy[i].unsqueeze(0), flat_proxy[j].unsqueeze(0))
    
    # 创建角色掩码
    role_mask = torch.zeros_like(sim_matrix)
    for r in range(R):
        indices = torch.arange(r, N*R, R, device=event_proxy.device)
        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices):
                if i != j:  # 排除自身
                    role_mask[idx1, idx2] = 1.0
    
    # 创建有效角色掩码，排除零向量（未填充的角色）
    valid_mask = (event_proxy.norm(dim=-1) > 0).view(-1, 1)  # (N*R, 1)
    valid_mask = valid_mask & valid_mask.transpose(0, 1)  # (N*R, N*R)
    
    # 计算正样本和负样本的相似度分布
    pos_sim = sim_matrix[role_mask > 0]
    neg_sim = sim_matrix[(role_mask == 0) & (sim_matrix != 0) & (valid_mask > 0)]
    
    # 计算统计信息
    stats = {
        "pos_sim_mean": pos_sim.mean().item() if len(pos_sim) > 0 else 0,
        "pos_sim_std": pos_sim.std().item() if len(pos_sim) > 0 else 0,
        "neg_sim_mean": neg_sim.mean().item() if len(neg_sim) > 0 else 0,
        "neg_sim_std": neg_sim.std().item() if len(neg_sim) > 0 else 0,
        "pos_sim_min": pos_sim.min().item() if len(pos_sim) > 0 else 0,
        "pos_sim_max": pos_sim.max().item() if len(pos_sim) > 0 else 0,
        "neg_sim_min": neg_sim.min().item() if len(neg_sim) > 0 else 0,
        "neg_sim_max": neg_sim.max().item() if len(neg_sim) > 0 else 0,
    }
    
    # 绘制相似度分布直方图
    plt.figure(figsize=(10, 6))
    
    if len(pos_sim) > 0:
        plt.hist(pos_sim.cpu().numpy(), bins=30, alpha=0.5, label='Positive Pairs', color='green')
    if len(neg_sim) > 0:
        plt.hist(neg_sim.cpu().numpy(), bins=30, alpha=0.5, label='Negative Pairs', color='red')
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Role Embedding Similarities')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "similarity_distribution.png"))
    plt.close()
    
    return stats