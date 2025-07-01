import torch
import numpy as np
from typing import List, Dict, Tuple, Any

def prepare_event_proxy_tensor(model_predictions, last_hidden_state, args):
    """
    构建一个形状为(N, R, D)的event_proxy张量，其中包含模型预测事件中每个角色填充论元的嵌入向量
    
    参数:
        model_predictions: 模型预测的事件记录列表，由postprocess_gplinker函数处理后的输出
                          每个事件是一个列表，包含多个论元，每个论元是[event_type, role, text, offset]格式
        last_hidden_state: 模型编码器输出的隐藏状态，形状为(batch_size, seq_length, hidden_size)
        args: 包含事件类型和角色信息的参数对象
        
    返回:
        event_proxy: 形状为(N, R, D)的张量，其中N是预测事件总数，R是每个事件类型的角色数，D是隐藏状态维度
    """
    # 获取设备信息
    device = last_hidden_state.device
    hidden_size = last_hidden_state.size(-1)
    
    # 计算预测的事件总数
    total_events = sum(len(events) for events in model_predictions)
    
    # 如果没有预测到事件，返回空张量
    if total_events == 0:
        return torch.zeros((0, args.max_role_num_within_one_event, hidden_size), device=device)
    
    # 创建event_proxy张量，初始化为零
    event_proxy = torch.zeros((total_events, args.max_role_num_within_one_event, hidden_size), device=device)
    
    # 创建事件类型到角色索引的映射
    event_type_to_roles = {}
    for i, event_type in enumerate(args.event_type_labels):
        roles = []
        for j, (t, r) in enumerate(args.labels):
            if t == event_type:
                roles.append((r, j))
        event_type_to_roles[event_type] = roles
    
    # 填充event_proxy张量
    event_idx = 0
    for batch_events in model_predictions:
        for event in batch_events:
            if not event:  # 跳过空事件
                continue
                
            event_type = event[0][0]  # 获取事件类型
            
            # 为每个角色找到对应的论元嵌入
            for role_name, role_idx in event_type_to_roles[event_type]:
                # 在当前事件中查找该角色的论元
                for arg in event:
                    if arg[1] == role_name:  # 找到匹配的角色
                        # 解析论元的起始和结束位置
                        start, end = map(int, arg[3].split(";"))
                        
                        # 计算论元的平均嵌入向量
                        batch_idx = model_predictions.index(batch_events)  # 获取批次索引
                        arg_embedding = last_hidden_state[batch_idx, start:end+1].mean(dim=0)
                        
                        # 将嵌入向量存储到event_proxy中
                        role_position = next(i for i, (r, _) in enumerate(event_type_to_roles[event_type]) if r == role_name)
                        event_proxy[event_idx, role_position] = arg_embedding
                        break
            
            event_idx += 1
    
    return event_proxy


def prepare_event_proxy_tensor_v2(batch_outputs, last_hidden_state, offset_mappings, args, threshold=0):
    """
    构建一个形状为(N, R, D)的event_proxy张量，直接从模型的原始输出构建，不依赖postprocess_gplinker的处理结果
    
    参数:
        batch_outputs: 模型forward方法返回的aht_output，包含(argu_output, head_output, tail_output)
        last_hidden_state: 模型编码器输出的隐藏状态，形状为(batch_size, seq_length, hidden_size)
        offset_mappings: 用于将token索引映射到原始文本位置
        args: 包含事件类型和角色信息的参数对象
        threshold: 预测分数阈值，默认为0
        
    返回:
        event_proxy: 形状为(N, R, D)的张量，其中N是预测事件总数，R是每个事件类型的角色数，D是隐藏状态维度
    """
    device = last_hidden_state.device
    hidden_size = last_hidden_state.size(-1)
    batch_size = last_hidden_state.size(0)
    
    # 创建事件类型到角色索引的映射
    event_type_to_roles = {}
    for i, event_type in enumerate(args.event_type_labels):
        roles = []
        for j, (t, r) in enumerate(args.labels):
            if t == event_type:
                roles.append((r, j))
        event_type_to_roles[event_type] = roles
    
    all_events = []
    all_event_embeddings = []
    
    for batch_idx in range(batch_size):
        argu_output = batch_outputs[0][batch_idx].cpu().numpy()
        head_output = batch_outputs[1][batch_idx].cpu().numpy()
        tail_output = batch_outputs[2][batch_idx].cpu().numpy()
        offset_mapping = offset_mappings[batch_idx]
        
        # 提取论元
        argus = set()
        argu_output[:, [0, -1]] -= np.inf
        argu_output[:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(argu_output > threshold)):
            argus.add((args.labels[l][0], args.labels[l][1], h, t))
        
        # 构建链接
        links = set()
        for i1, (e1, r1, h1, t1) in enumerate(argus):
            for i2, (e2, r2, h2, t2) in enumerate(argus):
                if i2 > i1 and e2 == e1:
                    if head_output[int(args.event_type_labels.index(e1)), min(h1, h2), max(h1, h2)] > threshold:
                        if tail_output[int(args.event_type_labels.index(e1)), min(t1, t2), max(t1, t2)] > threshold:
                            links.add((h1, t1, h2, t2))
                            links.add((h2, t2, h1, t1))
        
        # 析出事件
        batch_events = []
        for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
            for event in clique_search(list(sub_argus), links):
                event_type = event[0][0]  # 获取事件类型
                
                # 为每个事件创建角色嵌入张量
                event_embedding = torch.zeros(args.max_role_num_within_one_event, hidden_size, device=device)
                
                # 填充角色嵌入
                for role_name, role_idx in event_type_to_roles[event_type]:
                    # 在当前事件中查找该角色的论元
                    for arg in event:
                        if arg[1] == role_name:  # 找到匹配的角色
                            h, t = arg[2], arg[3]  # 获取头尾位置
                            
                            # 计算论元的平均嵌入向量
                            arg_embedding = last_hidden_state[batch_idx, h:t+1].mean(dim=0)
                            
                            # 将嵌入向量存储到event_embedding中
                            role_position = next(i for i, (r, _) in enumerate(event_type_to_roles[event_type]) if r == role_name)
                            event_embedding[role_position] = arg_embedding
                            break
                
                batch_events.append(event)
                all_event_embeddings.append(event_embedding)
        
        all_events.extend(batch_events)
    
    # 如果没有预测到事件，返回空张量
    if not all_events:
        return torch.zeros((0, args.max_role_num_within_one_event, hidden_size), device=device)
    
    # 将所有事件的嵌入堆叠成一个张量
    event_proxy = torch.stack(all_event_embeddings)
    
    return event_proxy


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    from utils.postprocess import DedupList, neighbors
    
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]