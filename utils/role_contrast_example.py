import torch
import os
from utils.role_contrast_loss import add_role_contrast_loss_to_model
from utils.visualization import visualize_role_embeddings, analyze_role_contrast_loss


def train_with_role_contrast(model, train_dataloader, optimizer, scheduler, args):
    """
    使用角色对比损失的训练循环示例
    
    参数:
        model: 模型
        train_dataloader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        args: 模型参数
    """
    model.train()
    total_loss = 0
    
    # 创建可视化输出目录
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 记录损失信息
    loss_records = {
        'original_loss': [],
        'role_contrast_loss': [],
        'dynamic_weight': [],
        'total_loss': []
    }
    
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            # 将数据移动到设备上
            batch = {k: v.to(args.my_device) for k, v in batch.items()}
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 模型前向传播
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                type_inputs_ids=batch['type_input_ids'],
                type_attention_mask=batch['type_attention_mask'],
                role_index_labels=batch['role_index_labels'],
                labels=batch['labels'],
                current_epoch_id=epoch
            )
            
            # 根据参数决定是否添加角色对比损失
            if args.use_role_contrast:
                loss, loss_info = add_role_contrast_loss_to_model(
                    model_outputs=outputs,
                    batch=batch,
                    args=args,
                    current_epoch=epoch,
                    max_epochs=args.num_train_epochs
                )
                
                # 记录损失信息
                for k, v in loss_info.items():
                    loss_records[k].append(v)
                    
                # 每隔一定步数打印损失信息
                if step % args.logging_steps == 0:
                    print(f"Step {step}: Original Loss = {loss_info['original_loss']:.4f}, "
                          f"Role Contrast Loss = {loss_info['role_contrast_loss']:.4f}, "
                          f"Weight = {loss_info['dynamic_weight']:.4f}, "
                          f"Total Loss = {loss_info['total_loss']:.4f}")
            else:
                loss = outputs[0]
            
            # 反向传播和优化器步骤
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # 每个epoch结束后可视化角色嵌入
        if args.use_role_contrast and (epoch + 1) % 5 == 0:  # 每5个epoch可视化一次
            # 获取验证集上的角色嵌入
            event_proxy, event_types, role_names = get_role_embeddings(model, val_dataloader, args)
            
            # 可视化角色嵌入
            visualize_role_embeddings(
                event_proxy=event_proxy,
                event_types=event_types,
                role_names=role_names,
                output_dir=os.path.join(vis_dir, f"epoch_{epoch+1}")
            )
            
            # 分析角色对比损失
            stats = analyze_role_contrast_loss(event_proxy, args)
            print(f"Epoch {epoch+1} Role Contrast Analysis:")
            for k, v in stats.items():
                print(f"  {k}: {v:.4f}")
    
    # 绘制损失曲线
    if args.use_role_contrast:
        plot_loss_curves(loss_records, vis_dir)
    
    return total_loss / len(train_dataloader)


def get_role_embeddings(model, dataloader, args):
    """
    获取角色嵌入
    
    参数:
        model: 模型
        dataloader: 数据加载器
        args: 模型参数
        
    返回:
        event_proxy: 角色嵌入张量
        event_types: 事件类型列表
        role_names: 角色名称列表
    """
    model.eval()
    all_event_proxies = []
    all_event_types = []
    all_role_names = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 将数据移动到设备上
            batch = {k: v.to(args.my_device) for k, v in batch.items()}
            
            # 模型前向传播
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                type_inputs_ids=batch['type_input_ids'],
                type_attention_mask=batch['type_attention_mask']
            )
            
            # 获取模型输出
            aht_output = outputs[2][0]  # (argu_output, head_output, tail_output)
            last_hidden_state = outputs[2][1]  # 最后的隐藏状态
            
            # 准备event_proxy张量
            from utils.event_proxy import prepare_event_proxy_tensor_v2
            event_proxy = prepare_event_proxy_tensor_v2(
                batch_outputs=aht_output,
                last_hidden_state=last_hidden_state,
                offset_mappings=batch['offset_mapping'],
                args=args,
                threshold=0.0
            )
            
            # 收集事件类型和角色名称
            # 基于FNDEE_schema.json的事件类型和角色映射
            event_type_names = ["Experiment", "Manoeuvre", "Deploy", "Support", "Accident", "Exhibit", "Conflict", "Injure"]
            
            # 角色名称映射（基于FNDEE_schema.json，每个事件类型的角色列表）
            role_names_by_event = {
                0: ["触发词", "Subject", "Equipment", "Date", "Location"],  # Experiment
                1: ["触发词", "Subject", "Date", "Area", "Content"],        # Manoeuvre
                2: ["触发词", "Subject", "Militaryforce", "Date", "Location"], # Deploy
                3: ["触发词", "Subject", "Object", "Date", "Materials"],    # Support
                4: ["触发词", "Subject", "Result", "Date", "Location"],     # Accident
                5: ["触发词", "Subject", "Equipment", "Date", "Location"],  # Exhibit
                6: ["触发词", "Subject", "Object", "Date", "Location"],     # Conflict
                7: ["触发词", "Subject", "Quantity", "Date", "Location"]    # Injure
            }
            
            batch_event_types = []
            batch_role_names = []
            
            # 从event_proxy的维度获取事件类型和角色信息
            batch_size, event_type_num, max_role_num, hidden_size = event_proxy.shape
            
            for b in range(batch_size):
                for e in range(event_type_num):
                    for r in range(max_role_num):
                        # 检查该位置是否有有效的角色嵌入（非零向量）
                        embedding = event_proxy[b, e, r]
                        if torch.norm(embedding) > 1e-6:  # 非零嵌入
                            # 获取真实的事件类型和角色名称
                            event_type = event_type_names[e] if e < len(event_type_names) else f"Event_{e}"
                            
                            # 获取该事件类型对应的角色名称
                            if e in role_names_by_event and r < len(role_names_by_event[e]):
                                role_name = role_names_by_event[e][r]
                            else:
                                role_name = f"Role_{r}"
                            
                            batch_event_types.append(event_type)
                            batch_role_names.append(role_name)
            
            all_event_proxies.append(event_proxy)
            all_event_types.extend(batch_event_types)
            all_role_names.extend(batch_role_names)
    
    # 合并所有batch的结果
    if all_event_proxies:
        event_proxy = torch.cat(all_event_proxies, dim=0)
    else:
        # 如果没有事件，返回空张量
        event_proxy = torch.zeros((0, args.max_role_num_within_one_event, 768), device=args.my_device)
    
    return event_proxy, all_event_types, all_role_names


def plot_loss_curves(loss_records, output_dir):
    """
    绘制损失曲线
    
    参数:
        loss_records: 损失记录字典
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # 绘制原始损失和总损失
    plt.subplot(2, 1, 1)
    plt.plot(loss_records['original_loss'], label='Original Loss')
    plt.plot(loss_records['total_loss'], label='Total Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # 绘制角色对比损失和权重
    plt.subplot(2, 1, 2)
    plt.plot(loss_records['role_contrast_loss'], label='Role Contrast Loss')
    plt.plot(loss_records['dynamic_weight'], label='Dynamic Weight')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Role Contrast Loss and Weight')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.close()