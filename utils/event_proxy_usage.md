# 角色对比损失（L_rc）的张量准备函数使用说明

## 函数概述

`prepare_event_proxy_tensor` 函数用于构建一个形状为 `(N, R, D)` 的 `event_proxy` 张量，其中包含模型预测事件中每个角色填充论元的嵌入向量。这个张量将作为后续计算角色对比损失（L_rc）的关键输入。

- `N`: 一个批次中模型预测出的事件记录总数
- `R`: 一个事件类型所包含的角色总数
- `D`: 论元表征的嵌入维度（例如，768）

## 函数版本说明

我们提供了两个版本的函数：

1. `prepare_event_proxy_tensor`: 使用 `postprocess_gplinker` 处理后的结果作为输入
2. `prepare_event_proxy_tensor_v2`: 直接使用模型的原始输出作为输入，不依赖 `postprocess_gplinker` 的处理结果

建议使用第二个版本，因为它更直接且效率更高。

## 调用位置

在训练循环中，应该在以下位置调用此函数：

```python
# 在 train.py 的训练循环中，模型前向传播之后，计算损失之前

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

# 获取模型输出
loss = outputs[0]  # 原始损失
loss_gp = outputs[1]  # GlobalPointer损失
aht_output = outputs[2][0]  # (argu_output, head_output, tail_output)
last_hidden_state = outputs[2][1]  # 最后的隐藏状态

# 准备event_proxy张量
event_proxy = prepare_event_proxy_tensor_v2(
    batch_outputs=aht_output,
    last_hidden_state=last_hidden_state,
    offset_mappings=batch['offset_mapping'],
    args=args,
    threshold=0.0  # 可以根据需要调整阈值
)

# 计算角色对比损失（这部分需要您自己实现）
role_contrast_loss = calculate_role_contrast_loss(event_proxy)

# 更新总损失
total_loss = loss + args.role_contrast_weight * role_contrast_loss
```

## 参数说明

### prepare_event_proxy_tensor_v2 函数参数

- `batch_outputs`: 模型forward方法返回的aht_output，包含(argu_output, head_output, tail_output)
- `last_hidden_state`: 模型编码器输出的隐藏状态，形状为(batch_size, seq_length, hidden_size)
- `offset_mappings`: 用于将token索引映射到原始文本位置
- `args`: 包含事件类型和角色信息的参数对象
- `threshold`: 预测分数阈值，默认为0

## 注意事项

1. 确保在调用函数前已经正确获取了模型的输出，特别是 `last_hidden_state` 和 `aht_output`
2. 函数会自动处理没有预测到事件的情况，返回空张量
3. 如果您的模型结构有变化，可能需要相应调整函数实现
4. 在计算角色对比损失时，需要考虑批次中事件数量可能为零的情况

## 实现细节

函数内部实现了以下步骤：

1. 从模型输出中提取论元预测
2. 构建事件链接关系
3. 析出完整事件
4. 为每个事件的每个角色提取对应的论元嵌入
5. 构建最终的event_proxy张量

这个实现充分利用了SEELE模型的输出结构，特别是GlobalPointer网络的预测结果和文本编码器的隐藏状态。