# 角色对比损失（Role Contrast Loss）使用说明

## 简介

角色对比损失（Role Contrast Loss）是一种针对事件抽取任务的特殊损失函数，旨在优化模型对不同事件中相同角色的表示学习。该损失函数鼓励同一事件类型中不同事件实例的相同角色有相似的表示，同时不同角色有不同的表示，从而提高模型对事件角色的识别能力。

## 功能特点

1. **分块矩阵乘法优化**：通过分块计算相似度矩阵，减少内存占用，支持处理更大批量的数据。
2. **难例挖掘机制**：自动识别相似度最高的负样本，并赋予其更高的权重，提高模型对难区分角色的学习能力。
3. **动态权重调整**：根据训练进度自动调整角色对比损失的权重，在训练初期减小权重，训练后期增大权重。
4. **角色嵌入可视化**：提供角色嵌入的t-SNE可视化功能，帮助理解模型学习的角色表示。
5. **相似度分布分析**：分析正负样本的相似度分布，辅助超参数调整。

## 参数配置

在训练时，可以通过以下命令行参数控制角色对比损失：

```bash
python train.py \
    --use_role_contrast True \
    --role_contrast_weight 0.1 \
    --role_contrast_temperature 0.1 \
    --role_contrast_margin 0.5
```

参数说明：

- `use_role_contrast`：是否启用角色对比损失，默认为False
- `role_contrast_weight`：角色对比损失的权重，默认为0.1
- `role_contrast_temperature`：温度参数，控制相似度分布的平滑程度，默认为0.1
- `role_contrast_margin`：边界参数，控制正负样本之间的距离，默认为0.5

## 使用示例

### 基本使用

只需在训练命令中添加`--use_role_contrast True`参数即可启用角色对比损失：

```bash
python train.py \
    --model_type bert \
    --my_device cuda:0 \
    --pretrained_model_name_or_path ./PLMs \
    --num_train_epochs 100 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 16 \
    --use_role_contrast True
```

### 高级配置

可以通过调整温度和边界参数来优化角色对比损失的效果：

```bash
python train.py \
    --model_type bert \
    --my_device cuda:0 \
    --pretrained_model_name_or_path ./PLMs \
    --num_train_epochs 100 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 16 \
    --use_role_contrast True \
    --role_contrast_weight 0.2 \
    --role_contrast_temperature 0.05 \
    --role_contrast_margin 0.3
```

## 可视化结果

训练过程中，角色对比损失模块会自动生成以下可视化结果，保存在`output_dir/visualizations`目录下：

1. **角色嵌入可视化**：`role_embeddings.png`，展示不同角色嵌入在二维空间的分布。
2. **相似度分布直方图**：`similarity_distribution.png`，展示正负样本相似度的分布情况。
3. **损失曲线**：`loss_curves.png`，展示训练过程中各类损失的变化趋势。

## 注意事项

1. 角色对比损失需要足够多的事件实例才能发挥作用，对于小数据集可能效果有限。
2. 温度参数(`role_contrast_temperature`)对损失的影响较大，建议在[0.05, 0.2]范围内调整。
3. 权重参数(`role_contrast_weight`)建议根据数据集大小和模型性能调整，一般在[0.05, 0.5]范围内。
4. 如果训练过程中出现梯度爆炸，可以尝试降低权重参数或增大温度参数。