import torch


def get_fusion_weights(instance_masks):
    """
    高效计算融合权重矩阵
    :param instance_masks: Tensor [H, W], 元素为实例ID (int)
    :return: fusion_weights: Tensor [H, W], float32
    """
    instance_masks = instance_masks.squeeze().squeeze()
    H, W = instance_masks.shape
    device = instance_masks.device

    # 1. 扁平化 mask 方便处理
    flat_masks = instance_masks.view(-1)

    # 获取最大 ID，用于确定 lookup table 的大小
    max_id = flat_masks.max().item()
    if max_id == 0:
        return torch.full((H, W), 0.5, device=device)

    # 2. 生成坐标网格并扁平化
    # coords_y: [H*W], coords_x: [H*W]
    y_range = torch.arange(H, device=device)
    x_range = torch.arange(W, device=device)
    coords_y, coords_x = torch.meshgrid(y_range, x_range, indexing='ij')
    flat_y = coords_y.reshape(-1)
    flat_x = coords_x.reshape(-1)

    # 3. 过滤掉背景 (ID=0) 的坐标，减少不必要的计算
    # 虽然 scatter_reduce 也可以处理，但过滤可以提速并避免背景影响 bounds
    mask_indices = flat_masks > 0
    fg_labels = flat_masks[mask_indices]
    fg_y = flat_y[mask_indices]
    fg_x = flat_x[mask_indices]

    # 4. 使用 scatter_reduce 计算每个实例的极值 (Group-by)
    # 初始化：min 填最大值，max 填最小值
    # 我们需要 max_id + 1 的长度来容纳所有 ID
    min_values = torch.full((max_id + 1,), float(H), device=device)
    max_values = torch.full((max_id + 1,), -1.0, device=device)

    # 计算各实例的 y_min, y_max, x_min, x_max
    y_min = min_values.clone().scatter_reduce(0, fg_labels.long(), fg_y.float(), reduce='amin', include_self=False)
    y_max = max_values.clone().scatter_reduce(0, fg_labels.long(), fg_y.float(), reduce='amax', include_self=False)
    x_min = min_values.clone().scatter_reduce(0, fg_labels.long(), fg_x.float(), reduce='amin', include_self=False)
    x_max = max_values.clone().scatter_reduce(0, fg_labels.long(), fg_x.float(), reduce='amax', include_self=False)

    # 5. 计算每个实例的正方形包围盒尺寸
    # side = max(height, width) / H
    # +1 是因为像素索引包含起始和终止
    inst_h = y_max - y_min + 1
    inst_w = x_max - x_min + 1
    inst_sides = torch.maximum(inst_h, inst_w)
    size_ratio = inst_sides / H

    # 计算权重表 (index i 对应实例 i 的权重)
    # 注意：整张图像尺寸假定为 H (正方形图像 H=W)
    weight_lut1 = torch.clamp(2.0 - 4.0 * size_ratio, min=0.0, max=1.0)
    weight_lut3 = torch.clamp(2.0 * size_ratio - 1.0, min=0.0, max=1.0)

    # 6. 处理背景：将 ID 0 的权重设为 0.5
    weight_lut1[0] = 0.5
    weight_lut3[0] = 0

    # 7. 通过索引映射回原图形状 (Gather 操作)
    fusion_weight1 = weight_lut1[instance_masks]
    fusion_weight3 = weight_lut3[instance_masks]
    fusion_weight2 = torch.ones_like(fusion_weight1) - fusion_weight1 - fusion_weight3
    fusion_weights = torch.stack([fusion_weight1, fusion_weight2, fusion_weight3], dim=0)

    return fusion_weights


def scatter_mean_replace(image_features, instance_masks, back_region=False):
    """
    用区域均值替换原特征图中的像素特征
    :param image_features: (1, n, 896, 896)
    :param instance_masks: (1, 1, 896, 896)
    :param back_region: bool, 是否处理背景区域
    :return: new_features: (1, n, 896, 896)
    """
    B, N, H, W = image_features.shape
    device = image_features.device

    # 1. 重塑与展平
    # [N, H, W] -> [H*W, N]
    flat_feats = image_features.squeeze(0).permute(1, 2, 0).reshape(-1, N)
    # [1, 1, H, W] -> [H*W]
    flat_masks = instance_masks.view(-1).long()

    # 2. 计算每个区域的均值 (作为 Lookup Table)
    max_id = flat_masks.max().item()
    # 构造 LUT: (max_id + 1, N)
    lut = torch.zeros((max_id + 1, N), dtype=image_features.dtype, device=device)

    # 准备索引以匹配特征维度
    target_idx = flat_masks.unsqueeze(1).expand(-1, N)

    # 使用 scatter_reduce 直接得到所有 ID 的均值表
    # reduce='mean' 自动处理了计数和求和
    lut = lut.scatter_reduce(0, target_idx, flat_feats, reduce='mean', include_self=False)

    # 3. 映射回原图 (Broadcasting Indexing)
    # 这一步是核心：通过 flat_masks 里的 ID 直接从 lut 中取对应的均值向量
    # mapped_feats shape: (H*W, N)
    mapped_feats = lut[flat_masks]

    # 4. 背景处理逻辑
    if not back_region:
        # 如果不处理背景，则将 mask 为 0 的位置替换回原始特征
        # 创建背景掩码
        is_bg = (flat_masks == 0).unsqueeze(1)  # (H*W, 1)
        # 仅在非背景区域使用均值，背景保留原值
        final_flat = torch.where(is_bg, flat_feats, mapped_feats)
    else:
        # 整个图（包含背景）都使用各自区域的均值
        final_flat = mapped_feats

    # 5. 还原形状: (H*W, N) -> (1, N, H, W)
    new_features = final_flat.view(H, W, N).permute(2, 0, 1).unsqueeze(0)

    return new_features


# --- 测试代码 ---
# if __name__ == "__main__":
#     masks = torch.tensor([
#         [1, 2, 3, 0],
#         [4, 2, 3, 0],
#         [4, 5, 5, 0],
#         [0, 0, 0, 0]
#     ], dtype=torch.int32).unsqueeze(0).unsqueeze(0)
#     features = torch.randn((1,2,4,4))
#     print(features.permute(0,2,3,1))
#     features = scatter_mean_replace(features, masks, back_region=False)
#     print(features.permute(0, 2, 3, 1))
if __name__ == "__main__":
    # 假设 4x4 图像，有两个实例 1 和 2
    masks = torch.tensor([
        [1, 1, 0, 0, 3],
        [1, 1, 0, 0, 4],
        [0, 0, 2, 2, 0],
        [0, 0, 2, 0, 0],
        [3, 0, 2, 0, 2]
    ], dtype=torch.int64)

    weights = get_fusion_weights(masks)
    print("Instance Masks:\n", masks)
    print("Fusion Weights:\n", weights)