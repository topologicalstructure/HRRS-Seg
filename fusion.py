import torch


def get_fusion_weights(instance_masks):
    """
    Compute the fusion-weight matrix efficiently.
    :param instance_masks: Tensor [H, W], instance IDs (int)
    :return: fusion_weights: Tensor [H, W], float32
    """
    instance_masks = instance_masks.squeeze().squeeze()
    H, W = instance_masks.shape
    device = instance_masks.device

    # Flatten the mask for processing.
    flat_masks = instance_masks.view(-1)

    # Get the maximum ID to determine the lookup-table size.
    max_id = flat_masks.max().item()
    if max_id == 0:
        return torch.full((H, W), 0.5, device=device)

    # Generate and flatten coordinate grids.
    # coords_y: [H*W], coords_x: [H*W]
    y_range = torch.arange(H, device=device)
    x_range = torch.arange(W, device=device)
    coords_y, coords_x = torch.meshgrid(y_range, x_range, indexing='ij')
    flat_y = coords_y.reshape(-1)
    flat_x = coords_x.reshape(-1)

    # Filter background coordinates (ID=0) to reduce unnecessary work.
    mask_indices = flat_masks > 0
    fg_labels = flat_masks[mask_indices]
    fg_y = flat_y[mask_indices]
    fg_x = flat_x[mask_indices]

    # Use scatter_reduce to compute instance bounds by label.
    # Initialize minima with large values and maxima with small values.
    min_values = torch.full((max_id + 1,), float(H), device=device)
    max_values = torch.full((max_id + 1,), -1.0, device=device)

    # Calculate y_min, y_max, x_min, and x_max for each instance.
    y_min = min_values.clone().scatter_reduce(0, fg_labels.long(), fg_y.float(), reduce='amin', include_self=False)
    y_max = max_values.clone().scatter_reduce(0, fg_labels.long(), fg_y.float(), reduce='amax', include_self=False)
    x_min = min_values.clone().scatter_reduce(0, fg_labels.long(), fg_x.float(), reduce='amin', include_self=False)
    x_max = max_values.clone().scatter_reduce(0, fg_labels.long(), fg_x.float(), reduce='amax', include_self=False)

    # Calculate the square bounding-box size for each instance.
    # side = max(height, width) / H
    # Add 1 because pixel indices include both endpoints.
    inst_h = y_max - y_min + 1
    inst_w = x_max - x_min + 1
    inst_sides = torch.maximum(inst_h, inst_w)
    size_ratio = inst_sides / H

    # Calculate the weight lookup table; H is assumed to equal W.
    weight_lut1 = torch.clamp(2.0 - 4.0 * size_ratio, min=0.0, max=1.0)
    weight_lut3 = torch.clamp(2.0 * size_ratio - 1.0, min=0.0, max=1.0)

    # Set the background weight (ID=0) to 0.5.
    weight_lut1[0] = 0.5
    weight_lut3[0] = 0

    # Map the lookup values back to the original image shape.
    fusion_weight1 = weight_lut1[instance_masks]
    fusion_weight3 = weight_lut3[instance_masks]
    fusion_weight2 = torch.ones_like(fusion_weight1) - fusion_weight1 - fusion_weight3
    fusion_weights = torch.stack([fusion_weight1, fusion_weight2, fusion_weight3], dim=0)

    return fusion_weights


def scatter_mean_replace(image_features, instance_masks, back_region=False):
    """
    Replace image features with per-region mean features.
    :param image_features: (1, n, 896, 896)
    :param instance_masks: (1, 1, 896, 896)
    :param back_region: Whether to replace background features.
    :return: new_features: (1, n, 896, 896)
    """
    B, N, H, W = image_features.shape
    device = image_features.device

    # Reshape and flatten.
    # [N, H, W] -> [H*W, N]
    flat_feats = image_features.squeeze(0).permute(1, 2, 0).reshape(-1, N)
    # [1, 1, H, W] -> [H*W]
    flat_masks = instance_masks.view(-1).long()

    # Calculate the mean of each region using a lookup table.
    max_id = flat_masks.max().item()
    # Construct LUT: (max_id + 1, N).
    lut = torch.zeros((max_id + 1, N), dtype=image_features.dtype, device=device)

    # Prepare indices to match the feature dimension.
    target_idx = flat_masks.unsqueeze(1).expand(-1, N)

    # Build the mean lookup table with scatter_reduce.
    lut = lut.scatter_reduce(0, target_idx, flat_feats, reduce='mean', include_self=False)

    # Map values back to the original shape.
    # Map each flat mask ID to the corresponding mean feature vector.
    # mapped_feats shape: (H*W, N)
    mapped_feats = lut[flat_masks]

    # Handle background regions.
    if not back_region:
        # Preserve original features for background pixels.
        is_bg = (flat_masks == 0).unsqueeze(1)  # (H*W, 1)
        # Replace only foreground features.
        final_flat = torch.where(is_bg, flat_feats, mapped_feats)
    else:
        # Replace features for the whole image, including background.
        final_flat = mapped_feats

    # Restore shape: (H*W, N) -> (1, N, H, W).
    new_features = final_flat.view(H, W, N).permute(2, 0, 1).unsqueeze(0)

    return new_features


# Unit-test example.
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
    # Example with instances 1 and 2.
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