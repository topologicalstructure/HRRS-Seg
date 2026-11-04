import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.analysis import get_model_complexity_info
from mmseg.registry import MODELS
import segearth_segmentor


class SinglePatchWrapper(nn.Module):
    """只追踪单个 224x224 patch 的 forward_feature，避免追踪 49 次循环导致 OOM"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        # 转换输入类型为 half (fp16)
        img = img.half()
        # 直接调用单块的前向特征提取与上采样
        return self.model.forward_feature(img)


def compute_num_crops(h_img, w_img, h_crop=224, w_crop=224, h_stride=112, w_stride=112):
    """计算滑动窗口的切片数量"""
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    return h_grids * w_grids, (h_grids, w_grids)


def format_flops(flops):
    if flops >= 1e12:
        return f"{flops / 1e12:.5f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    else:
        return f"{flops:.2f} FLOPs"


def main():
    config_path = 'configs/cfg_openearthmap.py'
    print(f"Loading config from {config_path}...")
    cfg = Config.fromfile(config_path)

    # 1. 临时关闭 scales_fusion，避免加载 SAM2 (可节省数 GB 显存)
    # cfg.model.scales_fusion = False

    print("Building model (without SAM2 memory overhead)...")
    model = MODELS.build(cfg.model)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # 2. 包装模型为单 patch 推理
    wrapped_model = SinglePatchWrapper(model)

    # 单个 patch 尺寸
    patch_shape = (3, 224, 224)
    # 完整测试图像尺寸
    full_image_size = (896, 896)
    stride = 112
    crop_size = 224

    total_crops, (hg, wg) = compute_num_crops(
        full_image_size[0], full_image_size[1],
        crop_size, crop_size,
        stride, stride
    )

    if cfg.model.scales_fusion:
        crops_scale_2, (hg2, wg2) = compute_num_crops(
            full_image_size[0] // 2, full_image_size[1] // 2,
            crop_size, crop_size,
            stride, stride
        )
        total_crops += crops_scale_2

    print(f"Single Crop Shape:  {patch_shape}")
    print(f"Full Image Shape:   (3, {full_image_size[0]}, {full_image_size[1]})")
    if not cfg.model.scales_fusion:
        print(f"Sliding Grid:       {hg} x {wg} = {total_crops} crops")
    else:
        print(f"Sliding Grid:       {hg} x {wg} + {hg2} x {wg2} = {total_crops} crops")

    # 3. 计算单个 patch 的复杂度
    analysis_results = get_model_complexity_info(
        wrapped_model,
        input_shape=patch_shape,
        show_table=False,
        show_arch=False
    )

    single_patch_flops = analysis_results['flops']
    total_flops = single_patch_flops * total_crops

    print("\n" + "=" * 50)
    print(f"Single Patch FLOPs: {analysis_results['flops_str']}")
    print(f"Total Crops Count:  {total_crops}")
    print(f"Total Image FLOPs:  {format_flops(total_flops)}")
    print(f"Model Parameters:   {analysis_results['params_str']}")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()