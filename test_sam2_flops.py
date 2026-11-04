import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from sam2.build_sam import build_sam2


def main():
    # 统一在 CPU 上进行 FLOPs 分析，避免设备不匹配与类型冲突
    device = "cpu"
    p1, p2, p3 = 32, 0.8, 0.8  # 填入你的 sam_parameters (points_per_side = p1)

    print(f"Building SAM2 model on {device}...")
    sam2 = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device=device, apply_postprocessing=False)

    sam2 = sam2.float().to(device)
    sam2.eval()

    # 1. 参数量计算
    total_params = sum(p.numel() for p in sam2.parameters())
    print(f"SAM2 Total Parameters: {total_params / 1e6:.2f} M")

    # 2. FLOPs 计算 (标准 1024x1024 输入)
    dummy_img = torch.randn(1, 3, 1024, 1024, device=device, dtype=torch.float32)

    # (1) 计算 Image Encoder FLOPs
    print("Calculating Image Encoder FLOPs...")
    encoder_flops = FlopCountAnalysis(sam2.image_encoder, dummy_img).total()

    # (2) 提取特征图并使用多级降级机制解析（防止变量未定义）
    with torch.no_grad():
        if hasattr(sam2, "forward_image"):
            backbone_out = sam2.forward_image(dummy_img)
        else:
            backbone_out = sam2.image_encoder(dummy_img)

        # 预先声明变量，防止 UnboundLocalError
        pix_feat = None
        image_pe = None
        high_res_feats = None

        # 尝试 1：通过 SAM2 原生的 _prepare_backbone_features
        if hasattr(sam2, "_prepare_backbone_features"):
            try:
                prep_res = sam2._prepare_backbone_features(backbone_out)
                if isinstance(prep_res, (tuple, list)):
                    if len(prep_res) == 3:
                        pix_feat, image_pe, high_res_feats = prep_res
                    elif len(prep_res) == 2:
                        pix_feat, high_res_feats = prep_res
            except Exception:
                pass

        # 尝试 2：如果尝试 1 未能成功提取，进行手动结构化解析
        if high_res_feats is None or pix_feat is None:
            if isinstance(backbone_out, dict):
                fpn = backbone_out.get("backbone_fpn", [])
                pix_feat = backbone_out.get("vision_features", fpn[-1] if len(fpn) > 0 else None)
                image_pe = backbone_out.get("vision_pos_enc", None)

                # 处理高分辨率特征的卷积投影
                if hasattr(sam2, "conv_s0") and hasattr(sam2, "conv_s1") and len(fpn) >= 2:
                    high_res_feats = [sam2.conv_s0(fpn[0]), sam2.conv_s1(fpn[1])]
                else:
                    high_res_feats = fpn[:-1] if len(fpn) > 1 else []
            elif isinstance(backbone_out, (list, tuple)):
                pix_feat = backbone_out[-1]
                high_res_feats = list(backbone_out[:-1])

        # 规范化 image_pe 格式
        if isinstance(image_pe, (list, tuple)) and len(image_pe) > 0:
            image_pe = image_pe[-1]

    # (3) 封装 Prompt Encoder + Mask Decoder 追踪单点 FLOPs
    class SAM2DecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, high_res_feats, pix_feat, image_pe, point_coords, point_labels):
            sparse_emb, dense_emb = self.model.sam_prompt_encoder(
                points=(point_coords, point_labels), boxes=None, masks=None
            )

            # 如果 position embedding 缺失，动态生成
            if image_pe is None or not isinstance(image_pe, torch.Tensor):
                image_pe = self.model.sam_prompt_encoder.get_dense_pe()

            return self.model.sam_mask_decoder(
                image_embeddings=pix_feat,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )

    dummy_point = torch.zeros(1, 1, 2, device=device, dtype=torch.float32)
    dummy_label = torch.ones(1, 1, device=device, dtype=torch.int32)
    decoder_wrapper = SAM2DecoderWrapper(sam2)

    print("Calculating Mask Decoder FLOPs...")
    single_point_flops = FlopCountAnalysis(
        decoder_wrapper, (high_res_feats, pix_feat, image_pe, dummy_point, dummy_label)
    ).total()

    # (4) 汇总统计
    total_points = p1 * p1
    total_decoder_flops = total_points * single_point_flops
    total_sam2_flops = encoder_flops + total_decoder_flops

    print("\n" + "=" * 50)
    print(f"Image Encoder FLOPs:        {encoder_flops / 1e9:.2f} GFLOPs")
    print(f"Single Point Decoder FLOPs: {single_point_flops / 1e6:.2f} MFLOPs")
    print(f"Total Points (p1 * p1):     {total_points} ({p1}x{p1})")
    print(f"Total Decoder FLOPs:        {total_decoder_flops / 1e9:.2f} GFLOPs")
    print(f"Total SAM2 FLOPs per Image: {total_sam2_flops / 1e9:.2f} GFLOPs")
    print("=" * 50)


if __name__ == '__main__':
    main()