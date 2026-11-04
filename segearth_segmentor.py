import os.path

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
from torchvision import transforms
import numpy as np
from PIL import Image

import torchvision

sys.path.append("..")

from prompts.imagenet_template import *

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F

from open_clip import tokenizer, create_model
from BLIP.models.blip_retrieval import blip_retrieval
import gem
from simfeatup_dev.upsamplers import get_upsampler

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

@MODELS.register_module()
class SegEarthSegmentation(BaseSegmentor):
    def __init__(self,
                 clip_type,
                 vit_type,
                 model_type,
                 name_path,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 prob_thd=0.0,
                 logit_scale=50,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0,
                 bg_idx=0,
                 feature_up=True,
                 scales_fusion=True,
                 sam_parameters=[16, 0.2, 0.5],
                 fusion_parameters=[0.125, 0.875, 1, 0.25, 0.875],
                 geometric_correction=True,
                 scale_normalize=True,
                 sub_confi_fusion=True,
                 experience_scale=True,
                 aggregation_method='vote',
                 geometric_parameters=[0.0, 1.0, 1.0, 0.0],
                 border_width=0.125,
                 paint_feature = False,
                 no_confidence = False,
                 feature_up_cfg=dict(
                     model_name='jbu_one',
                     model_path='your/model/path')):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='openai', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='openai', precision='fp16')
        elif clip_type == 'RemoteCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RemoteCLIP-ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RemoteCLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'GeoRSCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RS5M_ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RS5M_ViT-L-14.pt', precision='fp16')
            elif 'H' in vit_type:
                self.net = create_model('ViT-H-14', pretrained='checkpoint/RS5M_ViT-H-14.pt', precision='fp16')
        elif clip_type == 'SkyCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', \
                                        pretrained='checkpoint/SkyCLIP_ViT_B32_top50pct/epoch_20.pt', \
                                        precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', \
                                        pretrained='checkpoint/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt', \
                                        precision='fp16')
        elif clip_type == 'OpenCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='laion2b_s34b_b88k', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')
        elif clip_type == 'MetaCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B-16-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L/14-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
        elif clip_type == 'BLIP':
            if 'B' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_base_14M.pth', image_size=slide_crop, vit='base')
            elif 'L' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_large.pth', image_size=slide_crop, vit='large')
            self.net = self.net.half()
        elif clip_type == 'ALIP':
            self.net = create_model('ViT-B/32', pretrained='checkpoint/ALIP_YFCC15M_B32.pt', precision='fp16')

        if model_type == 'GEM':
            if 'B' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'laion2b_s34b_b88k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-B/16-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            elif 'L' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'laion2b_s32b_b82k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-L-14-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            self.net = self.net.model

        self.net.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.feature_up = feature_up
        self.cls_token_lambda = cls_token_lambda
        self.output_cls_token = cls_token_lambda != 0
        self.bg_idx = bg_idx

        if self.clip_type == 'BLIP':
            self.patch_size = self.net.visual_encoder.patch_size
        else:
            self.patch_size = self.net.visual.patch_size

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad(): # sub_imagenet_template, openai_imagenet_template
            for qw in query_words:
                if self.clip_type == 'BLIP':
                    query =self.net.tokenizer([temp(qw) for temp in openai_imagenet_template], padding='max_length',
                                           truncation=True, max_length=35,
                                           return_tensors="pt").to(device)
                    text_output = self.net.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                                        mode='text')
                    feature = F.normalize(self.net.text_proj(text_output.last_hidden_state[:, 0, :]))
                else:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.scales_fusion = scales_fusion
        self.fusion_parameters = fusion_parameters
        self.geometric_correction = geometric_correction
        self.scale_normalize = scale_normalize
        self.sub_confi_fusion = sub_confi_fusion
        self.aggregation_method = aggregation_method
        self.experience_scale = experience_scale
        self.geometric_parameters = geometric_parameters
        self.border_width = border_width
        self.no_confidence = no_confidence

        self.paint_feature = paint_feature
        self.paint_path = './middle'
        self.device = device

        if self.paint_feature and (not os.path.exists(self.paint_path)):
            os.makedirs(self.paint_path)

        if scales_fusion:
            sam2 = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device=self.device,
                              apply_postprocessing=False)
            sam2 = sam2.half()
            p1, p2, p3 = sam_parameters
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=p1,
                pred_iou_thresh=p2,
                stability_score_thresh=p3,
                multimask_output=False,
            )
        if feature_up:
            self.feat_dim = self.query_features.shape[-1]
            self.upsampler = get_upsampler(feature_up_cfg['model_name'], self.feat_dim).cuda().half()
            self.feature_upsampler = get_upsampler(feature_up_cfg['model_name'], 768).cuda().half()
            ckpt = torch.load(feature_up_cfg['model_path'])['state_dict']
            weights_dict = {k[10:]: v for k, v in ckpt.items()}
            self.upsampler.load_state_dict(weights_dict, strict=True)


    @staticmethod
    def build_size_bin_region_map(instance_masks: torch.Tensor, bins):
        """Build per-pixel size-bin map from instance id mask.

        Args:
            instance_masks: [B, 1, H, W], integer instance ids, 0 is background.
            bins: ascending boundaries in [0, 1], e.g. [0.0, 0.2, 0.5, 1.0].
        Returns:
            Tensor [B, 1, H, W], each pixel stores interval index in [0, n_bins-1].
        """
        if instance_masks.dim() != 4:
            raise ValueError('instance_masks must be [B,1,H,W]')
        if len(bins) < 2:
            raise ValueError('bins must contain at least 2 boundaries')
        if bins[0] != 0.0 or bins[-1] != 1.0:
            raise ValueError('bins must start at 0.0 and end at 1.0')
        if any(bins[i] > bins[i + 1] for i in range(len(bins) - 1)):
            raise ValueError('bins must be non-decreasing')
        if any(x < 0.0 or x > 1.0 for x in bins):
            raise ValueError('all boundaries in bins must be within [0,1]')

        B, C, H, W = instance_masks.shape
        if C != 1:
            raise ValueError('instance_masks channel dimension must be 1')

        out = torch.full((B, 1, H, W), -1, dtype=torch.long, device=instance_masks.device)
        bin_edges = torch.tensor(bins, device=instance_masks.device, dtype=torch.float32)
        n_bins = len(bins) - 1

        for b in range(B):
            m = instance_masks[b, 0].long()
            flat = m.view(-1)
            max_id = int(flat.max().item())
            if max_id <= 0:
                continue

            y_range = torch.arange(H, device=instance_masks.device)
            x_range = torch.arange(W, device=instance_masks.device)
            coords_y, coords_x = torch.meshgrid(y_range, x_range, indexing='ij')
            flat_y = coords_y.reshape(-1)
            flat_x = coords_x.reshape(-1)

            fg_indices = flat > 0
            fg_labels = flat[fg_indices]
            fg_y = flat_y[fg_indices]
            fg_x = flat_x[fg_indices]

            min_values = torch.full((max_id + 1,), float(H), device=instance_masks.device)
            max_values = torch.full((max_id + 1,), -1.0, device=instance_masks.device)
            y_min = min_values.clone().scatter_reduce(0, fg_labels.long(), fg_y.float(), reduce='amin', include_self=False)
            y_max = max_values.clone().scatter_reduce(0, fg_labels.long(), fg_y.float(), reduce='amax', include_self=False)
            x_min = min_values.clone().scatter_reduce(0, fg_labels.long(), fg_x.float(), reduce='amin', include_self=False)
            x_max = max_values.clone().scatter_reduce(0, fg_labels.long(), fg_x.float(), reduce='amax', include_self=False)

            inst_h = y_max - y_min + 1
            inst_w = x_max - x_min + 1
            inst_sides = torch.maximum(inst_h, inst_w)
            size_ratio = inst_sides / float(H)

            inst_bin = torch.bucketize(size_ratio, bin_edges[1:], right=True).clamp(0, n_bins - 1)
            out[b, 0, m > 0] = inst_bin[m[m > 0]]

        return out

    @staticmethod
    def build_confidence_bin_region_map(confidence: torch.Tensor, bins):
        """Build per-pixel confidence-bin map from confidence map.

        Args:
            confidence: [B, 1, H, W].
            bins: ascending boundaries in [0, 1], e.g. [0.0, 0.2, 0.5, 1.0].
        Returns:
            Tensor [B, 1, H, W], each pixel stores interval index in [0, n_bins-1].
        """
        if confidence.dim() != 4:
            raise ValueError('confidence must be [B,1,H,W]')
        if confidence.shape[1] != 1:
            raise ValueError('confidence channel dimension must be 1')
        if len(bins) < 2:
            raise ValueError('bins must contain at least 2 boundaries')
        if bins[0] != 0.0 or bins[-1] != 1.0:
            raise ValueError('bins must start at 0.0 and end at 1.0')
        if any(bins[i] > bins[i + 1] for i in range(len(bins) - 1)):
            raise ValueError('bins must be non-decreasing')
        if any(x < 0.0 or x > 1.0 for x in bins):
            raise ValueError('all boundaries in bins must be within [0,1]')

        conf = confidence.float().clamp(0.0, 1.0)
        edges = torch.tensor(bins[1:], device=conf.device, dtype=conf.dtype)
        n_bins = len(bins) - 1
        out = torch.bucketize(conf, edges, right=True).clamp(0, n_bins - 1).long()
        return out

    def generate_mask(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            masks = self.mask_generator.generate(image)
        instance_mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
        if len(masks) != 0:
            sorted_anns = sorted(masks, key=(lambda x: x['area']))  # predicted_iou
            instance_id = 1
            for ann in sorted_anns:
                m = ann['segmentation']
                instance_mask[m] = instance_id
                instance_id += 1
        # plt.axis('off')
        # plt.imshow(instance_mask, cmap="viridis")
        # plt.savefig('mask.png')
        instance_mask = torch.from_numpy(instance_mask).to(self.device)
        return instance_mask

    def get_fusion_weights(self, instance_masks):
        """
        compute the fusion weights based on the instance masks
        instance_masks : [1,1,H,W]
        """
        # get parameters
        x1, y1, x2, y2, back = self.fusion_parameters
        instance_masks = instance_masks.squeeze().squeeze()
        H, W = instance_masks.shape
        device = instance_masks.device

        # flattened the masks and get max id
        flat_masks = instance_masks.view(-1)
        max_id = flat_masks.max().item()
        if max_id == 0:
            return torch.full((2, H, W), back, device=device)

        # generate flattened coordinates
        # coords_y: [H*W], coords_x: [H*W]
        y_range = torch.arange(H, device=device)
        x_range = torch.arange(W, device=device)
        coords_y, coords_x = torch.meshgrid(y_range, x_range, indexing='ij')
        flat_y = coords_y.reshape(-1)
        flat_x = coords_x.reshape(-1)

        # exclude the coordinates of the background
        mask_indices = flat_masks > 0
        fg_labels = flat_masks[mask_indices]
        fg_y = flat_y[mask_indices]
        fg_x = flat_x[mask_indices]

        # use scatter_reduce to calculate the coordinate extrema for each index
        min_values = torch.full((max_id + 1,), float(H), device=device)
        max_values = torch.full((max_id + 1,), -1.0, device=device)

        y_min = min_values.clone().scatter_reduce(0, fg_labels.long(), fg_y.float(), reduce='amin', include_self=False)
        y_max = max_values.clone().scatter_reduce(0, fg_labels.long(), fg_y.float(), reduce='amax', include_self=False)
        x_min = min_values.clone().scatter_reduce(0, fg_labels.long(), fg_x.float(), reduce='amin', include_self=False)
        x_max = max_values.clone().scatter_reduce(0, fg_labels.long(), fg_x.float(), reduce='amax', include_self=False)

        # compute the bounding box sizes and weights
        inst_h = y_max - y_min + 1
        inst_w = x_max - x_min + 1
        inst_sides = torch.maximum(inst_h, inst_w)
        size_ratio = inst_sides / H

        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        weight_lut1 = torch.clamp(k * size_ratio + b, min=y2, max=y1)
        weight_lut1[0] = back

        # map back to the original shape using indexes
        fusion_weight1 = weight_lut1[instance_masks]
        fusion_weight2 = torch.ones_like(fusion_weight1) - fusion_weight1
        fusion_weights = torch.stack([fusion_weight1, fusion_weight2], dim=0)
        return fusion_weights

    def get_border_mask(self, instance_masks):
        """
        get the border within each instance mask except background
        instance_masks : (1,1,H,W)
        """
        _, N, H, W = instance_masks.shape
        border_width = int(self.border_width * H)

        # extracting region boundaries
        # to exclude image edges, use replicate padding
        m = instance_masks.float()
        m_pad = F.pad(m, (1, 1, 1, 1), mode='replicate')

        is_border = (
                (m != m_pad[:, :, 1:-1, :-2]) |  # left
                (m != m_pad[:, :, 1:-1, 2:]) |  # right
                (m != m_pad[:, :, :-2, 1:-1]) |  # up
                (m != m_pad[:, :, 2:, 1:-1])  # down
        )

        # max pooling is used to determine the pixels within a region that are a distance of border_width from the boundary
        if border_width > 1:
            kernel_size = 2 * border_width - 1
            near_border_mask = F.max_pool2d(
                is_border.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=border_width - 1
            ).bool()
        else:
            near_border_mask = is_border

        calc_mask = near_border_mask
        # calc_mask = calc_mask & (instance_masks > 0)
        return calc_mask

    def scatter_mean_replace_border_aware(self, image_features, instance_masks, back_region=False):
        """
        calculate the feature mean within the border_width range of the region boundary, and replace the entire region
        image_features : (1,n,H,W)
        instance_masks : (1,1,H,W)
        """
        B, N, H, W = image_features.shape
        device = image_features.device
        border_width = int(self.border_width * H)

        # extracting region boundaries
        # to exclude image edges, use replicate padding
        m = instance_masks.float()
        m_pad = F.pad(m, (1, 1, 1, 1), mode='replicate')

        is_border = (
                (m != m_pad[:, :, 1:-1, :-2]) |  # left
                (m != m_pad[:, :, 1:-1, 2:]) |  # right
                (m != m_pad[:, :, :-2, 1:-1]) |  # up
                (m != m_pad[:, :, 2:, 1:-1])  # down
        )

        # max pooling is used to determine the pixels within a region that are a distance of border_width from the boundary
        if border_width > 1:
            kernel_size = 2 * border_width - 1
            near_border_mask = F.max_pool2d(
                is_border.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=border_width - 1
            ).bool()
        else:
            near_border_mask = is_border

        calc_mask = near_border_mask
        if not back_region:
            calc_mask = calc_mask & (instance_masks > 0)

        if self.paint_feature:
            plt.imsave(os.path.join(self.paint_path, 'border_mask.png'), calc_mask.squeeze().squeeze().cpu().numpy(),
                       cmap='viridis')

        flat_masks = instance_masks.view(-1).long()
        flat_feats = image_features.squeeze(0).permute(1, 2, 0).reshape(-1, N)
        flat_calc_mask = calc_mask.view(-1)

        # use scatter_reduce to calculate the mean
        max_id = flat_masks.max().item()
        lut = torch.zeros((max_id + 1, N), dtype=image_features.dtype, device=device)

        valid_feats = flat_feats[flat_calc_mask]
        valid_ids = flat_masks[flat_calc_mask]

        if valid_ids.numel() > 0:
            target_idx = valid_ids.unsqueeze(1).expand(-1, N)
            lut = lut.scatter_reduce(0, target_idx, valid_feats, reduce='mean', include_self=False)

        # map back
        mapped_feats = lut[flat_masks]

        if not back_region:
            is_bg = (flat_masks == 0).unsqueeze(1)
            final_flat = torch.where(is_bg, flat_feats, mapped_feats)
        else:
            final_flat = mapped_feats

        new_features = final_flat.view(H, W, N).permute(2, 0, 1).unsqueeze(0)
        return new_features

    def scatter_mean_replace_confidence_aware(self, image_features, instance_masks, confidence):
        """
        Replace each foreground region with its confidence-weighted mean feature.
        - image_features: [B, N, H, W]
        - instance_masks : [B, 1, H, W], 0 is background
        - confidence     : [B, 1, H, W]

        Background pixels (mask==0) keep original features.
        """
        if image_features.dim() != 4 or instance_masks.dim() != 4 or confidence.dim() != 4:
            raise ValueError("Expected 4D tensors for image_features, instance_masks, confidence")

        B, N, H, W = image_features.shape
        if instance_masks.shape != (B, 1, H, W) or confidence.shape != (B, 1, H, W):
            raise ValueError(
                f"Shape mismatch: image_features={tuple(image_features.shape)}, "
                f"instance_masks={tuple(instance_masks.shape)}, confidence={tuple(confidence.shape)}"
            )

        device = image_features.device
        feat_dtype = image_features.dtype
        acc_dtype = torch.float32 if feat_dtype in (torch.float16, torch.bfloat16) else feat_dtype

        HW = H * W

        # Flatten to [B*HW, ...] for batched scatter accumulation.
        flat_feats = image_features.permute(0, 2, 3, 1).reshape(B * HW, N).to(acc_dtype)
        flat_masks = instance_masks.view(B, HW).long()
        flat_conf = confidence.view(B, HW).to(acc_dtype)

        # Build globally unique region ids per batch via offset.
        max_region_id = int(flat_masks.max().item())
        if max_region_id <= 0:
            return image_features

        stride = max_region_id + 1
        batch_offsets = (torch.arange(B, device=device).unsqueeze(1) * stride)
        global_ids = flat_masks + batch_offsets  # [B, HW]
        global_ids_flat = global_ids.reshape(-1)

        fg_mask = (flat_masks > 0).reshape(-1)
        if not fg_mask.any():
            return image_features

        fg_ids = global_ids_flat[fg_mask]
        fg_feats = flat_feats[fg_mask]
        fg_conf = flat_conf.reshape(-1, 1)[fg_mask]

        total_regions = B * stride

        # Weighted sums / weights for confidence-aware averaging.
        weighted_sum = torch.zeros((total_regions, N), device=device, dtype=acc_dtype)
        weight_sum = torch.zeros((total_regions, 1), device=device, dtype=acc_dtype)
        weighted_sum.index_add_(0, fg_ids, fg_feats * fg_conf)
        weight_sum.index_add_(0, fg_ids, fg_conf)

        # Fallback plain mean for regions whose confidence sum is zero.
        plain_sum = torch.zeros((total_regions, N), device=device, dtype=acc_dtype)
        plain_cnt = torch.zeros((total_regions, 1), device=device, dtype=acc_dtype)
        plain_sum.index_add_(0, fg_ids, fg_feats)
        plain_cnt.index_add_(0, fg_ids, torch.ones_like(fg_conf, dtype=acc_dtype))

        weighted_mean = weighted_sum / torch.clamp(weight_sum, min=1e-9)
        plain_mean = plain_sum / torch.clamp(plain_cnt, min=1.0)
        region_mean = torch.where(weight_sum > 0, weighted_mean, plain_mean).to(feat_dtype)

        mapped_feats = region_mean[global_ids_flat].view(B * HW, N)

        # Keep background unchanged, replace only foreground pixels.
        bg_mask = (flat_masks.reshape(-1) == 0).unsqueeze(1)
        orig_feats = image_features.permute(0, 2, 3, 1).reshape(B * HW, N)
        final_flat = torch.where(bg_mask, orig_feats, mapped_feats)

        return final_flat.view(B, H, W, N).permute(0, 3, 1, 2).contiguous()

    def scatter_vote_replace_confidence_aware(self, seg_pred, instance_masks, confidence):
        """
        Replace each foreground region with its confidence-voted predicted label.
        - seg_pred: [B, H, W]
        - instance_masks : [B, 1, H, W], 0 is background
        - confidence     : [B, 1, H, W]

        Background pixels (mask==0) keep original predicted label.
        Return processed seg_pred: [B, H, W]
        """
        if seg_pred.dim() != 3 or instance_masks.dim() != 4 or confidence.dim() != 4:
            raise ValueError("Expected seg_pred [B,H,W], instance_masks/confidence [B,1,H,W]")

        B, H, W = seg_pred.shape
        if instance_masks.shape != (B, 1, H, W) or confidence.shape != (B, 1, H, W):
            raise ValueError(
                f"Shape mismatch: seg_pred={tuple(seg_pred.shape)}, "
                f"instance_masks={tuple(instance_masks.shape)}, confidence={tuple(confidence.shape)}"
            )

        device = seg_pred.device
        pred_dtype = seg_pred.dtype
        acc_dtype = torch.float32

        HW = H * W
        flat_pred = seg_pred.reshape(B, HW)
        flat_masks = instance_masks.view(B, HW).long()
        flat_conf = confidence.view(B, HW).to(acc_dtype)

        max_region_id = int(flat_masks.max().item())
        if max_region_id <= 0:
            return seg_pred

        max_class_id = int(flat_pred.max().item())
        num_classes = max_class_id + 1 if max_class_id >= 0 else 0
        if num_classes <= 0:
            return seg_pred

        stride = max_region_id + 1
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * stride
        global_ids = (flat_masks + batch_offsets).reshape(-1)

        fg_mask = (flat_masks > 0).reshape(-1)
        if not fg_mask.any():
            return seg_pred

        fg_ids = global_ids[fg_mask]
        fg_classes = flat_pred.reshape(-1)[fg_mask].long()
        fg_conf = flat_conf.reshape(-1)[fg_mask]

        total_regions = B * stride
        total_bins = total_regions * num_classes

        # Weighted vote per (region, class).
        weighted_votes = torch.zeros((total_bins,), device=device, dtype=acc_dtype)
        vote_idx = fg_ids * num_classes + fg_classes
        weighted_votes.index_add_(0, vote_idx, fg_conf)
        weighted_votes = weighted_votes.view(total_regions, num_classes)

        # Fallback to unweighted vote when all confidences are zero in a region.
        plain_votes = torch.zeros((total_bins,), device=device, dtype=acc_dtype)
        plain_votes.index_add_(0, vote_idx, torch.ones_like(fg_conf, dtype=acc_dtype))
        plain_votes = plain_votes.view(total_regions, num_classes)

        weighted_sum = weighted_votes.sum(dim=1, keepdim=True)
        weighted_choice = weighted_votes.argmax(dim=1)
        plain_choice = plain_votes.argmax(dim=1)
        region_choice = torch.where(weighted_sum.squeeze(1) > 0, weighted_choice, plain_choice).to(pred_dtype)

        mapped_pred = region_choice[global_ids].view(B, HW)
        bg_mask = flat_masks == 0
        final_pred = torch.where(bg_mask, flat_pred, mapped_pred)

        return final_pred.view(B, H, W)

    def scatter_vote_replace_border_aware(self, seg_pred, instance_masks):
        """
        Replace each foreground region with its voted predicted label within the border_width range of the region boundary.
        - seg_pred: [B, H, W]
        - instance_masks : [B, 1, H, W], 0 is background

        Background pixels (mask==0) keep original predicted label.
        Return processed seg_pred: [B, H, W]
        """
        if seg_pred.dim() != 3 or instance_masks.dim() != 4:
            raise ValueError("Expected seg_pred [B,H,W], instance_masks [B,1,H,W]")

        B, H, W = seg_pred.shape
        if instance_masks.shape != (B, 1, H, W):
            raise ValueError(
                f"Shape mismatch: seg_pred={tuple(seg_pred.shape)}, instance_masks={tuple(instance_masks.shape)}"
            )

        device = seg_pred.device
        pred_dtype = seg_pred.dtype
        acc_dtype = torch.float32

        border_width = int(self.border_width * H)

        # extract region boundaries from instance masks (replicate padding to avoid edge effects)
        m = instance_masks.float()
        m_pad = F.pad(m, (1, 1, 1, 1), mode='replicate')

        is_border = (
            (m != m_pad[:, :, 1:-1, :-2]) |  # left
            (m != m_pad[:, :, 1:-1, 2:]) |  # right
            (m != m_pad[:, :, :-2, 1:-1]) |  # up
            (m != m_pad[:, :, 2:, 1:-1])    # down
        )

        # pixels within `border_width` distance from the region boundary
        if border_width > 1:
            kernel_size = 2 * border_width - 1
            near_border_mask = F.max_pool2d(
                is_border.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=border_width - 1
            ).bool()
        else:
            near_border_mask = is_border

        # only consider foreground regions
        calc_mask = near_border_mask & (instance_masks > 0)

        # Flatten for batched scatter operations
        HW = H * W
        flat_masks = instance_masks.view(B, HW).long()
        flat_pred = seg_pred.view(B, HW).long()
        flat_calc = calc_mask.view(B, HW).reshape(-1)

        max_region_id = int(flat_masks.max().item())
        if max_region_id <= 0:
            return seg_pred

        max_class_id = int(flat_pred.max().item())
        num_classes = max_class_id + 1 if max_class_id >= 0 else 0
        if num_classes <= 0:
            return seg_pred

        stride = max_region_id + 1
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * stride
        global_ids = (flat_masks + batch_offsets).reshape(-1)

        # votes from border-area pixels (may be empty for some regions)
        fg_border_mask = flat_calc & (flat_masks.view(-1) > 0)
        if not fg_border_mask.any():
            # no border votes available for any region -> fallback to no-op
            return seg_pred

        fg_border_ids = global_ids[fg_border_mask]
        fg_border_classes = flat_pred.view(-1)[fg_border_mask].long()

        total_regions = B * stride
        total_bins = total_regions * num_classes

        # count votes from border pixels
        border_votes = torch.zeros((total_bins,), device=device, dtype=acc_dtype)
        border_idx = fg_border_ids * num_classes + fg_border_classes
        border_votes.index_add_(0, border_idx, torch.ones_like(fg_border_classes, dtype=acc_dtype))
        border_votes = border_votes.view(total_regions, num_classes)

        # fallback: plain vote using all pixels inside the region (if border votes are zero)
        all_fg_mask = (flat_masks.view(-1) > 0)
        fg_all_ids = global_ids[all_fg_mask]
        fg_all_classes = flat_pred.view(-1)[all_fg_mask].long()
        plain_votes = torch.zeros((total_bins,), device=device, dtype=acc_dtype)
        plain_idx = fg_all_ids * num_classes + fg_all_classes
        plain_votes.index_add_(0, plain_idx, torch.ones_like(fg_all_classes, dtype=acc_dtype))
        plain_votes = plain_votes.view(total_regions, num_classes)

        border_sum = border_votes.sum(dim=1)
        border_choice = border_votes.argmax(dim=1)
        plain_choice = plain_votes.argmax(dim=1)

        region_choice = torch.where(border_sum > 0, border_choice, plain_choice).to(pred_dtype)

        mapped_pred = region_choice[global_ids].view(B, HW)

        bg_mask = flat_masks == 0
        final_pred = torch.where(bg_mask, flat_pred, mapped_pred)

        return final_pred.view(B, H, W)

    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, self.output_cls_token)
            
        if self.output_cls_token:
            image_cls_token, image_features = image_features
            image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            cls_logits = image_cls_token @ self.query_features.T

        # featup
        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            # [1, 196, 512] -> [1, 512, 14, 14]
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            # [1, 512, 14, 14] -> [1, 512, 224, 224]
            with torch.cuda.amp.autocast():
                image_features = self.upsampler(image_features, img).half()
            # [1, 512, 224, 224] -> [1, 50176, 512]
            image_features = image_features.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        if self.output_cls_token:
            logits = logits + cls_logits * self.cls_token_lambda

            # # CLIP Surgery
            # # weights to restrain influence of obvious classes on others
            # prob = (cls_logits * 2).softmax(-1)
            # w = prob / prob.mean(-1, keepdim=True)
            # # element-wise multiplied features
            # b, n_t, n_i, c = image_features.shape[0], self.query_features.shape[0], image_features.shape[1], image_features.shape[2]
            # feats = image_features.reshape(b, n_i, 1, c) * self.query_features.reshape(1, 1, n_t, c)
            # feats *= w.reshape(1, 1, n_t, 1)
            # redundant_feats = feats.mean(2, keepdim=True) # along cls dim
            # feats = feats - redundant_feats
            # # sum the element-wise multiplied features as cosine similarity
            # logits = feats.sum(-1)

        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        return logits

    def ori_forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    def forward_slide(self, img, masks, img_metas, stride=112, crop_size=224, upsample=True):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds_weighted = img.new_zeros((batch_size, out_channels, h_img, w_img))
        preds_plain = img.new_zeros((batch_size, out_channels, h_img, w_img))
        confidence_weighted = img.new_zeros((batch_size, 1, h_img, w_img))
        confidence_plain = img.new_zeros((batch_size, 1, h_img, w_img))
        weight_sum = img.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_mask = masks[:, :, y1:y2, x1:x2]
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img)
                # compute confidence on the same (unpadded) spatial size as crop_mask
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]
                if self.no_confidence:
                    crop_confidence = torch.ones_like(crop_mask)
                else:
                    crop_confidence = self.compute_confidence(crop_mask, crop_seg_logit)
                if self.paint_feature:
                    plt.axis('off')
                    plt.imsave(os.path.join(self.paint_path, str(h_img) + '_' + str(h_idx) + '_' + str(w_idx) + '_confidence.png'),
                               crop_confidence.squeeze().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                    # plt.imsave(os.path.join(self.paint_path, str(h_img) + '_' + str(h_idx) + '_' + str(w_idx) + '_feature.png'),
                    #            logit_visual.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                    img_visual = crop_img.squeeze()
                    mean = [0.48145466, 0.4578275, 0.40821073]
                    std = [0.26862954, 0.26130258, 0.27577711]
                    inv_normalize = transforms.Normalize(
                        mean=[-m / s for m, s in zip(mean, std)],
                        std=[1 / s for s in std]
                    )
                    img_visual = inv_normalize(img_visual).permute(1, 2, 0)
                    plt.imsave(os.path.join(self.paint_path, str(h_img) + '_' + str(h_idx) + '_' + str(w_idx) + '_img.png'),
                               img_visual.cpu().numpy())
                    # mask_visual = crop_mask.squeeze()
                    # plt.imsave(
                    #     os.path.join(self.paint_path, str(h_img) + '_' + str(h_idx) + '_' + str(w_idx) + '_mask.png'),
                    #     mask_visual.cpu().numpy(),cmap='viridis')
                padded_seg_logit = nn.functional.pad(crop_seg_logit,
                                                      (int(x1), int(w_img - x2), int(y1),
                                                       int(h_img - y2)))
                padded_confidence = nn.functional.pad(crop_confidence,
                                                       (int(x1), int(w_img - x2), int(y1),
                                                        int(h_img - y2)))
                if self.sub_confi_fusion:
                    preds_weighted += padded_seg_logit * padded_confidence
                    preds_plain += padded_seg_logit
                    confidence_weighted += padded_confidence * padded_confidence
                    confidence_plain += padded_confidence
                    weight_sum += padded_confidence
                else:
                    preds_plain += padded_seg_logit
                    confidence_plain += padded_confidence

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        plain_pred = preds_plain / count_mat
        plain_confidence = confidence_plain / count_mat
        if self.sub_confi_fusion:
            weighted_pred = preds_weighted / weight_sum.clamp_min(1e-9)
            weighted_confidence = confidence_weighted / weight_sum.clamp_min(1e-9)

            valid_weight_mask = weight_sum > 0
            preds = torch.where(valid_weight_mask.expand_as(weighted_pred), weighted_pred, plain_pred)
            confidence = torch.where(valid_weight_mask, weighted_confidence, plain_confidence)
        else:
            preds = plain_pred
            confidence = plain_confidence
        if self.scale_normalize:
            con_min = confidence.min()
            con_max = confidence.max()
            confidence = (confidence - con_min) / (con_max - con_min + 1e-9)
        if self.paint_feature:
            plt.axis('off')
            plt.imsave(os.path.join(self.paint_path, str(h_img) + '_' + 'confidence.png'),
                       confidence.squeeze().cpu().numpy(), cmap='viridis')
            plt.imsave(os.path.join(self.paint_path, str(h_img) + '_' + 'preds.png'),
                       torch.max(preds, dim=1)[0].squeeze().cpu().numpy(), cmap='viridis')
        img_size = img_metas[0]['ori_shape'][:2]
        if upsample:
            logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')
            confidence = nn.functional.interpolate(confidence, size=img_size, mode='bilinear')

        return logits, confidence

    def compute_confidence(self, seg_mask, crop_seg_logit):
        """
        计算子图的置信度图

        参数:
        ----------
        crop_seg_logit : torch.Tensor
            CLIP计算的相似度/Logit, 形状为 [B, n, H, W], n为类别数
        seg_mask : torch.Tensor, 可选
            SAM预分割的无类别mask, 形状为 [B, 1, H, W], 0为背景, 1,2,3...为不同实例标签

        返回:
        -------
        confidence_map : torch.Tensor
            置信度图, 形状为 [B, 1, H, W]
        """
        B, n, H, W = crop_seg_logit.shape
        if seg_mask.shape[-2:] != (H, W):
            seg_mask = F.interpolate(seg_mask.float(), size=(H, W), mode='nearest').to(seg_mask.dtype)

        top2_vals, _ = torch.topk(crop_seg_logit, k=2, dim=1)  # [B, 2, H, W]
        margin = top2_vals[:, 0:1, :, :] - top2_vals[:, 1:2, :, :]  # [B, 1, H, W]
        if not self.scale_normalize:
            min_v = margin.view(B, -1).min(dim=-1)[0].view(B, 1, 1, 1)
            max_v = margin.view(B, -1).max(dim=-1)[0].view(B, 1, 1, 1)
            margin = (margin - min_v) / (max_v - min_v + 1e-9)

        if self.experience_scale:
            # seg_mask: [B, 1, H, W], 0 为背景，1..N 为 SAM 的预分割区域
            flat_mask = seg_mask[:, 0].reshape(B, -1).long()
            flat_mask = flat_mask.reshape(-1)
            batch_ids = torch.arange(B, device=seg_mask.device).view(B, 1).expand(B, H * W).reshape(-1)

            valid = flat_mask > 0
            if valid.any():
                num_bins = int(flat_mask.max().item()) + 1
                combined = batch_ids[valid] * num_bins + flat_mask[valid]
                counts = torch.bincount(combined, minlength=B * num_bins).reshape(B, num_bins)
                fg_counts = counts[:, 1:]

                if fg_counts.shape[1] == 0:
                    scale = torch.ones((B, 1), device=margin.device, dtype=margin.dtype)
                else:
                    topk = torch.topk(fg_counts, k=min(2, fg_counts.shape[1]), dim=1).values
                    first = topk[:, 0:1]
                    second = topk[:, 1:2] if topk.shape[1] > 1 else torch.zeros_like(first)
                    ratio = (first - second).to(margin.dtype) / float(H * W)
                    scale = 1.0 - ratio
            else:
                scale = torch.ones((B, 1), device=margin.device, dtype=margin.dtype)

            return margin * scale.view(B, 1, 1, 1)
        else:
            return margin

    @torch.no_grad()
    def predict(self, inputs, data_samples):
        if data_samples is not None:
            if self.scales_fusion:
                instance_masks = [self.generate_mask(data_sample.img_path) for data_sample in data_samples]
                instance_masks = torch.stack(instance_masks, dim=0)
                # self.image_name = data_samples[0].img_path.split('\\')[-1]
                # print(self.image_name)
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            inputs, img_path = inputs
            # self.image_name = img_path.split('/')[-1]
            # print(self.image_name)
            if self.scales_fusion:
                instance_masks = (self.generate_mask(img_path)).unsqueeze(0)
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        inputs = inputs.half()
        if self.slide_crop > 0:
            if self.scales_fusion:
                instance_masks = instance_masks.int()
                instance_masks = F.interpolate(instance_masks.unsqueeze(1).float(),
                                               size=batch_img_metas[0]['ori_shape'],
                                               mode='nearest').int()
                resized_masks = F.interpolate(instance_masks.float(), size=inputs.shape[2:],
                                              mode='nearest').int()
            if self.scales_fusion:
                seg_logits, confidence = self.forward_slide(inputs, resized_masks, batch_img_metas, self.slide_stride,
                                                            self.slide_crop)
                half_img = torchvision.transforms.Resize([int(inputs.shape[-2] / 2), int(inputs.shape[-1] / 2)])(
                    inputs[0].unsqueeze(0))
                half_masks = F.interpolate(resized_masks.float(), size=half_img.shape[2:],
                                           mode='nearest').int()
                half_logits, half_confidence = self.forward_slide(half_img, half_masks, batch_img_metas,
                                                                  self.slide_stride, self.slide_crop)
                seg_logits = torch.cat((seg_logits, half_logits), dim=0)
                confidence = torch.cat((confidence, half_confidence), dim=0)
                fused_weights = self.get_fusion_weights(instance_masks)

                # fusion_weights = self.get_fusion_weights(instance_masks)
                # fusion_weights = fusion_weights.unsqueeze(1)
                # fused_weights = fusion_weights * confidence
                # fused_weight_sum = fused_weights.sum(dim=0, keepdim=True)
                # base_weight_sum = fusion_weights.sum(dim=0, keepdim=True)
                # fused_weights = torch.where(
                #     fused_weight_sum > 0,
                #     fused_weights / fused_weight_sum.clamp_min(1e-9),
                #     fusion_weights / base_weight_sum.clamp_min(1e-9),
                # )
                
                if self.paint_feature:
                    plt.axis('off')
                    plt.imsave(os.path.join(self.paint_path, 'region_masks.png'), instance_masks.squeeze().squeeze().cpu().numpy(),
                               cmap='viridis')
                    # plt.imsave(os.path.join(self.paint_path, 'weight_mask.png'), fused_weights[0, 0].cpu().numpy(),
                    #            cmap='viridis')
                    # plt.imsave(os.path.join(self.paint_path, 'weight_mask_half.png'), fused_weights[1, 0].cpu().numpy(),
                    #            cmap='viridis')
                    plt.imsave(os.path.join(self.paint_path, 'weight_mask.png'), fused_weights[0].cpu().numpy(),
                               cmap='viridis')
                    plt.imsave(os.path.join(self.paint_path, 'weight_mask_half.png'), fused_weights[1].cpu().numpy(),
                               cmap='viridis')
                fused_weights = fused_weights.unsqueeze(1)
                confidence = torch.sum(confidence * fused_weights, dim=0, keepdim=True)
                # size_bins = [0.0, 0.25, 0.75, 1.0]
                # conf_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                # self.size_bin_map = self.build_size_bin_region_map(instance_masks.long(), size_bins)
                # self.conf_bin_map = self.build_confidence_bin_region_map(confidence, conf_bins)
                fused_weights = fused_weights.expand_as(seg_logits)
                seg_logits = torch.sum(seg_logits * fused_weights, dim=0, keepdim=True)
                if self.paint_feature:
                    plt.axis('off')
                    plt.imsave(os.path.join(self.paint_path, 'fusion_confidence.png'),
                               confidence.squeeze().cpu().numpy(), cmap='viridis')
                    plt.imsave(os.path.join(self.paint_path, 'fusion_logits.png'),
                               torch.max(seg_logits, dim=1)[0].squeeze().cpu().numpy(), cmap='viridis')
                if self.geometric_correction and self.aggregation_method == 'mean':
                    seg_logits = self.scatter_mean_replace_confidence_aware(seg_logits, instance_masks, confidence)
            else:
                seg_logits = self.ori_forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        if self.geometric_correction:
            return self.postprocess_result(seg_logits, data_samples, instance_masks, confidence)
        else:
            return self.ori_postprocess_result(seg_logits, data_samples)

    def ori_postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                # data_samples[i].set_data({
                #     'seg_logits':
                #         PixelData(**{'data': seg_logits}),
                #     'pred_sem_seg':
                #         PixelData(**{'data': seg_pred}),
                #     'size_bin_map':
                #         PixelData(**{'data': self.size_bin_map[i]}),
                #     'conf_bin_map':
                #         PixelData(**{'data': self.conf_bin_map[i]})
                # })
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def postprocess_result(self, seg_logits, data_samples, instance_masks, confidence):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx
            if self.geometric_correction and self.aggregation_method == 'vote':
                # border_mask = self.get_border_mask(instance_masks)
                # confidence = confidence * border_mask
                # if self.paint_feature:
                #     plt.axis('off')
                #     plt.imsave(os.path.join(self.paint_path, 'border_masked_confidence.png'),
                #                confidence.squeeze().cpu().numpy(), cmap='viridis')
                seg_pred = self.scatter_vote_replace_confidence_aware(seg_pred, instance_masks, confidence)
            # output_path = os.path.join('./output', self.image_name)
            # print(output_path)
            # output = seg_pred.squeeze().to('cpu').numpy()
            # cv2.imwrite(output_path, np.uint8(output))
            if data_samples is None:
                return seg_pred
            else:
                # data_samples[i].set_data({
                #     'seg_logits':
                #         PixelData(**{'data': seg_logits}),
                #     'pred_sem_seg':
                #         PixelData(**{'data': seg_pred}),
                #     'size_bin_map':
                #         PixelData(**{'data': self.size_bin_map[i]}),
                #     'conf_bin_map':
                #         PixelData(**{'data': self.conf_bin_map[i]})
                # })
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices