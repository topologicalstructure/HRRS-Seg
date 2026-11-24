import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
from torchvision import transforms

from segearth_segmentor import SegEarthSegmentation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HRRS-Seg on one custom image."
    )
    parser.add_argument("--image", required=True, help="Input raster image.")
    parser.add_argument("--classes", required=True, help="UTF-8 class-name file.")
    parser.add_argument("--output", default="seg_pred_custom.png")
    parser.add_argument("--size", type=int, default=896)
    parser.add_argument("--save-npy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = REPO_ROOT

    classes = [
        line.strip()
        for line in Path(args.classes).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError("The class vocabulary is empty.")

    class_path = repo_root / "configs" / "custom_demo.txt"
    class_path.write_text("\n".join(classes), encoding="utf-8")

    image = Image.open(args.image).convert("RGB")
    image_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
            transforms.Resize((args.size, args.size)),
        ]
    )(image).unsqueeze(0).cuda()

    model = SegEarthSegmentation(
        clip_type="CLIP",
        vit_type="ViT-B/16",
        model_type="SegEarth",
        ignore_residual=True,
        feature_up=True,
        feature_up_cfg=dict(
            model_name="jbu_one",
            model_path=str(repo_root / "simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt"),
        ),
        cls_token_lambda=-0.3,
        name_path=str(class_path),
        prob_thd=0.1,
        scales_fusion=True,
        sam_parameters=[32, 0.2, 0.5],
        fusion_parameters=[0.125, 0.875, 1.0, 0.25, 0.875],
        geometric_correction=True,
        scale_normalize=False,
        aggregation_method="vote",
        sub_confi_fusion=False,
        experience_scale=True,
        paint_feature=True,
    )

    prediction = model.predict([image_tensor, str(args.image)], data_samples=None)
    mask = prediction.data.cpu().numpy().squeeze(0)

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Input")
    ax[0].axis("off")
    ax[1].imshow(mask, cmap="viridis", vmin=0, vmax=len(classes) - 1)
    ax[1].set_title("Prediction")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")

    if args.save_npy:
        np.save(Path(args.output).with_suffix(".npy"), mask)

    print(f"Saved visualization to {args.output}")
    if args.save_npy:
        print(f"Saved class IDs to {Path(args.output).with_suffix('.npy')}")


if __name__ == "__main__":
    main()
