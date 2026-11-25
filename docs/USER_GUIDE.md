# User Guide

## Inputs

The inference entry point accepts a raster image supported by Pillow, such as TIFF, PNG, or JPEG. Images are resized to 896 x 896 by the demo; input dimensions do not need to match exactly.

The vocabulary file is a UTF-8 text file with one category per line. Blank lines are ignored. Use short, concrete category names. CLIP matching is generally better when synonyms are comma-separated, for example:

```text
background
bareland,barren
grass
road
tree,forest
water,river
building,roof,house
```

The first line is class ID 0. Every other line receives the next ID in file order.

## Outputs

- `seg_pred.png`: a three-panel visualization of input, prediction, and ground truth when the demo label is available.
- `seg_pred_custom.png`: the custom-image output from `examples/run_custom.py`.
- Evaluation output: `work/[EXP_ID]/[EXP_ID].json` and `results.xlsx`.

The default colormap is `viridis`; class IDs are not altered by visualization.

## Configuration

The default configuration in `configs/base_config.py` uses:

| Option | Default | Purpose |
|---|---|---|
| `scales_fusion` | `True` | Enable RSAFF at 896 / 448 resolution |
| `geometric_correction` | `True` | Enable CWRV region-aware correction |
| `experience_scale` | `True` | Enable confidence correction |
| `fusion_parameters` | `[0.125, 0.875, 1.0, 0.25, 0.875]` | Scale weights and background weight |

For LoveDA and WBS-SI, use:

```text
fusion_parameters=[0.0, 0.75, 1.0, 0.25, 0.5]
```

## Typical Use Cases

### A. Segment a Custom Image

```bash
python examples/run_custom.py --image demo/oem_koeln_50.tif --classes examples/custom_classes.txt --output seg_pred_custom.png
```

### B. Use a Predefined Alternative Configuration

```bash
python eval.py --config ./configs/cfg_loveda.py --work-dir ./work
```

LoveDA and WBS-SI use the alternate `[0.0, 0.75, 1.0, 0.25]` fusion weights and 0.5 background weight.

### C. Run the Table 4 Ablation

```bash
ablation.bat
```

## Expected Behavior

The bundled demo should produce a 18 x 6 inch, three-panel visualization. The first panel is the source image, the second is the predicted map, and the third is the available ground-truth map.

On a successful run, `demo.py` exits normally and writes `seg_pred.png` in the repository root.

## Troubleshooting

- **CUDA out of memory:** close other GPU processes, reduce image resolution, or use one of the `wo-896` ablation configs.
- **SAM2 checkpoint not found:** ensure `sam2_hiera_large.pt` is in the repository root.
- **SimFeatUp checkpoint not found:** ensure `simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt` exists.
- **Vocabulary is empty or has blank lines:** provide at least one valid class line and ensure UTF-8 encoding.
- **MMCV or MMSegmentation import error:** reinstall the pinned versions in `requirements.txt`.
- **No CUDA device:** inference requires a CUDA-enabled PyTorch build for the supported paper workflow.
