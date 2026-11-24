# Reproduction Guide

## Paper Environment

The reported environment used an NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB VRAM. The implementation is training-free and uses no stochastic training phase.

The primary configuration uses a 896 x 896 scale and a 448 x 448 scale. For standard default evaluation:

```text
fusion_parameters=[0.125, 0.875, 1.0, 0.25, 0.875]
scales_fusion=True
geometric_correction=True
experience_scale=True
```

LoveDA and WBS-SI use:

```text
fusion_parameters=[0.0, 0.75, 1.0, 0.25, 0.5]
scales_fusion=True
geometric_correction=True
experience_scale=True
```

## Table and Dataset Reproduction

Prepare each dataset as described in `dataset_prepare.md`. Then run:

```bash
test.bat
```

To evaluate one dataset manually:

```bash
python eval.py --config ./configs/cfg_openearthmap.py --work-dir ./work
python eval.py --config ./configs/cfg_loveda.py --work-dir ./work
python eval.py --config ./configs/cfg_wbs-si.py --work-dir ./work
```

Replace the config name with the required dataset. Metrics use MMSegmentation `mIoU` and are written to `work/results.txt` and `work/results.xlsx`.

## Table 4 Ablations

Run:

```bash
ablation.bat
```

The configs compare:

- `wo-cc`: confidence correction disabled
- `wo-cwrv`: CWRV disabled
- `wo-rsaff`: RSAFF disabled
- `wo-896`: the 896-pixel scale disabled

Compare the resulting `mIoU` values in `work/`.

## Determinism

The evaluated method is designed to be deterministic. Dataset order is not shuffled. CUDA benchmark mode is enabled only through the base config and no random seed should be changed for the reported inference protocol.
