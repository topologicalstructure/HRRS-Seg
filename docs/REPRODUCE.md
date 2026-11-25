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

Replace the config name with the required dataset. For each experiment, an `[EXP_ID]` is assigned based on its start time to distinguish between experimental batches. For example, `20261024_173051`.

Metrics use MMSegmentation `mIoU` and are written to `work/[EXP_ID]/[EXP_ID].json` and `results.xlsx`. The file `work/[EXP_ID]/[EXP_ID].log` reports the `IoU` results for all individual categories; for datasets involving 8 distinct object categories, you should consult this file to find the `IoU` results for the specific target category.

It should be noted that the `potsdam` and `vaihingen` datasets exclude the "clutter" category; however, the script includes this category when calculating the mean IoU (`mIoU`)—using a denominator of 6 instead of the correct value of 5. Therefore, to obtain the accurate `mIoU` result, please multiply the output value by a correction factor of 1.2.

## Table 4 Ablations

Run:

```bash
ablation.bat
```

The configs compare:

- `wo-cc`: confidence correction disabled
- `wo-cwrv`: disabled CWRV baed on 'wo-cc'
- `wo-rsaff`: disabled RSAFF based on 'wo-cwrv', only use 896 scale for inference
- `wo-896`: only use 448 scale for inference

Compare the resulting `mIoU` or 'IoU' values.

## Determinism

The evaluated method is designed to be deterministic. Dataset order is not shuffled. CUDA benchmark mode is enabled only through the base config and no random seed should be changed for the reported inference protocol.
