# Scale-Adaptive and Confidence-Aware Open-Vocabulary Segmentation for High Resolution Remote Sensing Images

HRRS-Seg is a training-free open-vocabulary semantic segmentation framework for high-resolution remote sensing images. It combines Region-Scale Adaptive Feature Fusion (RSAFF) and Confidence-Weighted Regional Voting (CWRV).

## Environment

- Python 3.12
- CUDA 12.8 or newer is recommended for a Blackwell-based NVIDIA RTX 5060 Laptop GPU
- NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, Windows 11 (paper environment)
- 8 GB VRAM is a practical minimum; use CPU or fp16 only for diagnostic workloads

The repository is expected to reproduce the results on Linux or Windows. The training-free evaluation scripts require a CUDA-enabled PyTorch installation.

## Installation

```bash
git clone https://github.com/topologicalstructure/HRRS-Seg.git
cd HRRS-Seg
conda create -n hrrs-seg python=3.12
conda activate hrrs-seg
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA build of PyTorch matching the installed NVIDIA driver before installing the remaining requirements:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The bundled `SimFeatUp` implementation is used directly from this repository. If it is unavailable, please refer to [SimFeatUp](https://github.com/likyoo/SimFeatUp) to install it.

If the `mmcv` cannot be installed, please refer to [https://mmcv.readthedocs.io/en/latest/get_started/build.html](https://mmcv.readthedocs.io/en/latest/get_started/build.html) to build it from source.

## Pretrained Weights

Download the SAM2 Large checkpoint from [https://huggingface.co/facebook/sam2-hiera-large/blob/main/sam2_hiera_large.pt](https://huggingface.co/facebook/sam2-hiera-large/blob/main/sam2_hiera_large.pt).

Place `sam2_hiera_large.pt` in the repository root:

```text
HRRS-Seg/sam2_hiera_large.pt
```

The CLIP/SimFeatUp checkpoint is already included at `simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt`. The default CLIP backbone is downloaded or resolved by the code; set `HF_ENDPOINT` when network mirrors are unavailable.

## Data Preparation

The 16 supported datasets are not distributed in this repository. Follow [dataset_prepare.md](dataset_prepare.md) to obtain them from their original providers and convert them to the expected directory layout. Each dataset remains subject to its original license or application process. The GT for licensed datasets must be obtained directly from the original provider.

Typical converted layout:

```text
data/
  OpenEarthMap/
  LoveDA/
  iSAID/
  potsdam/
  vaihingen/
  ...
```

## Quick Start

Run the bundled OpenEarthMap sample:

```bash
python demo.py
```

This creates `seg_pred.png`. See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for custom images and vocabulary files.

Run a custom image with a custom class list:

```bash
python examples/run_custom.py --image demo/oem_koeln_50.tif --classes examples/custom_classes.txt --output seg_pred_custom.png
```

## Reproduce Paper Results

Evaluate all 16 datasets:

```bash
test.bat
```

Generate the ablation study used for Table 4:

```bash
ablation.bat
```

On Linux or PowerShell, run the equivalent `python eval.py --config ... --work-dir ./work` commands directly. All commands expect the corresponding converted data directory.

Additional reproduction and parameter information is in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Outputs and Self-Check

Evaluation writes metrics to `work/[EXP_ID]/[EXP_ID].json` and `results.xlsx`. Inference writes:

- a visualization in `seg_pred.png`
- different categories are represented by different colors

The bundled demo should produce a three-panel image containing the input, prediction, and ground truth.

## Citation

```bibtex
@unpublished{liang2026hrrsseg,
  title  = {Scale-Adaptive and Confidence-Aware Open-Vocabulary Segmentation for High Resolution Remote Sensing Images},
  author = {Shuang Liang and Jingyue Wang and Haotian Sun and Guihu Bai and Hongming Zhu and Shengjie Zhao},
  year   = {2026},
  note   = {Manuscript submitted to Computers and Geosciences}
}
```

## License

The source code is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgement

This implementation is based on [SegEarth-OV](https://github.com/likyoo/SegEarth-OV) and [SimFeatUp](https://github.com/likyoo/SimFeatUp).
