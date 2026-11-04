# Scale-Adaptive and Confidence-Aware Open-Vocabulary Segmentation for High Resolution Remote Sensing Images

## Dependencies and Installation
```
# 1. install SimFeatUp
# refer to https://github.com/likyoo/SimFeatUp

# 2. git clone this repository
git clone https://github.com/topologicalstructure/HRRS-Seg.git
cd HRRS-Seg

# 3. create new anaconda env
conda create -n HRRS-Seg python=3.12
conda activate HRRS-Seg

# 4. install torch and dependencies
pip install -r requirements.txt
# The dependent versions are not strict, and in general you only need to pay attention to mmcv and mmsegmentation.
```

## Download Pretrained SAM Model

SAM model can be downloaded from [here](https://huggingface.co/merve/sam2-hiera-large/blob/main/sam2_hiera_large.pt). Please put it in the main directory of this repository(HRRS-Seg/sam2_hiera_large.pt).

## Datasets

Please refer to [dataset_prepare.md](dataset_prepare.md) for dataset preparation.

## Quick Inference

```
python demo.py
```

## Model evaluation

```
python eval.py --config ./configs/cfg_{DATASET_NAME}.py --work-dir ./work
python eval.py --config ./configs/cfg_openearthmap.py --work-dir ./work
python eval.py --config ./configs/cfg_loveda.py --work-dir ./work
```

## Acknowledgement
This implementation is based on [SegEarth-OV](https://github.com/likyoo/SegEarth-OV). Thanks for the awesome work.

