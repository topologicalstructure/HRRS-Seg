# Prepare datasets

## OpenEarthMap

The data could be downloaded from [here](https://open-earth-map.org).

For OpenEarthMap dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/openearthmap.py /path/to/OpenEarthMap
```

We only use ``OpenEarthMap_wo_xBD``.

## LoveDA

The data could be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

Or it can be downloaded from [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF), you should run the following command:

```shell
# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

For LoveDA dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/loveda.py /path/to/loveDA
```

Using trained model to predict test set of LoveDA and submit it to server can be found [here](https://codalab.lisn.upsaclay.fr/competitions/421).

More details about LoveDA can be found [here](https://github.com/Junjue-Wang/LoveDA).

## ISPRS Potsdam

The [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx).
Or download on [BaiduNetdisk](https://pan.baidu.com/s/1K-cLVZnd1X7d8c26FQ-nGg?pwd=mseg)，password：mseg, [Google Drive](https://drive.google.com/drive/folders/1w3EJuyUGet6_qmLwGAWZ9vw5ogeG0zLz?usp=sharing) and [OpenDataLab](https://opendatalab.com/ISPRS_Potsdam/download).
The '2_Ortho_RGB.zip' and '5_Labels_all_noBoundary.zip' are required.

For Potsdam dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/potsdam.py /path/to/potsdam
```

In our default setting, it will generate 504 images for validation.

If you want to post-process the validation labels by replacing clutter pixels with the nearest non-clutter class, run:

```shell
python tools/dataset_converters/process_potsdam.py
```

The processed labels will be written to `data/potsdam/ann_dir_processed/val`.

## ISPRS Vaihingen

The [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/) dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
Or [BaiduNetdisk](https://pan.baidu.com/s/109D3WLrLafsuYtLeerLiiA?pwd=mseg)，password：mseg, [Google Drive](https://drive.google.com/drive/folders/1w3NhvLVA2myVZqOn2pbiDXngNC7NTP_t?usp=sharing).
The 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' are required.

For Vaihingen dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/vaihingen.py /path/to/vaihingen
```

In our default setting (`clip_size`=1024, `stride_size`=512), it will generate 113 images for validation.

If you want to post-process the validation labels by replacing clutter pixels with the nearest non-clutter class, run:

```shell
python tools/dataset_converters/process_vaihingen.py
```

The processed labels will be written to `data/vaihingen/ann_dir_processed/val`.

## iSAID

The data images could be download from [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) (train/val/test)

The data annotations could be download from [iSAID](https://captain-whu.github.io/iSAID/dataset.html) (train/val)

The dataset is a Large-scale Dataset for Instance Segmentation (also have semantic segmentation) in Aerial Images.

You may need to follow the following structure for dataset preparation after downloading iSAID dataset.

```none
├── data
│   ├── iSAID
│   │   ├── train
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
│   │   │   │   ├── part3.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── val
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── test
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
```

```shell
python tools/dataset_converters/isaid.py /path/to/iSAID
```

In our default setting (`patch_width`=896, `patch_height`=896, `overlap_area`=384), it will generate 33978 images for training and 11644 images for validation.

## UAVid

The data could be downloaded from [here](https://www.uavid.nl/#download).

For UAVid dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/uavid.py /path/to/UAVid
```

In our default setting (`patch_width`=1280, `patch_height`=1080, `overlap_area`=0).

## UDD5

The data could be downloaded from [here](https://github.com/MarcWong/UDD).

## VDD

The data could be downloaded from [here](https://github.com/RussRobin/VDD).


# Building Extraction

## WHU Building Dataset (Aerial)

The data could be downloaded from [here](http://gpcv.whu.edu.cn/data/building_dataset.html) or [Kaggle](https://www.kaggle.com/datasets/xiaoqian970429/whu-building-dataset).

You only need to convert the original 0-255 labels to 0-1 labels:

```
label[label < 128] = 0
label[label >= 128] = 1
```
## xBD

The data could be downloaded from [here](https://github.com/michal2409/xView2).

You need to convert the original 0-255 labels to 0-1 labels:

```
python tools/dataset_converters/xBD.py /path/to/xBD
```

Then, move the images and labels with "pre" in the filename to a new directory.
## Inria

The data could be downloaded from [here](https://project.inria.fr/aerialimagelabeling/), the data split refers to [FCtL](https://github.com/liqiokkk/FCtL).

For Inria dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/inria.py /path/to/Inria
```

In our default setting (`patch_width`=1000, `patch_height`=1000, `overlap_area`=0).

# Road Extraction

All road datasets need to be converted from the original 0-255 labels to 0-1 labels.

## DeepGlobe Road Detection

The data could be downloaded from [here](https://github.com/xiaoyan07/GRNet_GRSet?tab=readme-ov-file).

More details about DeepGlobe can be found [here](https://competitions.codalab.org/competitions/18467#participate-get_data).

## Massachusetts Roads Dataset

The data could be downloaded from [here](https://github.com/xiaoyan07/GRNet_GRSet?tab=readme-ov-file).

More details about Massachusetts can be found [here](https://www.cs.toronto.edu/~vmnih/data/).

## SpaceNet

The data could be downloaded from [here](https://github.com/xiaoyan07/GRNet_GRSet?tab=readme-ov-file).

More details about Massachusetts can be found [here](https://spacenet.ai/spacenet-roads-dataset/).

# Water Extraction

## WBS-SI

The data could be downloaded from [here](https://www.kaggle.com/datasets/shirshmall/water-body-segmentation-in-satellite-images).

Use the filenames listed in ``tools/dataset_converters/wbs-si_val.txt`` as the validation set.

# Landslide Extraction

## GVLM

The data could be downloaded from [here]([https://www.kaggle.com/datasets/shirshmall/water-body-segmentation-in-satellite-images](https://github.com/zxk688/GVLM)).

Use `tools/GVLM/1GVLM_Img_clip.py` to clip the images and labels into 1024x1024 patches. You will get 536 patches for testing.

