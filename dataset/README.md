# Dataset Preparation
We introduce the preparation for ImageNet-C/R/V2/Sketch here.

## ImageNet-C
- Step 1: Download [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [here 🔗](https://zenodo.org/record/2235448#.YpCSLxNBxAc). 

- Step 2: Extract the files from the tar archive and organize them in the following format:

```
imagenet-c
|-- brightness
|   |-- 1
|   |-- 2
|   |-- 3
|   |-- 4
|   `-- 5
|-- contrast
|-- defocus_blur
|-- elastic_transform
|-- fog
|-- frost
|-- gaussian_noise
|-- glass_blur
|-- impulse_noise
|-- jpeg_compression
|-- motion_blur
|-- pixelate
|-- shot_noise
|-- snow
`-- zoom_blur
```

## ImageNet-V2
- Step 1: Download [ImageNet-V2 🔗](https://github.com/modestyachts/ImageNetV2) dataset from [here 🔗](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main). 
- Step 2: Extract the files from _imagenetv2-matched-frequency.tar.gz_ and organize files in the following format:
```
imagenet-v2
|-- imagenetv2-matched-frequency-format-val
|   |-- 1
|   |-- 2
|   |-- 3
|   |-- 4
|   |-- 5
|   ...
```

## ImageNet-R
- Step 1: Download [ImageNet-R 🔗](https://github.com/hendrycks/imagenet-r) dataset from [here 🔗](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar). 

- Step 2: Extract the files from the tar archive.


## ImageNet-Sketch
- Step 1: Download [ImageNet-Sketch 🔗](https://github.com/HaohanWang/ImageNet-Sketch) dataset from [here 🔗](https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA/view). 
- Step 2: Extract the files from the zip archive.