# Rethinking 'Rethinking pre-training and self-training' a little bit

- [Goal](#goal)
- [**Data Collection & Data Analysis**](#--data-collection-------data-analysis--)
- [**Data Modeling**](#--data-modeling--)
- [**Predictive Outcomes**](#--predictive-outcomes--)
  * [YOLOv5m](#yolov5m)
  * [YOLOv5l](#yolov5l)
  * [YOLOv5x](#yolov5x)
- [**Answering those three questions we list at the begining**](#--answering-those-three-questions-we-list-at-the-begining--)
- [**Learning outcomes out of the project**](#--learning-outcomes-out-of-the-project--)
- [Tutorial to reproduce this project](#tutorial-to-reproduce-this-project)
  * [Set up environment](#set-up-environment)
  * [Download dataset](#download-dataset)
  * [Choose different levels of data augmentations](#choose-different-levels-of-data-augmentations)
  * [Training from scratch](#training-from-scratch)
  * [Pre-training](#pre-training)
  * [Self-training](#self-training)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>




## Goal

This project is motivated by the paper *Rethinking pre-training and self-training* (https://arxiv.org/abs/2006.06882). In the original paper, their goal is to find out the relationship between data augmentation and different training methods, including self-training and pre-training. Their train their models on MSCOCO, but use ImageNet to pre-train or generat psudel labeled data. However, due to the limitation of computation power, this project is experimenting on VOC (http://host.robots.ox.ac.uk/pascal/VOC/) dataset, using COCO (https://arxiv.org/abs/1405.0312) pretrain models.

We want to verify three issues:

1. How data augmentations perform on different kinds of training?
2. How different models with different sizes perform with different kinds of training?
3. What's the different if finetune on the same task, i.e. object detection to object detection?

![image-20201217151551338](https://i.loli.net/2020/12/17/57tVORYEnLDN8im.png)

The other differences are as follows:

| Projects          | Paper's                                                      | ours                                                         |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Data augmentation | Augment-S1 : Flips & Crops<br />Augment-S2: S1 + AutoAugment<br />Augment-S3: S2 + Large Scale Jittering<br />Augment-S4: S4 + RandAugment | Augment-S1 : Flips, Perspective & HSV<br />Augment-S2: S1 + Mosaic<br />Augment-S3: S2 + Mixup<br />Augment-S4: S4 + Cutout |
| Tasks             | Image Classification -> Object Detection<br />Image Classification -> Semantic Segmentation | Object Detection -> Object Detection                       |
| Datasets          | ImageNet + MSCOCO                                            | MSCOCO + VOC                                                 |
| Models            | EfficientNet-B7(backbone) + FPN + RetinaNet detector         | YOLOv5                                                       |
|                   |                                                              |                                                              |

## **Data Collection **& **Data Analysis**

In this project, we mainly use VOC (http://host.robots.ox.ac.uk/pascal/VOC/) to train and validate our models, while using MSCOCO (http://host.robots.ox.ac.uk/pascal/VOC/) to generate psudel labels.

|        | VOC         | MSCOCO                     |
| ------ | ----------- | -------------------------- |
| Size   | 17k+ images | 328k images, 2,500k labels |
| #Class | 20          | 91                         |

<center> COCO sample: </center>

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE1LmNuYmxvZ3MuY29tL2Jsb2cvMzY5Mjc3LzIwMTcwNC8zNjkyNzctMjAxNzA0MDEyMDE1MjIwMDgtMjAxNDE5NDI1Mi5wbmc)

<center> VOC samples: </center>

<img src="https://i.loli.net/2020/12/17/14oixzDBZlGqSFb.png" alt="image-20201217153232309" style="zoom:50%;" />


In this project, scripts are provided to download both of them, by entering:

```
!sh data/scripts/get_coco.sh
!sh data/scripts/get_voc.sh
```

## **Data Modeling**

In this project, we use YOLOv5 (thanks to https://github.com/ultralytics/yolov5) to model the data. The main structure is shown:

![img](https://lh4.googleusercontent.com/saE6POnJBxKeC5QNx73kNVrzC9Qs-OlpxfrzIC2gdG_KkgZ24u34nK4BLVDXyUg3iVU7APBX-51QGvpFTkTPMz-g3K0JKs80uN1aCg_3TllLkGWj2YrCD7CE43wvcYVwtXGQQi8LAxw)

And really recommend to read YOLO algorithms family:
1. YOLOv1: https://arxiv.org/abs/1506.02640 
2. YOLOv2: https://arxiv.org/abs/1612.08242 
3. YOLOv3: https://arxiv.org/abs/1804.02767 
4. YOLOv4: https://arxiv.org/abs/2004.10934 
5. YOLOv5: the paper is still working, according to the authors

And we use three different size of YOLOv5, including YOLOv5m, YOLOv5l and YOLOv5x:

![image-20201217154521010](https://i.loli.net/2020/12/17/tOHo4S9clafpEgK.png)

Besides, different data augmentations will be shown:

<center> Augment-S1: Flips, HSV and Perspective:  </center>

![image-20201217154146556](https://i.loli.net/2020/12/17/jf5rpMPoeEYzcvG.png)


<center> Augment-S2- S1 + Mosaic: </center>

![img](https://lh6.googleusercontent.com/M6tymDEEtbJlWWM298axqrQ-1vd3fQXKnraKjqCPIsQlnhT02Sncd3Gx2H9nvOi5enIvVH0RdFM_QfKSE4A8-yU9Z5f7M5-nl5AqXWNx6BUctOzURz42NAoDW4mXs55cbomMblrG59g)

<center> Augment-S3: S2 + MixUp: </center>

![img](https://lh6.googleusercontent.com/MjieJbMpoC6MEJa_HCrCubDQE0mdpDZu9yJZU-p7Ix1Vr1HHdu8XdcaaRebM6nCqtmDtK1c72Vid0p7Qd81rTqMSWXiqW_eRXBS5CFLPU2rXa-y1mP7unhdeQB5nnnhacqRZ9wORVzo)

<center> Augment-S4 : S3 + Cutout: </center>

![img](https://lh5.googleusercontent.com/hXLZ2iNlVt7Y8PvJTo482CuNEZJtffrlNJeHNdCd4XYmvPVKwkT8EkYZXSn3MT5HFcpq9t0B4H08DhxXCvte45k0kQKKcxB-Mmn2-71IXLjcUZQwDgodDhiXmgLEqdOYHCRAMf-49YI)

## **Predictive Outcomes**

We report mAP@0.5s of our works according to the model size.

For mAP@0.5, we recommend this paper: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

### YOLOv5m

<center> YOLOv5m: </center>

![img](https://lh4.googleusercontent.com/bWsF5PG-jHrdYbz3p4MmobQMwGVXysy3kHrt1RCh3FR-cEZBK7Yx-Sm4dlsPmgTverum1X5KiGAV8Pu5dz-UIIYYENvf9h7WnsoB8AtKeAB5QW4cM8mpR7JH2cqjIim43jxDW6IV8Gw)


Findings:

1. Using Random Initialization, stronger data augmentation works better.
2. But it stronger augmentation hurt performances of yolov5m, where S2 is strong enough for self-training and S3 is for pre-training.

### YOLOv5l


<center> YOLOv5l: </center>

![image-20201217204111772](https://i.loli.net/2020/12/17/LYyB3Rk79bqMJcw.png)

Findings:

1. Using Random Initialization, stronger data augmentation works better for yolov5l too;
2. But it stronger augmentation hurt performances of yolov5l, where S1 is strong enough for self-training and S3 is for pre-training.

### YOLOv5x

<center> YOLOv5x: </center>

![img](https://lh4.googleusercontent.com/4viJaoIe6KF8an2vhD4OiKhp7HzRHinKeuUjkSszcCUzi0puaDkrb_m5hdHjOC7CqXc_JGJY8oyr9VNi_E1HNNM_hEsqRu1ENsHPv8Gmr22H-XCsFaSd7U_UexbOCzcSwpeQmIJlfiY)

Findings:

1. Stronger data augmentation always hurts performance



## **Answering those three questions we list at the begining**

1. Even the weakest data augmentation can significantly improve the performance when training from scratch; in the environment of this project, self-training do not perform better than pre-training, maybe because of the incompatibility between augmentations;
2. Larger the model is, data augmentations are more likely to hurt the performance;
3. Fine-tuning from the same task is better than self-training, due to its faster training and high accuracy, while the conclusion of paper is totally different, i.e., finetuning from different tasks performs worse than self-training.

## **Learning outcomes out of the project**

1. Problem formulation and literature review: to select an appropriate topic that is a perfect match for the course scope and our interests can be challenging. In the meanwhile, we hope our project can be related to the cut-edge work of AI society. Thus we investigated several search engines and databases, including IEEE, ACM, Springer, arxiv, etc. Finally, we chose a high standard, up-to-date working on self-training published on NIPS 2020.
2. Training of deep neural networks (DNNs).
DNNs are the most powerful tools in machine learning by far and capable of achieving human performance on tasks in computer vision, natural language processing, etc. However, to train a DNN that typically involve millions of parameters can be tough and tricky. We have well-practiced this skill in this project.
3. Deploying models on an online platform.
Due to the limitation of computation resources, we have to use Google Cloud for GPU access. However, deploy a model on Google Cloud is challenging since we have to deal with datasets and models remotely. We also do not have the privilege of an administrator as we can enjoy in local machines. This project offers us a chance to get used to online platforms like Google Cloud.


## Tutorial to reproduce this project

### Set up environment

```python
!git clone https://github.com/ChristopherSTAN/UD-CISC849
%cd yolov5/
!git checkout 04081f8
!pip install -r requirements.txt
```

### Download dataset

```python
!sh data/scripts/get_voc.sh
!sh data/scripts/get_coco.sh
```

### Choose different levels of data augmentations

|                 | Files                            |
| --------------- | -------------------------------- |
| Augment-S1      | `data/hyp.geoaug.yaml`           |
| Augment-S2      | `data/hyp.geomosaic.yaml`        |
| Augment-S3      | `data/hyp.geo_mosaic_mixup.yaml` |
| No augmentation | `data/hyp.scratch.yaml`          |

Using Augment-S4 is different, add this line before running `train.py`:

```python
!cp data/datasets.py utils
```

### Training from scratch

Train yolov5l on VOC from scratch:

```python
%cd /content/yolov5
!python train.py --data data/voc.yaml --weights "" \
--hyp data/hyp.geomosaic.yaml \
--cfg models/yolov5l.yaml \
--batch-size 16 \
```

If you want to train different model like yolov5x, edit `--cfg` to `yolov5x.yaml`.

### Pre-training

Train pretrained yolov5l on VOC

```python
%cd /content/yolov5
!python train.py --data data/voc.yaml --weights yolov5l.pt \
--hyp data/hyp.geomosaic.yaml \
--cfg models/yolov5l.yaml \
--batch-size 16 \
```

### Self-training

We use the best-performing model to generate labels on data from COCO and you can access 

[here]: https://drive.google.com/file/d/1HhhmIBJXC1ZFcgR1OWHvoSMg-wtIpjBt/view?usp=sharing	"."

.

Download it and place at `/content` on Colab. Then run the following lines to generate new dataset for self-training.

```python
%cd /content/
!cp labels.tar .
!tar xvf 'labels.tar'

from tqdm import tqdm
import os

labels = os.listdir('inference/labels')
len(labels)

img_src = '/content/coco/images/train2017'
img_dst = '/content/VOC/images/train'
lbl_src = '/content/inference/labels'
lbl_dst = '/content/VOC/labels/train'


for label in tqdm(labels):
    img = label.replace('txt', 'jpg')
    img = os.path.join(img_src, img)
    #print(os.path.exists(img))
    lbl = os.path.join(lbl_src, label)
    #print(os.path.exists(lbl))
    os.system('cp ' + img + ' ' + img_dst)
    os.system('cp ' + lbl + ' ' + lbl_dst)

```

Then, the lines for training is the same:

```python
%cd /content/yolov5
!python train.py --data data/voc.yaml --weights "" \
--hyp 'data/hyp.geo_mosaic_mixup.yaml' \
--cfg models/yolov5x.yaml \
--batch-size 8 \
```


