<br/>

<div align="center">
    <img src="resources/logo.jpg" width="600"/>
</div>

<br/>


# Human-Art

This repository contains the implementation of the following paper:
> **Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes** [[Project Page]](https://idea-research.github.io/HumanArt/) [[Paper]](https://arxiv.org/abs/2303.02760) [[Code]](https://github.com/IDEA-Research/HumanArt) [[Data]](https://forms.gle/UVv1GiNJNQsE4qif7) [[Video]](https://www.youtube.com/watch?v=djmTKVlw53E) <br>
> [Xuan Ju](https://juxuan.space/)<sup>∗12</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>∗1</sup>, [Jianan Wang](https://github.com/wendyjnwang/)<sup>1</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
> <sup>∗</sup> Equal contribution <sup>1</sup>International Digital Economy Academy <sup>2</sup>The Chinese University of Hong Kong



**Table of Contents**

- [Human-Art](#human-art)
  - [General Description](#general-description)
  - [Dataset Download](#dataset-download)
  - [Human Pose Estimation](#human-pose-estimation)
  - [Human Detection](#human-detection)
  - [Citing Human-Art](#citing-human-art)



## General Description


<div  align="center">    
<img src="resources/dataset_overview.png" width="90%">
</div>


This paper proposes a large-scale dataset, Human-Art, that targets multi-scenario human-centric tasks to bridge the gap between natural and artificial scenes. It includes twenty high-quality human scenes, including natural and artificial humans in both 2D representation (yellow dashed boxes) and 3D representation (blue solid boxes).

**Contents of Human-Art:**

- `50,000` images including human figures in `20 scenarios` (5 natural scenarios, 3 2D artificial scenarios, and 12 2D artificial scenarios)
- Human-centric annotations include `human bounding box`, `21 2D human keypoints`, `human self-contact keypoints`, and `description text`
- baseline human detector and human pose estimator trained on the joint of [MSCOCO](https://cocodataset.org/) and Human-Art

**Tasks that Human-Art targets for:**

- multi-scenario `human detection`, `2D human pose estimation`, and `3D human mesh recovery`
    - Notably, after training with ED-Pose, results on MSCOCO raise 0.8, indicating multi-scenario images may benefit feature extraction and human understanding of real scenes.
- multi-scenario `human image generation` (especially `controllable` human image generation, e.g. with conditions such as pose and text)
- `out-of-domain` human detection and human pose estimation



## Dataset Download


Under the CC-license, Human-Art is available for download. Fill out [this form](https://forms.gle/UVv1GiNJNQsE4qif7) to request authorization to use Human-Art for non-commercial purposes. After you submit the form, an email containing the dataset will be instantly delivered to you. Please do not share or transfer the data privately.

For convenience of usage, Human-Art is processed using the same format as [MSCOCO](https://cocodataset.org/). Please save the dataset with the following file structure after downloading (we also include the file structure of COCO because we use it for joint training of COCO and Human-Art):

```
|-- data
    |-- HumanArt
        |-- annotations 
            |-- training_coco.json
            |-- training_humanart.json
            |-- training_humanart_coco.json
            |-- training_humanart_cartoon.json
            |-- ...
            |-- validation_coco.json
            |-- validation_humanart.json
            |-- validation_humanart_coco.json
            |-- validation_humanart_cartoon.json
            |-- ...
        |-- images
            |-- 2D_virtual_human
                |-- ...
            |-- 3D_virtual_human
                |-- ...
            |-- real_human
                |-- ...
    |-- coco
        |-- annotations 
        |-- train2017 
        |-- val2017 
```

Noted that we have several different json settings:

- the ones end with \_coco (e.g. training_coco.json) is reprocessed coco annotation json files (e.g. person\_keypoints\_train2017.json), which can be used in same format as Human-Art

- the ones end with \_humanart (e.g. training\_humanart.json) is the annotation json files of Human-Art

- the ones end with \_humanart\_coco (e.g. training\_humanart\_coco.json) is the annotation json files of the assemble of COCO and Human-Art

- the ones end with \_humanart\_\[scenario\] (e.g. training\_humanart_cartoon.json) is the annotation json files of one specific scenario of Human-Art

- HumanArt_validation_detections_AP_H_56_person.json is the detection results with an AP of 56 for the evaluation of top-down pose estimation models (similar with COCO_val2017_detections_AP_H_56_person.json in MSCOCO)


The annotation json files of Human-Art is described as follows:

```python
{
    "info":{xxx}, # some basic information of Human-Art
    "images":[
        {
            "file_name": "xxx" # the path of the image (same definition with COCO)
            "height": xxx, # the image height (same definition with COCO)
            "width": xxx, # the image width (same definition with COCO)
            "id": xxx, # the image id (same definition with COCO)
            "page_url": "xxx", # the web link of the page containing the image
            "image_url": "xxx", # the web link of the image
            "picture_name": "xxx", # the name of the image
            "author": "xxx", # the author of the image
            "description": "xxx", # the text description of the image
            "category": "xxx"  # the scenario of the image (e.g. cartoon)
        },
        ...
    ],
    "annotations":[
        {
            "keypoints":[xxx], # 17 COCO keypoints' position (same definition with COCO)
            "keypoints_21":[xxx], # 21 Human-Art keypoints' position 
            "self_contact": [xxx], # self contact keypoints, x1,y1,x2,y2...
            "num_keypoints": xxx, # annotated keypoints (not invisible) in 17 COCO format keypoints (same definition with COCO)
            "num_keypoints_21": xxx, # annotated keypoints (not invisible) in 21 Human-Art format keypoints 
            "iscrowd": xxx, # annotated or not (same definition with COCO)
            "image_id": xxx, # the image id (same definition with COCO)
            "area": xxx, # the human area (same definition with COCO)
            "bbox": [xxx], # the human bounding box (same definition with COCO)
            "category_id": 1, # category id=1 means it is a person category  (same definition with COCO)
            "id": xxx, # annotation id (same definition with COCO)
            "annotator": xxx # annotator id
        }
    ],
    "categories":[] # category infromation (same definition with COCO)
}
```

## Human Pose Estimation


Human pose estimators trained on Human-Art is now supported in [MMPose](https://github.com/open-mmlab/mmpose) in this [pr](https://github.com/open-mmlab/mmpose/pull/2304). The detailed usage and Model Zoo can be found in MMPose's documents: [(1) ViTPose](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/humanart/vitpose_humanart.md), [(2) HRNet](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/topdown_heatmap/humanart/hrnet_humanart.md), and [(3) RTMPose](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/humanart/rtmpose_humanart.md).


To train and evaluate human pose estimators, please refer to  [MMPose](https://github.com/open-mmlab/mmpose). Due to the frequent update of MMPose, we do not maintain a codebase in this repo. Since Human-Art is compatible with MSCOCO, you can train and evaluate any model in MMPose using its dataloader. 

The supported model include (xx-coco means trained on MSCOCO only and xx-humanart-coco means trained on Human-Art and MSCOCO):

Results of ViTPose on Human-Art validation dataset with ground-truth bounding-box

> With classic decoder

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [ViTPose-S-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py) |  256x192   | 0.507 |      0.758      |      0.531      | 0.551 |      0.780      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.json) |
| [ViTPose-S-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-small_8xb64-210e_humanart-256x192.py) |  256x192   | 0.738 |      0.905      |      0.802      | 0.768 |      0.911      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.json) |
| [ViTPose-B-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py) |  256x192   | 0.555 |      0.782      |      0.590      | 0.599 |      0.809      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.json) |
| [ViTPose-B-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-base_8xb64-210e_humanart-256x192.py) |  256x192   | 0.759 |      0.905      |      0.823      | 0.790 |      0.917      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.json) |
| [ViTPose-L-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py) |  256x192   | 0.637 |      0.838      |      0.689      | 0.677 |      0.859      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.json) |
| [ViTPose-L-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-base_8xb64-210e_humanart-256x192.py) |  256x192   | 0.789 |      0.916      |      0.845      | 0.819 |      0.929      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-large_8xb64-210e_humanart-256x192-9aba9345_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-large_8xb64-210e_humanart-256x192-9aba9345_20230614.json) |
| [ViTPose-H-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py) |  256x192   | 0.665 |      0.860      |      0.715      | 0.701 |      0.871      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.json) |
| [ViTPose-H-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-huge_8xb64-210e_humanart-256x192.py) |  256x192   | 0.800 |      0.926      |      0.855      | 0.828 |      0.933      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-huge_8xb64-210e_humanart-256x192-603bb573_20230612.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-huge_8xb64-210e_humanart-256x192-603bb573_20230612.json) |


Results of HRNet on Human-Art validation dataset with ground-truth bounding-box

> With classic decoder

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32-coco](configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) |  256x192   | 0.533 |      0.771      |      0.562      | 0.574 |      0.792      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192_20220909.log) |
| [pose_hrnet_w32-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w32_8xb64-210e_humanart-256x192.py) |  256x192   | 0.754 |      0.906      |      0.812      | 0.783 |      0.916      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.json) |
| [pose_hrnet_w48-coco](configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py) |  256x192   | 0.557 |      0.782      |      0.593      | 0.595 |      0.804      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192_20220913.log) |
| [pose_hrnet_w48-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w48_8xb32-210e_humanart-256x192.py) |  256x192   | 0.769 |      0.906      |      0.825      | 0.796 |      0.919      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w48_8xb32-210e_humanart-256x192-05178983_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w48_8xb32-210e_humanart-256x192-05178983_20230614.json) |


Results of RTM-Pose on Human-Art validation dataset with ground-truth bounding-box 

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [rtmpose-t-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py) |  256x192   | 0.444 |      0.725      |      0.453      | 0.488 |      0.750      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.json) |
| [rtmpose-t-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-t_8xb256-420e_humanart-256x192.py) |  256x192   | 0.655 |      0.872      |      0.720      | 0.693 |      0.890      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.json) |
| [rtmpose-s-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   | 0.480 |      0.739      |      0.498      | 0.521 |      0.763      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.json) |
| [rtmpose-s-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-s_8xb256-420e_humanart-256x192.py) |  256x192   | 0.698 |      0.893      |      0.768      | 0.732 |      0.903      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.json) |
| [rtmpose-m-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   | 0.532 |      0.765      |      0.563      | 0.571 |      0.789      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.json) |
| [rtmpose-m-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-m_8xb256-420e_humanart-256x192.py) |  256x192   | 0.728 |      0.895      |      0.791      | 0.759 |      0.906      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.json) |
| [rtmpose-l-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   | 0.564 |      0.789      |      0.602      | 0.599 |      0.808      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.json) |
| [rtmpose-l-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-l_8xb256-420e_humanart-256x192.py) |  256x192   | 0.753 |      0.905      |      0.812      | 0.783 |      0.915      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.json) |


## Human Detection


Human detectors trained on Human-Art is now supported in [MMPose](https://github.com/open-mmlab/mmpose) in this [pr](https://github.com/open-mmlab/mmpose/pull/2724). The detailed usage and Model Zoo can be found [here](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose).


To train and evaluate human detectors, please refer to  [MMDetection](https://github.com/open-mmlab/mmdetection), which is an open source object detection toolbox based on PyTorch that support diverse detection frameworks with higher efficiency and higher accuracy. Due to the frequent update of MMDetection, we do not maintain a codebase in this repo. Since Human-Art is compatible with MSCOCO, you can train and evaluate any model in MMDetection using its dataloader.

The supported model include:


|       Detection Config        |  Model AP<sup><br> |      Download        |
| :---------------------------: | :---------------------------: | :---------------------------: |
| [RTMDet-tiny](./rtmdet/person/rtmdet_tiny_8xb32-300e_humanart.py) |            46.6        | [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_tiny_8xb32-300e_humanart-7da5554e.pth) |
| [RTMDet-s](./rtmdet/person/rtmdet_s_8xb32-300e_humanart.py) |          50.6              | [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_s_8xb32-300e_humanart-af5bd52d.pth) |
| [YOLOX-nano](./yolox/humanart/yolox_nano_8xb8-300e_humanart.py) |        38.9          | [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/yolox_nano_8xb8-300e_humanart-40f6f0d0.pth) |
| [YOLOX-tiny](./yolox/humanart/yolox_tiny_8xb8-300e_humanart.py) |             47.7              |         [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/yolox_tiny_8xb8-300e_humanart-6f3252f9.pth) |
| [YOLOX-s](./yolox/humanart/yolox_s_8xb8-300e_humanart.py) |             54.6              |     [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/yolox_s_8xb8-300e_humanart-3ef259a7.pth) |
| [YOLOX-m](./yolox/humanart/yolox_m_8xb8-300e_humanart.py) |            59.1              |      [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/yolox_m_8xb8-300e_humanart-c2c7a14a.pth) |
| [YOLOX-l](./yolox/humanart/yolox_l_8xb8-300e_humanart.py) |           60.2              |       [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/yolox_l_8xb8-300e_humanart-ce1d7a62.pth) |
| [YOLOX-x](./yolox/humanart/yolox_x_8xb8-300e_humanart.py) |             61.3              |     [Det Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/yolox_x_8xb8-300e_humanart-a39d44ed.pth) |





## Citing Human-Art

If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@inproceedings{ju2023human,
    title={Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
    author={Ju, Xuan and Zeng, Ailing and Wang, Jianan and Xu, Qiang and Zhang, Lei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023},
}
```
