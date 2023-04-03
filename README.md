<br/>

<div align="center">
    <img src="resources/logo.jpg" width="600"/>
</div>

<br/>


# Human-Art

This repository contains the implementation of the following paper:
> **Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes** [[Project Page]](https://idea-research.github.io/HumanArt/) [[Paper]](https://arxiv.org/abs/2303.02760) [[Code]](https://github.com/IDEA-Research/HumanArt) [[Data]](https://forms.gle/UVv1GiNJNQsE4qif7) <br>
> [Xuan Ju](https://juxuan.space/)<sup>∗12</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>∗1</sup>, [Jianan Wang](https://github.com/wendyjnwang/)<sup>1</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
> <sup>∗</sup> Equal contribution <sup>1</sup>International Digital Economy Academy <sup>2</sup>The Chinese University of Hong Kong



**Table of Contents**

  1. [General Description](#general-description)
  2. [Features](#features)
  3. [Dataset Download](#dataset-download)
  4. [Human Detection](#human-detection)
  5. [Human Pose Estimation](#human-pose-estimation)
  6. [Citing Human-Art](#citing-human-art)
  7. [Acknowledgement](#acknowledgement)
  8. [Reference](#reference)



## General Description


<div  align="center">    
<img src="resources/dataset_overview.png" width="90%">
</div>


This paper proposes a large-scale dataset, Human-Art, that targets multi-scenario human-centric tasks to bridge the gap between natural and artificial scenes. It includes twenty high-quality human scenes, including natural and artificial humans in both 2D representation (yellow dashed boxes) and 3D representation (blue solid boxes).

**Contents of Human-Art:**

- `50,000` images including human figures in `20 scenarios` (5 natural scenarios, 3 2D artificial scenarios, and 12 2D artificial scenarios)
- Human-centric annotations include `human bounding box`, `21 2D human keypoints`, `human self-contact keypoints`, and `description text`
- baseline human detector and human pose estimator trained on the assembly of MSCOCO[<sup>[1]</sup>](#refer-anchor-1) and Human-Art

**Tasks that Human-Art targets for:**

- multi-scenario `human detection`, `2D human pose estimation`, and `3D human mesh recovery`
- multi-scenario `human image generation` (especially `controllable` human image generation, e.g. with conditions such as pose and text)
- `out-of-domain` human detection and human pose estimation



## Dataset Download


Under the CC-license, Human-Art is available for download. Fill out [this form](https://forms.gle/UVv1GiNJNQsE4qif7) to request authorization to use Human-Art for non-commercial purposes. After you submit the form, an email containing the dataset will be instantly delivered to you.

For convenience of usage, Human-Art is processed using the same format as MSCOCO[<sup>[1]</sup>](#refer-anchor-1). Please save the dataset with the following file structure after downloading (we also include the file structure of COCO because we use it for assembly training of COCO and Human-Art):

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

## Human Detection

You need to use the following clone command to include sub module mmdetection

```shell
git clone --recurse-submodules git@github.com:IDEA-Research/HumanArt.git
```

For ease of use, we employ [MMDetection](https://github.com/open-mmlab/mmdetection) as the code base of human detection. MMDetection is an open source object detection toolbox based on PyTorch that support diverse detection frameworks with higher efficiency and higher accuracy.

To train and evaluate human detection, you can directly edit config files in `mmdetection/configs`. We have provided one example in `mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_humanart.py`. Future releases will include more models.

The supported model include:

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf">Faster R-CNN (CVPR'2019)</a></summary>

```bibtex
@article{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  journal={Advances in neural information processing systems},
  volume={28},
  year={2015}

```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2303.02760">Human-Art (CVPR'2023)</a></summary>

```bibtex
@inproceedings{ju2023human,
  title={Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
  author={Ju, Xuan and Zeng, Ailing and Wang, Jianan and Xu, Qiang and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```

</details>

Results on Human-Art validation set

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | human box AP |                                                          Config                                                           |                                                                                                                                                                          Download                                                                                                                                                                           |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |   4.0    |      21.4      |  44.2  |    [config](mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_humanart.py)     |                          [model](https://drive.google.com/file/d/16gBKkNlLNdJfKeagV-KBirJBk3Ocvd7Y/view?usp=share_link)   

Results on COCO val2017

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | human box AP |                                                          Config                                                           |                                                                                                                                                                          Download                                                                                                                                                                           |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |   4.0    |      21.4      |  51.6  |    [config](mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_humanart.py)     |                          [model](https://drive.google.com/file/d/16gBKkNlLNdJfKeagV-KBirJBk3Ocvd7Y/view?usp=share_link)  


## Human Pose Estimation

You need to use the following clone command to include sub module mmpose

```shell
git clone --recurse-submodules git@github.com:IDEA-Research/HumanArt.git
```

For ease of use, we employ [MMPose](https://github.com/open-mmlab/mmpose) as the code base of human pose estimation. MMPose is an open-source toolbox for pose estimation based on PyTorch that support diverse tasks, various models and datasets with higher efficiency and higher accuracy.

To train and evaluate human pose estimator, you can directly edit config files in `mmpose/configs`. We have provided one example for both bottom-up algorithm and top-down algorithm (in `mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/humanart` and `mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/humanart`). Future releases will include more models. You can also train your own model following the edition shown in our example.

The supported models include:

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2303.02760">Human-Art (CVPR'2023)</a></summary>

```bibtex
@inproceedings{ju2023human,
  title={Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
  author={Ju, Xuan and Zeng, Ailing and Wang, Jianan and Xu, Qiang and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```

</details>

Results on Human-Art validation set with ground-truth bounding box

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | 
| [pose_hrnet_w48](mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/humanart/hrnet_w48_humanart_256x192.py) |  256x192   | 0.764 |      0.906      |      0.824      | 0.794 |      0.918      | [ckpt](https://drive.google.com/file/d/1gs1RCxRcItUHwA5N8P5_9mKcgwLiBjOO/view?usp=share_link) | 


Results on COCO val2017 set with ground-truth bounding box

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |   
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | 
| [pose_hrnet_w48](mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/humanart/hrnet_w48_humanart_256x192.py) |  256x192   | 0.772 |      0.936      |      0.847      | 0.800 |      0.942      | [ckpt](https://drive.google.com/file/d/1gs1RCxRcItUHwA5N8P5_9mKcgwLiBjOO/view?usp=share_link) | 


<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1611.05424">Associative Embedding (NIPS'2017)</a></summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_HigherHRNet_Scale-Aware_Representation_Learning_for_Bottom-Up_Human_Pose_Estimation_CVPR_2020_paper.html">HigherHRNet (CVPR'2020)</a></summary>

```bibtex
@inproceedings{cheng2020higherhrnet,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Cheng, Bowen and Xiao, Bin and Wang, Jingdong and Shi, Honghui and Huang, Thomas S and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5386--5395},
  year={2020}
}
```


</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html">UDP (CVPR'2020)</a></summary>

```bibtex
@InProceedings{Huang_2020_CVPR,
  author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
  title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2303.02760">Human-Art (CVPR'2023)</a></summary>

```bibtex
@inproceedings{ju2023human,
  title={Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
  author={Ju, Xuan and Zeng, Ailing and Wang, Jianan and Xu, Qiang and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```

</details>

Results on Human-Art validation set without multi-scale test

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      | 
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | 
| [HigherHRNet-w48_udp](mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/humanart/higherhrnet_w48_humanart_512x512_udp.py) |  512x512   | 0.664 |      0.851      |      0.709      | 0.740 |      0.880      | [ckpt](https://drive.google.com/file/d/1fqPy44k-bYb6XqJXMZ5b9gVTzcIjOlPQ/view?usp=share_link) | 


Results on COCO val2017 without multi-scale test

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      | 
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | 
| [HigherHRNet-w48_udp](mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/humanart/higherhrnet_w48_humanart_512x512_udp.py) |  512x512   | 0.668 |      0.860      |      0.726      | 0.718 |      0.886      | [ckpt](https://drive.google.com/file/d/1fqPy44k-bYb6XqJXMZ5b9gVTzcIjOlPQ/view?usp=share_link) |


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

## Acknowledgement

This work was supported in part by the Shenzhen-Hong Kong-Macau Science and Technology Program (Category C) of the Shenzhen Science Technology and Innovation Commission under Grant No. SGDX2020110309500101 and in part by Research Matching Grant CSE-7-2022.



## Reference

<div id="refer-anchor-1"></div>
[1] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In European Conference on Computer Vision (ECCV), pages 740–755. Springer, 2014.
