### <p align="center">FishDreamer: Towards Fisheye Semantic Completion via Unified Image Outpainting and Segmentation
<br>
<div align="center">
  <a href="https://www.researchgate.net/profile/Shi-Hao-10" target="_blank">Hao&nbsp;Shi</a> <b>&middot;</b>
  Yu&nbsp;Li</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Kailun-Yang" target="_blank">Kailun&nbsp;Yang</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Jiaming-Zhang-10" target="_blank">Jiaming&nbsp;Zhang</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Kunyu-Peng" target="_blank">Kunyu&nbsp;Peng</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Alina-Roitberg-2" target="_blank">Alina&nbsp;Roitberg</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Yaozu-Ye" target="_blank">Yaozu&nbsp;Ye</a> <b>&middot;</b>
  Huajian&nbsp;Ni</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Kaiwei-Wang-4" target="_blank">Kaiwei&nbsp;Wang</a> <b>&middot;</b>
  <a href="https://www.researchgate.net/profile/Rainer-Stiefelhagen" target="_blank">Rainer&nbsp;Stiefelhagen</a>
  <br> <br>

  <a href="https://arxiv.org/pdf/2303.13842.pdf" target="_blank">Paper</a>

####
</div>
<br>

<div align=center><img src="assets/teaser.png" width="661" height="777" /></div>


### Update
- 2023.03.20 Init repository.
- 2023.03.24 Release the [arXiv](https://arxiv.org/abs/2303.13842) version.
- 2023.04.05 :rocket: FishDreamer has been accepted to 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW OmniCV2023).
- 2023.09.05 Release code.

### Abstract
This paper raises the new task of Fisheye Semantic Completion (FSC), where dense texture, structure, and semantics of a fisheye image are inferred even beyond the sensor field-of-view (FoV).
Fisheye cameras have larger FoV than ordinary pinhole cameras, yet its unique special imaging model naturally leads to a blind area at the edge of the image plane.
This is suboptimal for safety-critical applications since   important perception tasks, such as semantic segmentation, become very challenging within the blind zone.
Previous works  considered the out-FoV outpainting  and in-FoV segmentation separately. 
However, we observe that these two tasks are actually closely coupled. 
To jointly estimate the tightly intertwined complete fisheye image and scene semantics, we introduce the new FishDreamer which relies on successful ViTs enhanced with a novel Polar-aware Cross Attention module (PCA)  to leverage dense context and guide semantically-consistent content generation while considering different polar distributions.
In addition to the contribution of the novel task and architecture, we also derive Cityscapes-BF and KITTI360-BF datasets to facilitate training and evaluation of this new track. Our experiments demonstrate that the proposed FishDreamer outperforms methods solving each task in isolation and surpasses alternative approaches on the Fisheye Semantic Completion. 

## Method

<p align="center">
    (Overview)
</p>
<p align="center">
    <div align=center><img src="assets/method.png" width="800" height="674" /></div>
<br><br>

## Results
<div align=center><img src="assets/compare.png" width="800" height="416" /></div>

After code reorganization, we retrained FishDreamer on Cityscapes-BF:

| Method     | PSNR | mIoU |
| :--------- | :----------: | :------------: |
| _FishDreamer (Paper)_ | _25.05_ | _54.54_ |
| FishDreamer (This Repo) | **25.21** | **54.88** |

### Dependencies
This repo has been tested in the following environment:
```angular2html
torch == 1.9.0
pytorch-lightning == 1.8.6
mmcv-full == 1.5.2
```

### Usage
To train FishDreamer, first set environment variable:
```angular2html
export USER=$(whoami)
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```
Then start training:
```angular2html
python3 bin/train.py \
-cn FishDreamer \
data.batch_size=3 \
trainer.kwargs.max_epochs=70 \
data.train.transform_variant=resize
```

To output visual result, run:
```angular2html
python3 bin/predict.py \
model.path=$ModelPath \
indir=$DataPath \
outdir=$OutPath
```
You can eval your result by:
```angular2html
python3 bin/evaluate_predicts.py \
config=$ConfigPath \
datadir=$DataPath \
predictdir=$OutPath \
outpath=$OutMetricPath
```

### Pretrained Models & Dataset
The pretrained model and Cityscapes-BF dataset can be found there:
```angular2html
https://share.weiyun.com/7ShuPa2Y
```
For KITTI360-BF, please follow the instruction of [FisheyeEX](https://arxiv.org/pdf/2206.05844.pdf).

### Citation

   If you find our paper or repo useful, please consider citing our paper:

   ```bibtex
@inproceedings{shi2023fishdreamer,
  title={FishDreamer: Towards Fisheye Semantic Completion via Unified Image Outpainting and Segmentation},
  author={Shi, Hao and Li, Yu and Yang, Kailun and Zhang, Jiaming and Peng, Kunyu and Roitberg, Alina and Ye, Yaozu and Ni, Huajian and Wang, Kaiwei and Stiefelhagen, Rainer},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6433--6443},
  year={2023}
}
   ```

### Acknowledgement
This project would not have been possible without the following outstanding repositories:

[LaMa](https://github.com/advimman/lama), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)

### Devs
Hao Shi, Yu Li

### Contact
Feel free to contact me if you have additional questions or have interests in collaboration. Please drop me an email at haoshi@zju.edu.cn. =)
