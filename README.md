# RetinaFace

Here we have retinaface training code with pytorch, and also TensorRT inference code on retinaface model. This model is continue upgrade and fix code. Currently we have these functionality implemented:

- [x] Training on wider face;
- [x] Inference retinaface with mobilenet with TensorRT in pure C++;
- [x] python inference on pytorch speed issue needs fix;
- [x] root reason for why img_folder inference slow than video inference????  Now using video we can get a normal speed.
- [x] convert pytorch model to onnx.



Some todos here:

- [ ] Add a hand keypoint detection using retinaface, which have BoundingBox and  20 keypoints;



## Updates

- **2019.11.01**: 250fps TensorRT inference on retinaface has been opensource!! checkout on our platform: http://manaai.cn
- **2019.09.23**: Now we add a standalone onnxruntime inference demo! You can using onnxruntime-gpu or onnxruntime to do inference on our onnx model. It runs about 40ms on CPU!!! Amazing fast! With acceleration of GPU, it would got massive speed enhancement!



## Install

To running python inference, simply install pytorch. Also:

````
sudo pip3 install alfred-py
````

To install TensorRT inference code, you should:

- install [thor](https://github.com/jinfagang/thor) lib first.
- Donwload TensorRT 5.1.xx and softlink it to `~/TensorRT`, the cmake will found TensorRT default from `~/TensorRT`;

```
cd trt_cc
mkdir build
cd build && cmake ..
make -j8
```

**note**: the caffe version is deprecated, we have brand new onnx version now! checkout: http://manaai.cn



## Train

- a). Train on wider face.

  To train on wider face, you should download data from: https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0 , it was a single txt format contains both face boxes and landmarks:

  ```
  # 0--Parade/0_Parade_marchingband_1_849.jpg
  449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82
  # 0--Parade/0_Parade_Parade_0_904.jpg
  361 98 263 339 424.143 251.656 0.0 547.134 232.571 0.0 494.121 325.875 0.0 453.83 368.286 0.0 561.978 342.839 0.0 0.89
  # 0--Parade/0_Parade_marchingband_1_799.jpg
  78 221 7 8 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 0.2
  78 238 14 17 84.188 244.607 1.0 89.527 244.491 1.0 86.973 247.857 1.0 85.116 250.643 1.0 88.482 250.643 1.0 0.36
  113 212 11 15 117.0 220.0 0.0 122.0 220.0 0.0 119.0 222.0 0.0 118.0 225.0 0.0 122.0 225.0 0.0 0.3
  134 260 15 15 142.0 265.0 0.0 146.0 265.0 0.0 145.0 267.0 0.0 142.0 272.0 0.0 146.0 271.0 0.0 0.24
  163 250 14 17 169.357 256.5 1.0 175.25 257.143 1.0 172.357 260.786 1.0 170.214 262.929 1.0 174.071 262.821 1.0 0.41
  201 218 10 12 203.0 222.0 0.0 207.0 223.0 0.0 204.0 225.0 0.0 203.0 227.0 0.0 206.0 227.0 0.0 0.22
  182 266 15 17 189.723 271.487 0.0 195.527 271.152 0.0 193.741 274.835 0.0 191.062 278.406 0.0 194.969 278.295 0.0 0.32
  245 279 18 15 254.116 281.973 1.0 259.571 281.277 1.0 257.83 284.411 1.0 255.161 287.661 1.0 259.223 287.08 1.0 0.26
  ```

  once you got label, place it into `data/widerface/train`, and images place to `./data/widerface/train/images/0--Parade/0_Parade_marchingband_1_849.jpg` like this. then it will be found automatically.

  If you want train on any other dataset, simply change it into this format.

- b) Train on HandPose data.

  this is to be done.



## Run

to run python inference:

```
python3 demo_video.py --img_folder /path/to/img/folder
```

to run trt inference:

```
./retinaface /a/video.mp4
```





## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
