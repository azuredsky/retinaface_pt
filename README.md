# RetinaFace

Here we have retinaface training code with pytorch, and also TensorRT inference code on retinaface model. This model is continue upgrade and fix code. Currently we have these functionality implemented:

- [x] Training on wider face;
- [x] Inference retinaface with mobilenet with TensorRT in pure C++;
- [x] python inference on pytorch speed issue needs fix;
- [ ] root reason for why img_folder inference slow than video inference????  Now using video we can get a normal speed.
- [x] convert pytorch model to onnx.



## Updates

- **2019.09.23**: Now we add a standalone onnxruntime inference demo! You can using onnxruntime-gpu or onnxruntime to do inference on our onnx model. It runs about 40ms on CPU!!! Amazing fast! With acceleration of GPU, it would got massive speed enhancement!



## Install

To running python inference, simply install pytorch, we haven't match the theoritically speed of retinaface, **if anyone tested matchs paper speed pls report to us!**.

To install TensorRT inference code, you should:

- install [thor](https://github.com/jinfagang/thor) lib first.
- Donwload TensorRT 5.1.xx and softlink it to `~/TensorRT`, the cmake will found TensorRT default from `~/TensorRT`;

```
cd trt_cc
mkdir build
cd build && cmake ..
make -j8
```



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
