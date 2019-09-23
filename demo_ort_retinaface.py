"""

Standalone inference with onnxruntime
it doesn't need pytorch
using onnxruntime can directly do inference

"""
import onnxruntime
from PIL import Image
import numpy as np
from alfred.vis.image.get_dataset_label_map import coco_label_map_list
from alfred.utils.log import logger as logging
import cv2
import time
from math import ceil
from itertools import product as product
import os
import sys


img_w = 1024
img_h = 678


class PriorBox(object):
    def __init__(self, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        anchors = np.reshape(anchors, (-1, 4))
        return np.array(anchors)

priorbox = PriorBox(image_size=(678, 1024))
priors = priorbox.forward()
print('priors.shape: ', priors.shape)


def decode_box(loc, priors, variances=[0.1, 0.2]):
    boxes = np.hstack([priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances=[0.1, 0.2]):
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def preprocess(image):
    image = cv2.resize(image, (img_w, img_h))
    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])
    mean_vec = np.array([104, 117, 123])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]
    image = np.array([image])
    return image


logging.info('preparing...')
logging.info('onnxruntime supported devices: {}'.format(onnxruntime.get_device()))
session = onnxruntime.InferenceSession('retinaface_mbv2_sim.onnx')
logging.info('onnx session loaded.')

# data_f = 'images/0--Parade_0_Parade_marchingband_1_657.jpg'
data_f = '/media/fagangjin/wd/permanent/datasets/TestVideos/ellenshow.mp4'
if os.path.basename(data_f).split('.')[-1] == 'mp4':
    cap = cv2.VideoCapture(data_f)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img = np.array(frame, dtype=np.float32)
            img_data = preprocess(img)
            tic = time.time()
            out1, out2, out3 = session.run(None, {session.get_inputs()[0].name: img_data})
            logging.info('finished in: {}'.format(time.time() - tic))

            scores = np.squeeze(out2, axis=0)[:, 1]
            boxes = np.squeeze(out1, axis=0)
            landmarks = np.squeeze(out3, axis=0)
            inds = np.where(scores > 0.8)[0]
            boxes = decode_box(boxes, priors)
            boxes *= [img_w, img_h, img_w, img_h]
            scores = scores[inds]
            boxes = boxes[inds]
            landmarks = landmarks[inds]

            ori_img_resized = cv2.resize(frame, (img_w, img_h))
            for b in boxes:
                b = np.array(b, dtype=np.int)
                cv2.rectangle(ori_img_resized, (b[0], b[1]),
                                        (b[2], b[3]), (0, 0, 255), 2)
            cv2.imshow('rr', ori_img_resized)
            cv2.waitKey(1)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
else:
    ori_img = cv2.imread(data_f)
    img = np.array(ori_img, dtype=np.float32)
    img_data = preprocess(img)
    tic = time.time()
    out1, out2, out3 = session.run(None, {session.get_inputs()[0].name: img_data})
    logging.info('finished in: {}'.format(time.time() - tic))

    scores = np.squeeze(out2, axis=0)[:, 1]
    boxes = np.squeeze(out1, axis=0)
    landmarks = np.squeeze(out3, axis=0)
    inds = np.where(scores > 0.8)[0]
    boxes = decode_box(boxes, priors)
    boxes *= [img_w, img_h, img_w, img_h]
    scores = scores[inds]
    boxes = boxes[inds]
    landmarks = landmarks[inds]

    ori_img_resized = cv2.resize(ori_img, (img_w, img_h))
    for b in boxes:
        b = np.array(b, dtype=np.int)
        print(b)
        cv2.rectangle(ori_img_resized, (b[0], b[1]),
                                (b[2], b[3]), (0, 0, 255), 2)
    cv2.imshow('rr', ori_img_resized)
    cv2.waitKey(0)